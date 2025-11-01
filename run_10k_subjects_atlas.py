"""
10,000 Subject Simulation with Real Atlases (HarvardOxford or HCP-MMP1)
Uses actual atlas connectivity matrices with subject-specific variations
"""

import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
import time

from tvb.simulator.lab import *
from tvb.datatypes import connectivity


def create_subject_specific_connectivity(atlas_type, subject_id, connectivity_dir):
    """
    Create subject-specific connectivity with individual variability

    Parameters:
    -----------
    atlas_type : str
        'harvard_oxford' or 'hcpmmp1'
    subject_id : int
        Subject identifier (used as random seed)
    connectivity_dir : str
        Directory containing atlas connectivity files

    Returns:
    --------
    conn : Connectivity
        Subject-specific TVB connectivity
    """
    np.random.seed(subject_id)  # Reproducible per subject
    conn_dir = Path(connectivity_dir)

    # Load atlas-specific connectivity
    if atlas_type == 'harvard_oxford':
        weights = np.load(conn_dir / 'ho_weights_48.npy')
        tract_lengths = np.load(conn_dir / 'ho_tract_lengths_48.npy')
        centers = np.load(conn_dir / 'ho_centers_48.npy')

        metadata_file = conn_dir / 'ho_connectivity_48_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                region_labels = np.array(metadata.get('regions', [f"ROI_{i}" for i in range(48)]))
        else:
            region_labels = np.array([f"HO_ROI_{i}" for i in range(48)])

        num_nodes = 48

    elif atlas_type == 'hcpmmp1':
        weights = np.load(conn_dir / 'mmp_weights_360.npy')
        tract_lengths = np.load(conn_dir / 'mmp_tract_lengths_360.npy')
        centers = np.load(conn_dir / 'mmp_centers_360.npy')

        metadata_file = conn_dir / 'mmp_connectivity_360_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                region_labels = np.array(metadata.get('regions', [f"ROI_{i}" for i in range(360)]))
        else:
            region_labels = np.array([f"MMP_ROI_{i}" for i in range(360)])

        num_nodes = 360

    else:
        raise ValueError(f"Unknown atlas type: {atlas_type}")

    # Add subject-specific variability (±20%)
    variability = 0.2
    subject_variation = 1.0 + np.random.uniform(-variability, variability, weights.shape)
    weights_subject = weights * subject_variation

    # Ensure non-negative and symmetric
    weights_subject = np.maximum(weights_subject, 0)
    weights_subject = (weights_subject + weights_subject.T) / 2

    # Create TVB connectivity
    conn = connectivity.Connectivity(
        weights=weights_subject,
        tract_lengths=tract_lengths,
        region_labels=region_labels,
        centres=centers,
        speed=np.array([3.0])
    )
    conn.configure()

    return conn, num_nodes


def simulate_single_subject(args):
    """
    Run simulation for a single subject

    Parameters:
    -----------
    args : tuple
        (subject_id, atlas_type, connectivity_dir, num_timepoints, TR, model_type, output_dir)

    Returns:
    --------
    result : dict
        Simulation result summary
    """
    subject_id, atlas_type, connectivity_dir, num_timepoints, TR, model_type, output_dir = args

    try:
        start_time = time.time()

        # Create subject directory
        subject_dir = Path(output_dir) / f"subject_{subject_id:05d}"
        subject_dir.mkdir(parents=True, exist_ok=True)

        # Setup subject-specific connectivity
        conn, num_nodes = create_subject_specific_connectivity(
            atlas_type, subject_id, connectivity_dir
        )

        # Setup model
        if model_type == 'wong_wang':
            model = models.ReducedWongWang()
        elif model_type == 'kuramoto':
            model = models.Kuramoto()
        elif model_type == 'generic_2d_oscillator':
            model = models.Generic2dOscillator()

        # Setup coupling
        coupling_func = coupling.Linear(a=np.array([0.0152]))

        # Setup integrator with subject-specific noise
        np.random.seed(subject_id)
        heunint = integrators.HeunStochastic(
            dt=2**-4,
            noise=noise.Additive(nsig=np.array([0.001]))
        )

        # Setup monitors
        mon_tavg = monitors.TemporalAverage(period=TR * 1000.0)
        mon_bold = monitors.Bold(period=TR * 1000.0)

        # Initialize simulator
        sim = simulator.Simulator(
            model=model,
            connectivity=conn,
            coupling=coupling_func,
            integrator=heunint,
            monitors=[mon_tavg, mon_bold]
        )
        sim.configure()

        # Run simulation
        simulation_length = num_timepoints * TR * 1000.0  # ms
        results = {}

        for data in sim(simulation_length=simulation_length):
            for i, monitor in enumerate(sim.monitors):
                monitor_name = monitor.__class__.__name__
                if data[i] is not None:
                    if monitor_name not in results:
                        results[monitor_name] = []
                    results[monitor_name].append(data[i])

        # Process and save results
        for key in results:
            concatenated = np.concatenate([x[1] for x in results[key] if x is not None])

            # Extract first state variable
            if len(concatenated.shape) == 4:
                ts_data = concatenated[:, :, 0, 0]
            elif len(concatenated.shape) == 3:
                ts_data = concatenated[:, :, 0]
            else:
                ts_data = concatenated

            # Truncate/pad to expected timepoints
            if ts_data.shape[0] > num_timepoints:
                ts_data = ts_data[:num_timepoints, :]
            elif ts_data.shape[0] < num_timepoints:
                padded = np.zeros((num_timepoints, num_nodes))
                padded[:ts_data.shape[0], :] = ts_data
                ts_data = padded

            # Transpose to (ROI, timepoints)
            ts_data = ts_data.T

            # Save
            suffix = 'neural' if key == 'TemporalAverage' else 'bold'
            save_path = subject_dir / f"sim_{atlas_type}_{num_nodes}nodes_{suffix}.npy"
            np.save(save_path, ts_data)

        # Save metadata
        metadata = {
            'subject_id': subject_id,
            'atlas': atlas_type,
            'num_nodes': num_nodes,
            'num_timepoints': num_timepoints,
            'TR': TR,
            'model_type': model_type,
            'seed': subject_id,
            'timestamp': str(datetime.now())
        }

        metadata_path = subject_dir / f"subject_{subject_id:05d}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        elapsed_time = time.time() - start_time

        return {
            'subject_id': subject_id,
            'status': 'success',
            'atlas': atlas_type,
            'num_nodes': num_nodes,
            'output_dir': str(subject_dir),
            'elapsed_time': elapsed_time,
            'seed': subject_id
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            'subject_id': subject_id,
            'status': 'failed',
            'atlas': atlas_type,
            'error': str(e),
            'elapsed_time': elapsed_time
        }


def run_10k_subjects(
    atlas_type='harvard_oxford',
    num_subjects=10000,
    num_timepoints=375,
    TR=0.8,
    model_type='wong_wang',
    connectivity_dir=None,
    output_dir='results/10k_subjects',
    num_workers=32,
    start_subject=0
):
    """
    Run 10,000 subject simulations with real atlas

    Parameters:
    -----------
    atlas_type : str
        'harvard_oxford' or 'hcpmmp1'
    num_subjects : int
        Number of subjects to simulate
    num_timepoints : int
        Timepoints per simulation
    TR : float
        Repetition time in seconds
    model_type : str
        Neural mass model type
    connectivity_dir : str
        Directory containing atlas connectivity files
    output_dir : str
        Base output directory
    num_workers : int
        Number of parallel workers
    start_subject : int
        Starting subject ID (for resuming)
    """

    # Set default connectivity directory
    if connectivity_dir is None:
        if atlas_type == 'harvard_oxford':
            connectivity_dir = 'data/HarvardOxford'
        elif atlas_type == 'hcpmmp1':
            connectivity_dir = 'data/HCPMMP1'

    # Validate connectivity exists
    conn_dir = Path(connectivity_dir)
    if atlas_type == 'harvard_oxford':
        required_file = conn_dir / 'ho_weights_48.npy'
    else:
        required_file = conn_dir / 'mmp_weights_360.npy'

    if not required_file.exists():
        raise FileNotFoundError(
            f"Connectivity file not found: {required_file}\n"
            f"Run the appropriate loader first:\n"
            f"  HarvardOxford: python src/harvard_oxford_loader.py --method distance\n"
            f"  HCP-MMP1: python src/hcpmmp1_loader.py --method distance"
        )

    print("="*80)
    print(f"10,000 SUBJECT SIMULATION - {atlas_type.upper()}")
    print("="*80)
    print(f"Atlas: {atlas_type}")
    print(f"Total subjects: {num_subjects}")
    print(f"Timepoints: {num_timepoints}")
    print(f"TR: {TR} seconds")
    print(f"Model: {model_type}")
    print(f"Workers (CPUs): {num_workers}")
    print(f"Connectivity: {connectivity_dir}")
    print(f"Output: {output_dir}")
    print(f"Starting subject ID: {start_subject}")
    print("="*80)
    print()

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare arguments
    args_list = [
        (subject_id, atlas_type, connectivity_dir, num_timepoints, TR, model_type, output_dir)
        for subject_id in range(start_subject, start_subject + num_subjects)
    ]

    # Run simulations in parallel
    start_time = time.time()
    results = []

    print(f"Starting parallel simulation with {num_workers} workers...")
    print()

    with Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(simulate_single_subject, args_list), 1):
            results.append(result)

            if result['status'] == 'success':
                status_msg = f"✓ Subject {result['subject_id']:05d} completed in {result['elapsed_time']:.1f}s"
            else:
                status_msg = f"✗ Subject {result['subject_id']:05d} FAILED: {result.get('error', 'Unknown')}"

            # Print progress every 10 subjects
            if i % 10 == 0 or i == num_subjects:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (num_subjects - i) / rate if rate > 0 else 0

                print(f"Progress: {i}/{num_subjects} ({100*i/num_subjects:.1f}%) | "
                      f"Rate: {rate:.2f} subj/s | "
                      f"ETA: {eta/60:.1f} min | "
                      f"{status_msg}")

    total_time = time.time() - start_time

    # Summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    avg_time = np.mean([r['elapsed_time'] for r in results if r['status'] == 'success'])

    print()
    print("="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print(f"Atlas: {atlas_type}")
    print(f"Total subjects: {num_subjects}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Average time per subject: {avg_time:.1f} seconds")
    print(f"Total elapsed time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
    print(f"Effective rate: {num_subjects/total_time:.2f} subjects/second")
    print("="*80)

    # Save summary
    summary = {
        'atlas': atlas_type,
        'num_subjects': num_subjects,
        'num_timepoints': num_timepoints,
        'TR': TR,
        'model_type': model_type,
        'connectivity_dir': connectivity_dir,
        'num_workers': num_workers,
        'start_subject': start_subject,
        'successful': successful,
        'failed': failed,
        'total_time_seconds': total_time,
        'total_time_hours': total_time / 3600,
        'avg_time_per_subject': avg_time,
        'timestamp': str(datetime.now()),
        'results': results
    }

    summary_path = Path(output_dir) / 'simulation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run 10,000 subject simulations with real atlases (HarvardOxford or HCP-MMP1)'
    )
    parser.add_argument('--atlas', type=str, required=True,
                       choices=['harvard_oxford', 'hcpmmp1'],
                       help='Atlas to use: harvard_oxford (48 ROIs) or hcpmmp1 (360 ROIs)')
    parser.add_argument('--subjects', type=int, default=10000,
                       help='Number of subjects to simulate (default: 10000)')
    parser.add_argument('--timepoints', type=int, default=375,
                       help='Number of timepoints (default: 375)')
    parser.add_argument('--tr', type=float, default=0.8,
                       help='Repetition time in seconds (default: 0.8)')
    parser.add_argument('--model', type=str, default='wong_wang',
                       choices=['wong_wang', 'kuramoto', 'generic_2d_oscillator'],
                       help='Neural mass model type')
    parser.add_argument('--connectivity-dir', type=str, default=None,
                       help='Directory containing connectivity files (auto-detected if not specified)')
    parser.add_argument('--output', type=str, default='results/10k_subjects',
                       help='Base output directory')
    parser.add_argument('--workers', type=int, default=32,
                       help='Number of parallel workers (default: 32)')
    parser.add_argument('--start', type=int, default=0,
                       help='Starting subject ID for resuming (default: 0)')

    args = parser.parse_args()

    # Run simulation
    run_10k_subjects(
        atlas_type=args.atlas,
        num_subjects=args.subjects,
        num_timepoints=args.timepoints,
        TR=args.tr,
        model_type=args.model,
        connectivity_dir=args.connectivity_dir,
        output_dir=args.output,
        num_workers=args.workers,
        start_subject=args.start
    )


if __name__ == '__main__':
    main()
