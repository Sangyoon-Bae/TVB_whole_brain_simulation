"""
Parallel Brain Simulation for Multiple Subjects
Simulates 10,000 different brains using multiprocessing with 32 CPUs
Each subject has unique connectivity and noise patterns
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
from simulation import BrainSimulation


def create_subject_connectivity(num_nodes, subject_id, seed):
    """
    Create unique connectivity for each subject

    Parameters:
    -----------
    num_nodes : int
        Number of brain regions
    subject_id : int
        Subject identifier
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    conn : Connectivity
        Subject-specific connectivity
    """
    np.random.seed(seed)

    # Load default connectivity
    conn = connectivity.Connectivity.from_file()

    if num_nodes == 48:
        # Use reduced connectivity
        indices = np.linspace(0, len(conn.weights) - 1, 48, dtype=int)
        conn.weights = conn.weights[indices][:, indices]
        conn.tract_lengths = conn.tract_lengths[indices][:, indices]
        conn.region_labels = conn.region_labels[indices]
        conn.centres = conn.centres[indices]

    # Add subject-specific variability to connection weights
    # Each subject has unique connectivity strength variations (±20%)
    variability = 0.2
    subject_variation = 1.0 + np.random.uniform(-variability, variability, conn.weights.shape)
    conn.weights = conn.weights * subject_variation

    # Ensure weights are non-negative and symmetric
    conn.weights = np.maximum(conn.weights, 0)
    conn.weights = (conn.weights + conn.weights.T) / 2

    conn.configure()
    return conn


class SubjectBrainSimulation(BrainSimulation):
    """Extended BrainSimulation with subject-specific variations"""

    def __init__(self, subject_id, num_nodes=48, num_timepoints=375, TR=0.8, model_type='wong_wang'):
        super().__init__(num_nodes, num_timepoints, TR, model_type)
        self.subject_id = subject_id
        # Use subject_id as seed for reproducibility
        self.seed = subject_id

    def setup_connectivity(self):
        """Setup subject-specific connectivity"""
        return create_subject_connectivity(self.num_nodes, self.subject_id, self.seed)

    def setup_integrator(self):
        """Setup integration scheme with subject-specific noise"""
        # Each subject has unique noise pattern
        np.random.seed(self.seed)
        heunint = integrators.HeunStochastic(
            dt=2**-4,
            noise=noise.Additive(nsig=np.array([0.001]))
        )
        return heunint


def simulate_single_subject(args):
    """
    Simulate a single subject's brain activity

    Parameters:
    -----------
    args : tuple
        (subject_id, num_nodes, num_timepoints, TR, model_type, base_output_dir)

    Returns:
    --------
    result : dict
        Simulation result summary
    """
    subject_id, num_nodes, num_timepoints, TR, model_type, base_output_dir = args

    try:
        start_time = time.time()

        # Create subject-specific output directory
        subject_dir = Path(base_output_dir) / f"subject_{subject_id:05d}"
        subject_dir.mkdir(parents=True, exist_ok=True)

        # Initialize simulation
        sim = SubjectBrainSimulation(
            subject_id=subject_id,
            num_nodes=num_nodes,
            num_timepoints=num_timepoints,
            TR=TR,
            model_type=model_type
        )

        # Run simulation (suppress progress output for parallel jobs)
        import sys
        import io

        # Redirect stdout to suppress verbose output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        sim.run_simulation()

        # Restore stdout
        sys.stdout = old_stdout

        # Save results
        output_path = subject_dir / f"sim_{num_nodes}nodes_{num_timepoints}tp_TR{TR}.npy"
        sim.save_results(output_path)

        elapsed_time = time.time() - start_time

        return {
            'subject_id': subject_id,
            'status': 'success',
            'output_dir': str(subject_dir),
            'elapsed_time': elapsed_time,
            'seed': subject_id
        }

    except Exception as e:
        return {
            'subject_id': subject_id,
            'status': 'failed',
            'error': str(e),
            'elapsed_time': 0
        }


def run_parallel_simulations(
    num_subjects=10000,
    num_nodes=48,
    num_timepoints=375,
    TR=0.8,
    model_type='wong_wang',
    output_dir='results/parallel_sims',
    num_workers=32,
    start_subject=0
):
    """
    Run parallel brain simulations for multiple subjects

    Parameters:
    -----------
    num_subjects : int
        Total number of subjects to simulate (default: 10000)
    num_nodes : int
        Number of brain regions per subject
    num_timepoints : int
        Number of timepoints per simulation
    TR : float
        Repetition time in seconds
    model_type : str
        Neural mass model type
    output_dir : str
        Base output directory
    num_workers : int
        Number of parallel workers (default: 32)
    start_subject : int
        Starting subject ID (for resuming)
    """

    print("="*80)
    print("PARALLEL BRAIN SIMULATION")
    print("="*80)
    print(f"Total subjects: {num_subjects}")
    print(f"Nodes per subject: {num_nodes}")
    print(f"Timepoints: {num_timepoints}")
    print(f"TR: {TR} seconds")
    print(f"Model: {model_type}")
    print(f"Workers (CPUs): {num_workers}")
    print(f"Output directory: {output_dir}")
    print(f"Starting subject ID: {start_subject}")
    print("="*80)
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for all subjects
    subject_args = [
        (subject_id, num_nodes, num_timepoints, TR, model_type, output_dir)
        for subject_id in range(start_subject, start_subject + num_subjects)
    ]

    # Run parallel simulations
    start_time = time.time()
    results = []

    print(f"Starting parallel simulation with {num_workers} workers...")
    print()

    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for progress tracking
        for i, result in enumerate(pool.imap_unordered(simulate_single_subject, subject_args), 1):
            results.append(result)

            if result['status'] == 'success':
                status_msg = f"✓ Subject {result['subject_id']:05d} completed in {result['elapsed_time']:.1f}s"
            else:
                status_msg = f"✗ Subject {result['subject_id']:05d} FAILED: {result['error']}"

            # Print progress every 10 subjects
            if i % 10 == 0 or i == num_subjects:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (num_subjects - i) / rate if rate > 0 else 0

                print(f"Progress: {i}/{num_subjects} ({100*i/num_subjects:.1f}%) | "
                      f"Rate: {rate:.2f} subjects/s | "
                      f"ETA: {eta/60:.1f} min | "
                      f"{status_msg}")

    total_time = time.time() - start_time

    # Summary statistics
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    avg_time = np.mean([r['elapsed_time'] for r in results if r['status'] == 'success'])

    print()
    print("="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print(f"Total subjects: {num_subjects}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Average time per subject: {avg_time:.1f} seconds")
    print(f"Total elapsed time: {total_time/60:.1f} minutes")
    print(f"Effective rate: {num_subjects/total_time:.2f} subjects/second")
    print("="*80)

    # Save summary
    summary = {
        'num_subjects': num_subjects,
        'num_nodes': num_nodes,
        'num_timepoints': num_timepoints,
        'TR': TR,
        'model_type': model_type,
        'num_workers': num_workers,
        'start_subject': start_subject,
        'successful': successful,
        'failed': failed,
        'total_time_seconds': total_time,
        'avg_time_per_subject': avg_time,
        'timestamp': str(datetime.now()),
        'results': results
    }

    summary_path = output_path / 'simulation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run parallel whole-brain simulations for multiple subjects'
    )
    parser.add_argument('--subjects', type=int, default=10000,
                       help='Number of subjects to simulate (default: 10000)')
    parser.add_argument('--nodes', type=int, default=48, choices=[48, 360],
                       help='Number of brain regions (default: 48)')
    parser.add_argument('--timepoints', type=int, default=375,
                       help='Number of timepoints (default: 375)')
    parser.add_argument('--tr', type=float, default=0.8,
                       help='Repetition time (TR) in seconds (default: 0.8)')
    parser.add_argument('--model', type=str, default='wong_wang',
                       choices=['wong_wang', 'kuramoto', 'generic_2d_oscillator'],
                       help='Neural mass model type')
    parser.add_argument('--output', type=str, default='results/parallel_sims',
                       help='Base output directory')
    parser.add_argument('--workers', type=int, default=32,
                       help='Number of parallel workers (default: 32, max: 32)')
    parser.add_argument('--start', type=int, default=0,
                       help='Starting subject ID (for resuming, default: 0)')

    args = parser.parse_args()

    # Validate workers
    max_workers = min(args.workers, 32)
    available_cpus = cpu_count()

    if max_workers > available_cpus:
        print(f"Warning: Requested {max_workers} workers but only {available_cpus} CPUs available")
        print(f"Using {available_cpus} workers instead")
        max_workers = available_cpus

    # Run parallel simulations
    run_parallel_simulations(
        num_subjects=args.subjects,
        num_nodes=args.nodes,
        num_timepoints=args.timepoints,
        TR=args.tr,
        model_type=args.model,
        output_dir=args.output,
        num_workers=max_workers,
        start_subject=args.start
    )


if __name__ == '__main__':
    main()
