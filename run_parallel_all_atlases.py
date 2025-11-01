"""
Run HarvardOxford and HCP-MMP1 simulations in parallel
Executes multiple atlas simulations concurrently using multiprocessing
"""

import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
import time
import sys

# Import atlas-specific simulation classes
from src.harvard_oxford_simulation import HarvardOxfordSimulation
from src.hcpmmp1_simulation import HCPMMP1Simulation
from src.simulation import BrainSimulation


def run_harvard_oxford_simulation(config):
    """
    Run HarvardOxford 48-ROI simulation

    Parameters:
    -----------
    config : dict
        Configuration with keys: num_timepoints, TR, model_type, output_path

    Returns:
    --------
    result : dict
        Simulation result with timing and status
    """
    try:
        start_time = time.time()

        print(f"\n[HarvardOxford 48 ROIs] Starting simulation...")

        # Create simulation
        sim = HarvardOxfordSimulation(
            num_timepoints=config['num_timepoints'],
            TR=config['TR'],
            model_type=config['model_type'],
            connectivity_dir=config.get('connectivity_dir', 'data/HarvardOxford')
        )

        # Run simulation
        sim.run_simulation()

        # Save results
        output_path = config.get('output_path', 'results/ho_sim_48nodes.npy')
        sim.save_results(output_path)

        elapsed_time = time.time() - start_time

        result = {
            'atlas': 'HarvardOxford',
            'num_nodes': 48,
            'status': 'success',
            'output_path': output_path,
            'elapsed_time_seconds': elapsed_time,
            'elapsed_time_minutes': elapsed_time / 60,
            'timestamp': str(datetime.now())
        }

        print(f"[HarvardOxford 48 ROIs] ✓ Completed in {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")

        return result

    except Exception as e:
        elapsed_time = time.time() - start_time

        result = {
            'atlas': 'HarvardOxford',
            'num_nodes': 48,
            'status': 'failed',
            'error': str(e),
            'elapsed_time_seconds': elapsed_time,
            'timestamp': str(datetime.now())
        }

        print(f"[HarvardOxford 48 ROIs] ✗ FAILED: {e}")

        return result


def run_hcpmmp1_simulation(config):
    """
    Run HCP-MMP1 360-ROI simulation

    Parameters:
    -----------
    config : dict
        Configuration with keys: num_timepoints, TR, model_type, output_path

    Returns:
    --------
    result : dict
        Simulation result with timing and status
    """
    try:
        start_time = time.time()

        print(f"\n[HCP-MMP1 360 ROIs] Starting simulation...")

        # Create simulation
        sim = HCPMMP1Simulation(
            num_timepoints=config['num_timepoints'],
            TR=config['TR'],
            model_type=config['model_type'],
            connectivity_dir=config.get('connectivity_dir', 'data/HCPMMP1')
        )

        # Run simulation
        sim.run_simulation()

        # Save results
        output_path = config.get('output_path', 'results/mmp_sim_360nodes.npy')
        sim.save_results(output_path)

        elapsed_time = time.time() - start_time

        result = {
            'atlas': 'HCP-MMP1',
            'num_nodes': 360,
            'status': 'success',
            'output_path': output_path,
            'elapsed_time_seconds': elapsed_time,
            'elapsed_time_minutes': elapsed_time / 60,
            'timestamp': str(datetime.now())
        }

        print(f"[HCP-MMP1 360 ROIs] ✓ Completed in {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")

        return result

    except Exception as e:
        elapsed_time = time.time() - start_time

        result = {
            'atlas': 'HCP-MMP1',
            'num_nodes': 360,
            'status': 'failed',
            'error': str(e),
            'elapsed_time_seconds': elapsed_time,
            'timestamp': str(datetime.now())
        }

        print(f"[HCP-MMP1 360 ROIs] ✗ FAILED: {e}")

        return result


def run_desikan_killiany_simulation(config):
    """
    Run Desikan-Killiany 68-ROI simulation

    Parameters:
    -----------
    config : dict
        Configuration with keys: num_timepoints, TR, model_type, output_path

    Returns:
    --------
    result : dict
        Simulation result with timing and status
    """
    try:
        start_time = time.time()

        print(f"\n[Desikan-Killiany 68 ROIs] Starting simulation...")

        # Create simulation
        sim = BrainSimulation(
            num_nodes=68,
            num_timepoints=config['num_timepoints'],
            TR=config['TR'],
            model_type=config['model_type']
        )

        # Run simulation
        sim.run_simulation()

        # Save results
        output_path = config.get('output_path', 'results/dk_sim_68nodes.npy')
        sim.save_results(output_path)

        elapsed_time = time.time() - start_time

        result = {
            'atlas': 'Desikan-Killiany',
            'num_nodes': 68,
            'status': 'success',
            'output_path': output_path,
            'elapsed_time_seconds': elapsed_time,
            'elapsed_time_minutes': elapsed_time / 60,
            'timestamp': str(datetime.now())
        }

        print(f"[Desikan-Killiany 68 ROIs] ✓ Completed in {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")

        return result

    except Exception as e:
        elapsed_time = time.time() - start_time

        result = {
            'atlas': 'Desikan-Killiany',
            'num_nodes': 68,
            'status': 'failed',
            'error': str(e),
            'elapsed_time_seconds': elapsed_time,
            'timestamp': str(datetime.now())
        }

        print(f"[Desikan-Killiany 68 ROIs] ✗ FAILED: {e}")

        return result


def dispatch_simulation(task):
    """
    Dispatcher function for multiprocessing

    Parameters:
    -----------
    task : tuple
        (atlas_type, config) where atlas_type is 'harvard_oxford', 'hcpmmp1', or 'desikan_killiany'

    Returns:
    --------
    result : dict
        Simulation result
    """
    atlas_type, config = task

    if atlas_type == 'harvard_oxford':
        return run_harvard_oxford_simulation(config)
    elif atlas_type == 'hcpmmp1':
        return run_hcpmmp1_simulation(config)
    elif atlas_type == 'desikan_killiany':
        return run_desikan_killiany_simulation(config)
    else:
        raise ValueError(f"Unknown atlas type: {atlas_type}")


def run_all_atlases_parallel(
    atlases=['harvard_oxford', 'desikan_killiany', 'hcpmmp1'],
    num_timepoints=375,
    TR=0.8,
    model_type='wong_wang',
    output_dir='results',
    num_workers=None
):
    """
    Run all atlas simulations in parallel

    Parameters:
    -----------
    atlases : list of str
        List of atlas names to run ('harvard_oxford', 'desikan_killiany', 'hcpmmp1')
    num_timepoints : int
        Number of timepoints (default: 375)
    TR : float
        Repetition time in seconds (default: 0.8)
    model_type : str
        Neural mass model type (default: 'wong_wang')
    output_dir : str
        Output directory for results
    num_workers : int, optional
        Number of parallel workers (default: number of atlases)

    Returns:
    --------
    results : list of dict
        Results for each atlas
    """

    # Determine number of workers
    if num_workers is None:
        num_workers = min(len(atlases), cpu_count())
    else:
        num_workers = min(num_workers, cpu_count())

    print("="*80)
    print("PARALLEL MULTI-ATLAS SIMULATION")
    print("="*80)
    print(f"Atlases to run: {', '.join(atlases)}")
    print(f"  - HarvardOxford: 48 cortical ROIs" if 'harvard_oxford' in atlases else "")
    print(f"  - Desikan-Killiany: 68 cortical ROIs" if 'desikan_killiany' in atlases else "")
    print(f"  - HCP-MMP1: 360 cortical ROIs" if 'hcpmmp1' in atlases else "")
    print(f"Timepoints: {num_timepoints}")
    print(f"TR: {TR} seconds")
    print(f"Model: {model_type}")
    print(f"Parallel workers: {num_workers}")
    print(f"Total CPU cores: {cpu_count()}")
    print(f"Output directory: {output_dir}/")
    print("="*80)
    print()

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare tasks
    tasks = []

    for atlas in atlases:
        config = {
            'num_timepoints': num_timepoints,
            'TR': TR,
            'model_type': model_type
        }

        if atlas == 'harvard_oxford':
            config['output_path'] = f"{output_dir}/ho_sim_48nodes_{num_timepoints}tp_TR{TR}.npy"
            config['connectivity_dir'] = 'data/HarvardOxford'
        elif atlas == 'desikan_killiany':
            config['output_path'] = f"{output_dir}/dk_sim_68nodes_{num_timepoints}tp_TR{TR}.npy"
        elif atlas == 'hcpmmp1':
            config['output_path'] = f"{output_dir}/mmp_sim_360nodes_{num_timepoints}tp_TR{TR}.npy"
            config['connectivity_dir'] = 'data/HCPMMP1'

        tasks.append((atlas, config))

    # Run simulations in parallel
    batch_start = time.time()

    print(f"Starting parallel execution with {num_workers} workers...")
    print(f"Running {len(tasks)} atlas simulations simultaneously!\n")

    with Pool(processes=num_workers) as pool:
        results = pool.map(dispatch_simulation, tasks)

    total_elapsed = time.time() - batch_start

    # Print summary
    print("\n" + "="*80)
    print("PARALLEL EXECUTION SUMMARY")
    print("="*80)

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    print(f"\nCompletion Status:")
    print(f"  Total simulations: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    print(f"\nTiming:")
    print(f"  Total parallel execution time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    if successful:
        sequential_time = sum(r['elapsed_time_seconds'] for r in successful)
        speedup = sequential_time / total_elapsed
        print(f"  Estimated sequential time: {sequential_time:.1f}s ({sequential_time/60:.1f} min)")
        print(f"  Speedup factor: {speedup:.2f}x")

    print(f"\nResults by Atlas:")
    for r in sorted(results, key=lambda x: x['num_nodes']):
        if r['status'] == 'success':
            print(f"  ✓ {r['atlas']:20s} ({r['num_nodes']:3d} ROIs): {r['elapsed_time_seconds']:6.1f}s ({r['elapsed_time_minutes']:.1f} min)")
        else:
            print(f"  ✗ {r['atlas']:20s} ({r['num_nodes']:3d} ROIs): FAILED - {r.get('error', 'Unknown error')}")

    if failed:
        print(f"\n⚠ Warning: {len(failed)} simulation(s) failed")

    print("="*80)

    # Save summary
    summary = {
        'timestamp': str(datetime.now()),
        'atlases': atlases,
        'num_workers': num_workers,
        'num_timepoints': num_timepoints,
        'TR': TR,
        'model_type': model_type,
        'total_simulations': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'total_time_seconds': total_elapsed,
        'total_time_minutes': total_elapsed / 60,
        'speedup_factor': sequential_time / total_elapsed if successful else 0,
        'results': results
    }

    summary_path = Path(output_dir) / 'parallel_multi_atlas_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run HarvardOxford, Desikan-Killiany, and HCP-MMP1 simulations in parallel'
    )
    parser.add_argument('--atlases', type=str, nargs='+',
                       default=['harvard_oxford', 'desikan_killiany', 'hcpmmp1'],
                       choices=['harvard_oxford', 'desikan_killiany', 'hcpmmp1'],
                       help='Atlases to run (default: all)')
    parser.add_argument('--timepoints', type=int, default=375,
                       help='Number of timepoints (default: 375)')
    parser.add_argument('--tr', type=float, default=0.8,
                       help='Repetition time in seconds (default: 0.8)')
    parser.add_argument('--model', type=str, default='wong_wang',
                       choices=['wong_wang', 'kuramoto', 'generic_2d_oscillator'],
                       help='Neural mass model type (default: wong_wang)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto)')

    args = parser.parse_args()

    # Run parallel simulations
    results = run_all_atlases_parallel(
        atlases=args.atlases,
        num_timepoints=args.timepoints,
        TR=args.tr,
        model_type=args.model,
        output_dir=args.output,
        num_workers=args.workers
    )

    # Print final summary
    successful = len([r for r in results if r['status'] == 'success'])
    total = len(results)

    if successful == total:
        print(f"🎉 All {total} atlas simulations completed successfully!")
    elif successful > 0:
        print(f"⚠  {successful}/{total} atlas simulations completed successfully")
    else:
        print(f"❌ All simulations failed")

    return results


if __name__ == '__main__':
    main()
