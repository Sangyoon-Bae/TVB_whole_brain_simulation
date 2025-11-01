"""
Run multiple atlas simulations in parallel
Executes 48, 68, and 360 node simulations concurrently using multiprocessing
"""

import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
import time

from src.simulation import BrainSimulation


def run_single_atlas_simulation(config):
    """
    Run simulation for a single atlas configuration

    Parameters:
    -----------
    config : dict
        Configuration with keys: num_nodes, num_timepoints, TR, model_type, output_prefix

    Returns:
    --------
    result : dict
        Simulation result with timing and status
    """
    num_nodes = config['num_nodes']
    num_timepoints = config['num_timepoints']
    TR = config['TR']
    model_type = config['model_type']
    output_prefix = config.get('output_prefix', 'results')

    atlas_names = {
        48: 'Downsampled',
        68: 'Desikan-Killiany',
        360: 'HCP-MMP1'
    }

    try:
        start_time = time.time()

        print(f"\n[Atlas: {num_nodes} nodes - {atlas_names.get(num_nodes, 'Unknown')}] Starting simulation...")

        # Create simulation
        sim = BrainSimulation(
            num_nodes=num_nodes,
            num_timepoints=num_timepoints,
            TR=TR,
            model_type=model_type
        )

        # Run simulation
        sim.run_simulation()

        # Save results
        output_path = f"{output_prefix}/sim_{num_nodes}nodes_{num_timepoints}tp_TR{TR}.npy"
        sim.save_results(output_path)

        elapsed_time = time.time() - start_time

        result = {
            'num_nodes': num_nodes,
            'atlas_name': atlas_names.get(num_nodes, 'Unknown'),
            'status': 'success',
            'output_path': output_path,
            'elapsed_time_seconds': elapsed_time,
            'elapsed_time_minutes': elapsed_time / 60,
            'timestamp': str(datetime.now())
        }

        print(f"[Atlas: {num_nodes} nodes] ✓ Completed in {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")

        return result

    except Exception as e:
        elapsed_time = time.time() - start_time

        result = {
            'num_nodes': num_nodes,
            'atlas_name': atlas_names.get(num_nodes, 'Unknown'),
            'status': 'failed',
            'error': str(e),
            'elapsed_time_seconds': elapsed_time,
            'timestamp': str(datetime.now())
        }

        print(f"[Atlas: {num_nodes} nodes] ✗ FAILED: {e}")

        return result


def run_all_atlases_parallel(num_timepoints=375, TR=0.8, model_type='wong_wang',
                             output_prefix='results', num_workers=None):
    """
    Run simulations for all atlas configurations in parallel

    Parameters:
    -----------
    num_timepoints : int
        Number of timepoints (default: 375)
    TR : float
        Repetition time in seconds (default: 0.8)
    model_type : str
        Neural mass model type (default: 'wong_wang')
    output_prefix : str
        Output directory prefix (default: 'results')
    num_workers : int, optional
        Number of parallel workers (default: 3 for 3 atlases)

    Returns:
    --------
    results : list of dict
        Results for each atlas configuration
    """

    # Determine number of workers
    if num_workers is None:
        num_workers = min(3, cpu_count())  # 3 atlases, so max 3 workers
    else:
        num_workers = min(num_workers, cpu_count())

    print("="*80)
    print("PARALLEL ATLAS SIMULATION")
    print("="*80)
    print(f"Atlas configurations: 48, 68, 360 nodes")
    print(f"Timepoints: {num_timepoints}")
    print(f"TR: {TR} seconds")
    print(f"Model: {model_type}")
    print(f"Parallel workers: {num_workers}")
    print(f"Total CPU cores available: {cpu_count()}")
    print(f"Output directory: {output_prefix}/")
    print("="*80)
    print()

    # Create configurations for all atlases
    configs = [
        {
            'num_nodes': 48,
            'num_timepoints': num_timepoints,
            'TR': TR,
            'model_type': model_type,
            'output_prefix': output_prefix
        },
        {
            'num_nodes': 68,
            'num_timepoints': num_timepoints,
            'TR': TR,
            'model_type': model_type,
            'output_prefix': output_prefix
        },
        {
            'num_nodes': 360,
            'num_timepoints': num_timepoints,
            'TR': TR,
            'model_type': model_type,
            'output_prefix': output_prefix
        }
    ]

    # Run simulations in parallel
    batch_start = time.time()

    print(f"Starting parallel execution with {num_workers} workers...")
    print("This will run all 3 atlas simulations simultaneously!\n")

    with Pool(processes=num_workers) as pool:
        results = pool.map(run_single_atlas_simulation, configs)

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
        print(f"  Sequential time would be: {sequential_time:.1f}s ({sequential_time/60:.1f} min)")
        print(f"  Speedup factor: {speedup:.2f}x")

    print(f"\nResults by Atlas:")
    for r in sorted(results, key=lambda x: x['num_nodes']):
        if r['status'] == 'success':
            print(f"  ✓ {r['num_nodes']:3d} nodes ({r['atlas_name']:20s}): {r['elapsed_time_seconds']:6.1f}s ({r['elapsed_time_minutes']:.1f} min)")
        else:
            print(f"  ✗ {r['num_nodes']:3d} nodes ({r['atlas_name']:20s}): FAILED - {r['error']}")

    if failed:
        print(f"\n⚠ Warning: {len(failed)} simulation(s) failed")

    print("="*80)

    # Save summary
    output_path = Path(output_prefix)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = {
        'timestamp': str(datetime.now()),
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

    summary_path = output_path / 'parallel_atlas_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}\n")

    return results


def run_parameter_sweep(node_configs, timepoints_list, TR_list, model_types,
                        output_prefix='results/parameter_sweep', num_workers=None):
    """
    Run parameter sweep across multiple configurations in parallel

    Parameters:
    -----------
    node_configs : list of int
        List of node counts to test (e.g., [48, 68, 360])
    timepoints_list : list of int
        List of timepoint values to test
    TR_list : list of float
        List of TR values to test
    model_types : list of str
        List of model types to test
    output_prefix : str
        Output directory prefix
    num_workers : int, optional
        Number of parallel workers (default: CPU count - 1)

    Returns:
    --------
    results : list of dict
        Results for all parameter combinations
    """

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    # Generate all combinations
    configs = []
    for nodes in node_configs:
        for tp in timepoints_list:
            for tr in TR_list:
                for model in model_types:
                    config = {
                        'num_nodes': nodes,
                        'num_timepoints': tp,
                        'TR': tr,
                        'model_type': model,
                        'output_prefix': output_prefix
                    }
                    configs.append(config)

    print("="*80)
    print("PARALLEL PARAMETER SWEEP")
    print("="*80)
    print(f"Node configurations: {node_configs}")
    print(f"Timepoints: {timepoints_list}")
    print(f"TRs: {TR_list}")
    print(f"Models: {model_types}")
    print(f"Total combinations: {len(configs)}")
    print(f"Parallel workers: {num_workers}")
    print("="*80)
    print()

    # Run in parallel
    batch_start = time.time()

    with Pool(processes=num_workers) as pool:
        results = pool.map(run_single_atlas_simulation, configs)

    total_elapsed = time.time() - batch_start

    # Summary
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    print("\n" + "="*80)
    print("PARAMETER SWEEP SUMMARY")
    print("="*80)
    print(f"Total combinations: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print("="*80)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run brain simulations for multiple atlases in parallel'
    )
    parser.add_argument('--timepoints', type=int, default=375,
                       help='Number of timepoints (default: 375)')
    parser.add_argument('--tr', type=float, default=0.8,
                       help='Repetition time in seconds (default: 0.8)')
    parser.add_argument('--model', type=str, default='wong_wang',
                       choices=['wong_wang', 'kuramoto', 'generic_2d_oscillator'],
                       help='Neural mass model type (default: wong_wang)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory prefix (default: results)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: 3)')
    parser.add_argument('--atlases', type=int, nargs='+', default=[48, 68, 360],
                       choices=[48, 68, 360],
                       help='Atlas configurations to run (default: 48 68 360)')

    args = parser.parse_args()

    # Run parallel simulations
    results = run_all_atlases_parallel(
        num_timepoints=args.timepoints,
        TR=args.tr,
        model_type=args.model,
        output_prefix=args.output,
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
