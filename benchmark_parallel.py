"""
Benchmark Sequential vs Parallel Execution
Compares performance of sequential vs parallel atlas simulations
"""

import time
import argparse
from pathlib import Path
import json
from datetime import datetime

from src.simulation import BrainSimulation
from run_parallel_atlases import run_all_atlases_parallel


def run_sequential_benchmark(timepoints=375, TR=0.8, model_type='wong_wang', output_dir='results/benchmark_seq'):
    """
    Run simulations sequentially (one after another)

    Returns:
    --------
    results : dict
        Benchmark results with timing information
    """
    print("\n" + "="*80)
    print("SEQUENTIAL EXECUTION BENCHMARK")
    print("="*80)
    print(f"Running 3 atlas simulations sequentially (48, 68, 360 nodes)")
    print(f"Timepoints: {timepoints}, TR: {TR}, Model: {model_type}")
    print("="*80)
    print()

    atlas_configs = [
        {'num_nodes': 48, 'name': 'Downsampled'},
        {'num_nodes': 68, 'name': 'Desikan-Killiany'},
        {'num_nodes': 360, 'name': 'HCP-MMP1'}
    ]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []
    total_start = time.time()

    for config in atlas_configs:
        num_nodes = config['num_nodes']
        atlas_name = config['name']

        print(f"\n[Atlas {len(results)+1}/3: {num_nodes} nodes - {atlas_name}]")
        print("-" * 60)

        try:
            start_time = time.time()

            # Create and run simulation
            sim = BrainSimulation(
                num_nodes=num_nodes,
                num_timepoints=timepoints,
                TR=TR,
                model_type=model_type
            )

            sim.run_simulation()

            # Save results
            output_path = f"{output_dir}/sim_{num_nodes}nodes_{timepoints}tp_TR{TR}.npy"
            sim.save_results(output_path)

            elapsed_time = time.time() - start_time

            result = {
                'num_nodes': num_nodes,
                'atlas_name': atlas_name,
                'status': 'success',
                'elapsed_time_seconds': elapsed_time,
                'elapsed_time_minutes': elapsed_time / 60
            }

            print(f"✓ Completed in {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")

        except Exception as e:
            elapsed_time = time.time() - start_time
            result = {
                'num_nodes': num_nodes,
                'atlas_name': atlas_name,
                'status': 'failed',
                'error': str(e),
                'elapsed_time_seconds': elapsed_time
            }
            print(f"✗ FAILED: {e}")

        results.append(result)

    total_elapsed = time.time() - total_start

    print("\n" + "="*80)
    print("SEQUENTIAL EXECUTION COMPLETE")
    print("="*80)
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print()
    for r in results:
        if r['status'] == 'success':
            print(f"  {r['num_nodes']:3d} nodes: {r['elapsed_time_seconds']:6.1f}s ({r['elapsed_time_minutes']:.1f} min)")
        else:
            print(f"  {r['num_nodes']:3d} nodes: FAILED")
    print("="*80)

    return {
        'method': 'sequential',
        'total_time_seconds': total_elapsed,
        'total_time_minutes': total_elapsed / 60,
        'results': results,
        'timestamp': str(datetime.now())
    }


def run_parallel_benchmark(timepoints=375, TR=0.8, model_type='wong_wang',
                          output_dir='results/benchmark_parallel', num_workers=3):
    """
    Run simulations in parallel (all at once)

    Returns:
    --------
    results : dict
        Benchmark results with timing information
    """
    print("\n" + "="*80)
    print("PARALLEL EXECUTION BENCHMARK")
    print("="*80)
    print(f"Running 3 atlas simulations in parallel (48, 68, 360 nodes)")
    print(f"Timepoints: {timepoints}, TR: {TR}, Model: {model_type}")
    print(f"Workers: {num_workers}")
    print("="*80)
    print()

    total_start = time.time()

    results = run_all_atlases_parallel(
        num_timepoints=timepoints,
        TR=TR,
        model_type=model_type,
        output_prefix=output_dir,
        num_workers=num_workers
    )

    total_elapsed = time.time() - total_start

    return {
        'method': 'parallel',
        'total_time_seconds': total_elapsed,
        'total_time_minutes': total_elapsed / 60,
        'num_workers': num_workers,
        'results': results,
        'timestamp': str(datetime.now())
    }


def compare_benchmarks(seq_results, par_results):
    """
    Compare sequential vs parallel performance

    Parameters:
    -----------
    seq_results : dict
        Sequential benchmark results
    par_results : dict
        Parallel benchmark results
    """
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON: SEQUENTIAL vs PARALLEL")
    print("="*80)

    seq_time = seq_results['total_time_seconds']
    par_time = par_results['total_time_seconds']
    speedup = seq_time / par_time
    time_saved = seq_time - par_time

    print(f"\nTotal Execution Time:")
    print(f"  Sequential: {seq_time:.1f}s ({seq_time/60:.1f} min)")
    print(f"  Parallel:   {par_time:.1f}s ({par_time/60:.1f} min)")
    print()
    print(f"Performance Gain:")
    print(f"  Speedup: {speedup:.2f}x faster")
    print(f"  Time saved: {time_saved:.1f}s ({time_saved/60:.1f} min)")
    print(f"  Efficiency: {(speedup/par_results['num_workers'])*100:.1f}%")

    # Per-atlas breakdown
    print(f"\nPer-Atlas Timing Breakdown:")
    print(f"{'Atlas':<20} {'Sequential':>12} {'Parallel':>12} {'Speedup':>10}")
    print("-" * 60)

    for seq_r, par_r in zip(seq_results['results'], sorted(par_results['results'], key=lambda x: x['num_nodes'])):
        if seq_r['status'] == 'success' and par_r['status'] == 'success':
            atlas_name = f"{seq_r['num_nodes']} nodes"
            seq_t = seq_r['elapsed_time_seconds']
            par_t = par_r['elapsed_time_seconds']
            atlas_speedup = seq_t / par_t if par_t > 0 else 0

            print(f"{atlas_name:<20} {seq_t:>10.1f}s {par_t:>10.1f}s {atlas_speedup:>9.2f}x")

    print("="*80)

    # Recommendations
    print("\nRecommendations:")
    if speedup > 1.5:
        print(f"  ✓ Parallel execution is {speedup:.2f}x faster - RECOMMENDED for batch jobs!")
    elif speedup > 1.2:
        print(f"  ✓ Parallel execution is {speedup:.2f}x faster - beneficial for multiple runs")
    else:
        print(f"  ⚠ Parallel speedup is only {speedup:.2f}x - overhead may be significant")

    if par_results['num_workers'] == 3:
        print(f"  ✓ Using optimal worker count (3 workers for 3 atlases)")
    else:
        print(f"  ℹ Consider using 3 workers for 3 atlas simulations")

    print()

    return {
        'speedup': speedup,
        'time_saved_seconds': time_saved,
        'efficiency_percent': (speedup / par_results['num_workers']) * 100
    }


def save_benchmark_results(seq_results, par_results, comparison, output_file='results/benchmark_results.json'):
    """Save benchmark results to JSON file"""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    results = {
        'timestamp': str(datetime.now()),
        'sequential': seq_results,
        'parallel': par_results,
        'comparison': comparison
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Benchmark results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark sequential vs parallel atlas simulations'
    )
    parser.add_argument('--mode', type=str, choices=['sequential', 'parallel', 'both'], default='both',
                       help='Benchmark mode (default: both)')
    parser.add_argument('--timepoints', type=int, default=375,
                       help='Number of timepoints (default: 375)')
    parser.add_argument('--tr', type=float, default=0.8,
                       help='Repetition time in seconds (default: 0.8)')
    parser.add_argument('--model', type=str, default='wong_wang',
                       choices=['wong_wang', 'kuramoto', 'generic_2d_oscillator'],
                       help='Neural mass model type (default: wong_wang)')
    parser.add_argument('--workers', type=int, default=3,
                       help='Number of parallel workers (default: 3)')
    parser.add_argument('--output', type=str, default='results/benchmark_results.json',
                       help='Output file for benchmark results')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("ATLAS SIMULATION BENCHMARK")
    print("="*80)
    print(f"Configuration:")
    print(f"  Timepoints: {args.timepoints}")
    print(f"  TR: {args.tr} seconds")
    print(f"  Model: {args.model}")
    print(f"  Mode: {args.mode}")
    if args.mode in ['parallel', 'both']:
        print(f"  Parallel workers: {args.workers}")
    print("="*80)

    seq_results = None
    par_results = None

    # Run sequential benchmark
    if args.mode in ['sequential', 'both']:
        seq_results = run_sequential_benchmark(
            timepoints=args.timepoints,
            TR=args.tr,
            model_type=args.model,
            output_dir='results/benchmark_seq'
        )

    # Run parallel benchmark
    if args.mode in ['parallel', 'both']:
        par_results = run_parallel_benchmark(
            timepoints=args.timepoints,
            TR=args.tr,
            model_type=args.model,
            output_dir='results/benchmark_parallel',
            num_workers=args.workers
        )

    # Compare results
    if args.mode == 'both':
        comparison = compare_benchmarks(seq_results, par_results)
        save_benchmark_results(seq_results, par_results, comparison, args.output)

    print("\n✓ Benchmark complete!")


if __name__ == '__main__':
    main()
