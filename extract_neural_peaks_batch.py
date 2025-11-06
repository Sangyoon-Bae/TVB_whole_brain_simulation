"""
Batch extraction of neural activity peak metadata from existing simulation results
Processes all subjects in parallel to extract peak width, timing, and amplitude

This script scans existing simulation output directories and extracts neural peak features
without re-running the simulations.
"""

import numpy as np
import argparse
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time
from datetime import datetime

from src.neural_peak_analysis import find_neural_peaks, save_peak_analysis


def process_single_subject(args):
    """
    Process neural peak extraction for a single subject

    Parameters:
    -----------
    args : tuple
        (subject_dir, tr, num_peaks)

    Returns:
    --------
    result : dict
        Processing result summary
    """
    subject_dir, tr, num_peaks = args
    subject_dir = Path(subject_dir)

    try:
        start_time = time.time()

        # Find neural activity file
        neural_files = list(subject_dir.glob("*_neural.npy"))

        if not neural_files:
            return {
                'subject_dir': str(subject_dir),
                'status': 'skipped',
                'reason': 'No neural activity file found'
            }

        neural_file = neural_files[0]

        # Check if peak analysis already exists
        peak_file = subject_dir / f"{neural_file.stem}_neural_peaks.json"
        if peak_file.exists():
            return {
                'subject_dir': str(subject_dir),
                'status': 'skipped',
                'reason': 'Peak analysis already exists',
                'elapsed_time': 0
            }

        # Load neural data
        neural_data = np.load(neural_file)

        # Calculate sampling rate from TR
        sampling_rate = 1.0 / tr

        # Extract peaks
        results = find_neural_peaks(
            neural_data,
            sampling_rate=sampling_rate,
            num_peaks=num_peaks
        )

        # Save results
        json_path = save_peak_analysis(results, neural_file)

        elapsed_time = time.time() - start_time

        return {
            'subject_dir': str(subject_dir),
            'status': 'success',
            'num_rois': neural_data.shape[0],
            'num_timepoints': neural_data.shape[1],
            'total_peaks': results['global_statistics']['total_peaks_found'],
            'output_file': str(json_path),
            'elapsed_time': elapsed_time
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            'subject_dir': str(subject_dir),
            'status': 'failed',
            'error': str(e),
            'elapsed_time': elapsed_time
        }


def extract_peaks_batch(
    input_dir='results/10k_subjects',
    tr=0.8,
    num_peaks=12,
    num_workers=None,
    start_subject=None,
    end_subject=None
):
    """
    Batch process neural peak extraction for all subjects

    Parameters:
    -----------
    input_dir : str
        Directory containing subject folders
    tr : float
        Repetition time in seconds (default: 0.8)
    num_peaks : int
        Number of peaks to extract per ROI (default: 12)
    num_workers : int
        Number of parallel workers (default: CPU count)
    start_subject : int or None
        Starting subject ID for processing subset
    end_subject : int or None
        Ending subject ID for processing subset (exclusive)

    Returns:
    --------
    results : list
        List of processing results
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all subject directories
    subject_dirs = sorted([d for d in input_path.iterdir() if d.is_dir() and d.name.startswith('subject_')])

    # Filter by subject range if specified
    if start_subject is not None or end_subject is not None:
        filtered_dirs = []
        for d in subject_dirs:
            try:
                subject_id = int(d.name.split('_')[1])
                if start_subject is not None and subject_id < start_subject:
                    continue
                if end_subject is not None and subject_id >= end_subject:
                    continue
                filtered_dirs.append(d)
            except (IndexError, ValueError):
                continue
        subject_dirs = filtered_dirs

    if not subject_dirs:
        print("No subject directories found!")
        return []

    # Set number of workers
    if num_workers is None:
        num_workers = cpu_count()

    print("="*80)
    print("BATCH NEURAL PEAK EXTRACTION")
    print("="*80)
    print(f"Input directory: {input_dir}")
    print(f"Total subjects to process: {len(subject_dirs)}")
    print(f"TR: {tr} seconds")
    print(f"Peaks per ROI: {num_peaks}")
    print(f"Workers (CPUs): {num_workers}")
    if start_subject is not None:
        print(f"Starting subject ID: {start_subject}")
    if end_subject is not None:
        print(f"Ending subject ID: {end_subject}")
    print("="*80)
    print()

    # Prepare arguments
    args_list = [(subject_dir, tr, num_peaks) for subject_dir in subject_dirs]

    # Process in parallel
    start_time = time.time()
    results = []

    print(f"Starting parallel processing with {num_workers} workers...")
    print()

    with Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_subject, args_list), 1):
            results.append(result)

            # Print status
            if result['status'] == 'success':
                status_msg = f"✓ {Path(result['subject_dir']).name} - {result['total_peaks']} peaks - {result['elapsed_time']:.1f}s"
            elif result['status'] == 'skipped':
                status_msg = f"⊘ {Path(result['subject_dir']).name} - {result['reason']}"
            else:
                status_msg = f"✗ {Path(result['subject_dir']).name} - FAILED: {result.get('error', 'Unknown')}"

            # Print progress every 10 subjects
            if i % 10 == 0 or i == len(subject_dirs):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(subject_dirs) - i) / rate if rate > 0 else 0

                print(f"Progress: {i}/{len(subject_dirs)} ({100*i/len(subject_dirs):.1f}%) | "
                      f"Rate: {rate:.2f} subj/s | "
                      f"ETA: {eta/60:.1f} min | "
                      f"{status_msg}")

    total_time = time.time() - start_time

    # Summary
    successful = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    failed = sum(1 for r in results if r['status'] == 'failed')

    if successful > 0:
        avg_time = np.mean([r['elapsed_time'] for r in results if r['status'] == 'success'])
        total_peaks = sum([r.get('total_peaks', 0) for r in results if r['status'] == 'success'])
    else:
        avg_time = 0
        total_peaks = 0

    print()
    print("="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"Total subjects processed: {len(subject_dirs)}")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    if successful > 0:
        print(f"Total peaks extracted: {total_peaks}")
        print(f"Average time per subject: {avg_time:.1f} seconds")
    print(f"Total elapsed time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
    if successful > 0:
        print(f"Effective rate: {successful/total_time:.2f} subjects/second")
    print("="*80)

    # Save summary
    summary = {
        'input_dir': str(input_dir),
        'tr': tr,
        'num_peaks': num_peaks,
        'num_workers': num_workers,
        'start_subject': start_subject,
        'end_subject': end_subject,
        'total_subjects': len(subject_dirs),
        'successful': successful,
        'skipped': skipped,
        'failed': failed,
        'total_peaks': total_peaks if successful > 0 else 0,
        'total_time_seconds': total_time,
        'total_time_hours': total_time / 3600,
        'avg_time_per_subject': avg_time if successful > 0 else 0,
        'timestamp': str(datetime.now()),
        'results': results
    }

    summary_path = Path(input_dir) / 'peak_extraction_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Batch extract neural peak metadata from existing simulation results'
    )
    parser.add_argument('--input', type=str, default='results/10k_subjects',
                       help='Input directory containing subject folders (default: results/10k_subjects)')
    parser.add_argument('--tr', type=float, default=0.8,
                       help='Repetition time in seconds (default: 0.8)')
    parser.add_argument('--num-peaks', type=int, default=12,
                       help='Number of peaks to extract per ROI (default: 12)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto-detect)')
    parser.add_argument('--start', type=int, default=None,
                       help='Starting subject ID for processing subset (default: process all)')
    parser.add_argument('--end', type=int, default=None,
                       help='Ending subject ID for processing subset (default: process all)')

    args = parser.parse_args()

    # Run batch extraction
    extract_peaks_batch(
        input_dir=args.input,
        tr=args.tr,
        num_peaks=args.num_peaks,
        num_workers=args.workers,
        start_subject=args.start,
        end_subject=args.end
    )


if __name__ == '__main__':
    main()
