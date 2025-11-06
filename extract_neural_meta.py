"""
Extract neural activity meta information (peak timing, amplitude, width)
from existing simulation results

Based on 6mass_generator.py preprocessing style
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths
from pathlib import Path
import json
import argparse
from multiprocessing import Pool, cpu_count
import time


def calculate_neural_properties(neural_activity, TR=0.8, num_peaks=12, roi_idx=0):
    """
    Calculate neural activity properties for a single ROI

    Parameters:
    -----------
    neural_activity : np.ndarray
        Neural activity time series (1D array)
    TR : float
        Repetition time in seconds
    num_peaks : int
        Target number of peaks to extract
    roi_idx : int
        ROI index for logging

    Returns:
    --------
    dict : Peak properties
    """
    dt = TR  # Sampling interval

    # Peak detection (more sensitive threshold)
    threshold = np.std(neural_activity) * 0.3
    peaks, properties = find_peaks(
        neural_activity,
        prominence=threshold,
        distance=int(3.0 / dt)  # Minimum 3 seconds apart
    )

    if len(peaks) == 0:
        return {
            'peak_timings': [],
            'peak_amplitudes': [],
            'peak_widths': []
        }

    # If too many peaks, select top N by prominence
    if len(peaks) > num_peaks:
        top_indices = np.argsort(properties['prominences'])[-num_peaks:]
        peaks = np.sort(peaks[top_indices])

    # Peak timing (in seconds)
    peak_timings_sec = peaks * dt

    # Peak amplitudes
    peak_amplitudes = neural_activity[peaks]

    # Peak widths (FWHM)
    widths_result = peak_widths(neural_activity, peaks, rel_height=0.5)
    peak_widths_samples = widths_result[0]
    peak_widths_sec = peak_widths_samples * dt

    return {
        'peak_timings': peak_timings_sec.tolist(),
        'peak_amplitudes': peak_amplitudes.tolist(),
        'peak_widths': peak_widths_sec.tolist()
    }


def process_single_subject(args):
    """
    Process a single subject's neural activity

    Parameters:
    -----------
    args : tuple
        (subject_dir, TR, num_peaks)

    Returns:
    --------
    dict : Processing result
    """
    subject_dir, TR, num_peaks = args
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

        # Check if meta file already exists
        meta_file = subject_dir / f"{neural_file.stem}_meta.json"
        if meta_file.exists():
            return {
                'subject_dir': str(subject_dir),
                'status': 'skipped',
                'reason': 'Meta file already exists'
            }

        # Load neural activity (ROI, timepoints)
        neural_data = np.load(neural_file)
        num_rois, num_timepoints = neural_data.shape

        # Process each ROI
        peak_timings_list = []
        peak_amplitudes_list = []
        peak_widths_list = []

        for roi_idx in range(num_rois):
            neural_activity = neural_data[roi_idx, :]

            # Calculate properties
            props = calculate_neural_properties(
                neural_activity,
                TR=TR,
                num_peaks=num_peaks,
                roi_idx=roi_idx
            )

            peak_timings_list.append(props['peak_timings'])
            peak_amplitudes_list.append(props['peak_amplitudes'])
            peak_widths_list.append(props['peak_widths'])

        # Create metadata
        meta_data = {
            'num_rois': num_rois,
            'num_timepoints': num_timepoints,
            'TR': TR,
            'peak_timings': peak_timings_list,      # List of lists (ROI, variable peaks)
            'peak_amplitudes': peak_amplitudes_list,  # List of lists (ROI, variable peaks)
            'peak_widths': peak_widths_list          # List of lists (ROI, variable peaks)
        }

        # Calculate summary statistics
        all_timings = [t for roi_timings in peak_timings_list for t in roi_timings]
        all_amplitudes = [a for roi_amps in peak_amplitudes_list for a in roi_amps]
        all_widths = [w for roi_widths in peak_widths_list for w in roi_widths]

        meta_data['summary'] = {
            'total_peaks': len(all_timings),
            'mean_peaks_per_roi': len(all_timings) / num_rois if num_rois > 0 else 0,
            'amplitude_mean': float(np.mean(all_amplitudes)) if all_amplitudes else 0.0,
            'amplitude_std': float(np.std(all_amplitudes)) if all_amplitudes else 0.0,
            'width_mean': float(np.mean(all_widths)) if all_widths else 0.0,
            'width_std': float(np.std(all_widths)) if all_widths else 0.0
        }

        # Save to JSON
        with open(meta_file, 'w') as f:
            json.dump(meta_data, f, indent=2)

        elapsed_time = time.time() - start_time

        return {
            'subject_dir': str(subject_dir),
            'status': 'success',
            'num_rois': num_rois,
            'total_peaks': len(all_timings),
            'output_file': str(meta_file),
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


def main():
    parser = argparse.ArgumentParser(
        description='Extract neural activity meta information from simulation results'
    )
    parser.add_argument('--input', type=str, default='results/10k_subjects',
                       help='Input directory containing subject folders')
    parser.add_argument('--tr', type=float, default=0.8,
                       help='Repetition time in seconds (default: 0.8)')
    parser.add_argument('--num-peaks', type=int, default=12,
                       help='Target number of peaks per ROI (default: 12)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto)')
    parser.add_argument('--start', type=int, default=None,
                       help='Starting subject ID')
    parser.add_argument('--end', type=int, default=None,
                       help='Ending subject ID (exclusive)')

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input directory not found: {args.input}")
        return

    # Find all subject directories
    subject_dirs = sorted([d for d in input_path.iterdir()
                          if d.is_dir() and d.name.startswith('subject_')])

    # Filter by subject range
    if args.start is not None or args.end is not None:
        filtered_dirs = []
        for d in subject_dirs:
            try:
                subject_id = int(d.name.split('_')[1])
                if args.start is not None and subject_id < args.start:
                    continue
                if args.end is not None and subject_id >= args.end:
                    continue
                filtered_dirs.append(d)
            except (IndexError, ValueError):
                continue
        subject_dirs = filtered_dirs

    if not subject_dirs:
        print("No subject directories found!")
        return

    # Set workers
    num_workers = args.workers if args.workers else cpu_count()

    print("="*80)
    print("NEURAL ACTIVITY META EXTRACTION")
    print("="*80)
    print(f"Input directory: {args.input}")
    print(f"Total subjects: {len(subject_dirs)}")
    print(f"TR: {args.tr} seconds")
    print(f"Target peaks per ROI: {args.num_peaks}")
    print(f"Workers: {num_workers}")
    print("="*80)
    print()

    # Prepare arguments
    args_list = [(subject_dir, args.tr, args.num_peaks)
                 for subject_dir in subject_dirs]

    # Process in parallel
    start_time = time.time()
    results = []

    print(f"Starting parallel processing with {num_workers} workers...")
    print()

    with Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_subject, args_list), 1):
            results.append(result)

            # Status message
            if result['status'] == 'success':
                status_msg = f"✓ {Path(result['subject_dir']).name} - {result['total_peaks']} peaks - {result['elapsed_time']:.1f}s"
            elif result['status'] == 'skipped':
                status_msg = f"⊘ {Path(result['subject_dir']).name} - {result['reason']}"
            else:
                status_msg = f"✗ {Path(result['subject_dir']).name} - FAILED: {result.get('error', 'Unknown')}"

            # Progress every 10 subjects
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
    print(f"Total subjects: {len(subject_dirs)}")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    if successful > 0:
        print(f"Total peaks extracted: {total_peaks}")
        print(f"Average peaks per subject: {total_peaks/successful:.1f}")
        print(f"Average time per subject: {avg_time:.2f} seconds")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    if successful > 0:
        print(f"Processing rate: {successful/total_time:.2f} subjects/second")
    print("="*80)

    # Save summary
    summary = {
        'input_dir': str(args.input),
        'tr': args.tr,
        'num_peaks': args.num_peaks,
        'num_workers': num_workers,
        'total_subjects': len(subject_dirs),
        'successful': successful,
        'skipped': skipped,
        'failed': failed,
        'total_peaks': total_peaks if successful > 0 else 0,
        'total_time_seconds': total_time,
        'avg_time_per_subject': avg_time if successful > 0 else 0,
        'results': results
    }

    summary_path = input_path / 'neural_meta_extraction_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
