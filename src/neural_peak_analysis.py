"""
Neural Activity Peak Analysis Module
Analyzes neural activity peaks from brain simulation data (downsampled at TR)
"""

import numpy as np
from scipy import signal
from scipy.stats import describe
import json
from pathlib import Path


def find_neural_peaks(neural_data, sampling_rate=1.25, num_peaks=12, min_peak_distance=None):
    """
    Find peaks in neural activity signal for each ROI

    Parameters:
    -----------
    neural_data : ndarray
        Neural activity data with shape (ROI, timepoints)
    sampling_rate : float
        Sampling rate in Hz (default: 1.25 Hz for TR=0.8s)
    num_peaks : int
        Target number of peaks to find per ROI (default: 12)
    min_peak_distance : int
        Minimum distance between peaks in samples (default: auto-calculated)

    Returns:
    --------
    dict : Dictionary containing peak information for each ROI
    """
    num_rois, num_timepoints = neural_data.shape

    # Auto-calculate minimum peak distance if not provided
    # Ensure peaks are reasonably spaced (at least 5% of total time apart)
    if min_peak_distance is None:
        min_peak_distance = max(int(num_timepoints / (num_peaks * 2.5)), 1)

    # Time vector in seconds
    time_vec = np.arange(num_timepoints) / sampling_rate

    results = {
        'num_rois': num_rois,
        'num_timepoints': num_timepoints,
        'sampling_rate': sampling_rate,
        'time_duration': time_vec[-1],
        'roi_peaks': []
    }

    for roi_idx in range(num_rois):
        signal_data = neural_data[roi_idx, :]

        # Normalize signal for better peak detection
        signal_normalized = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-10)

        # Find peaks with prominence to get the most significant ones
        peaks, properties = signal.find_peaks(
            signal_normalized,
            distance=min_peak_distance,
            prominence=0.1  # Minimum prominence threshold
        )

        # If we found more peaks than needed, select the top N by prominence
        if len(peaks) > num_peaks:
            # Sort by prominence (highest first)
            top_indices = np.argsort(properties['prominences'])[-num_peaks:]
            # Sort selected peaks by time order
            peaks = np.sort(peaks[top_indices])
        elif len(peaks) < num_peaks:
            # If we found fewer peaks, try with lower prominence
            peaks_relaxed, properties_relaxed = signal.find_peaks(
                signal_normalized,
                distance=min_peak_distance,
                prominence=0.01  # Lower threshold
            )
            if len(peaks_relaxed) >= num_peaks:
                top_indices = np.argsort(properties_relaxed['prominences'])[-num_peaks:]
                peaks = np.sort(peaks_relaxed[top_indices])
            else:
                peaks = peaks_relaxed

        # Calculate peak properties
        peak_info = {
            'roi_index': int(roi_idx),
            'num_peaks_found': len(peaks),
            'peaks': []
        }

        for peak_idx, peak_pos in enumerate(peaks):
            # Peak timing
            peak_time = time_vec[peak_pos]

            # Peak amplitude (use original signal, not normalized)
            amplitude = float(signal_data[peak_pos])

            # Calculate peak width at half maximum (FWHM)
            widths, width_heights, left_ips, right_ips = signal.peak_widths(
                signal_data,
                [peak_pos],
                rel_height=0.5
            )

            # Convert width from samples to seconds
            width_seconds = float(widths[0] / sampling_rate)

            peak_data = {
                'peak_number': int(peak_idx + 1),
                'sample_index': int(peak_pos),
                'timing_seconds': float(peak_time),
                'amplitude': amplitude,
                'width_seconds': width_seconds,
                'width_samples': float(widths[0]),
                'normalized_amplitude': float(signal_normalized[peak_pos])
            }

            peak_info['peaks'].append(peak_data)

        # Calculate statistics for this ROI
        if peak_info['peaks']:
            amplitudes = [p['amplitude'] for p in peak_info['peaks']]
            widths = [p['width_seconds'] for p in peak_info['peaks']]
            timings = [p['timing_seconds'] for p in peak_info['peaks']]

            # Inter-peak intervals
            if len(timings) > 1:
                intervals = np.diff(timings)
                peak_info['mean_interval'] = float(np.mean(intervals))
                peak_info['std_interval'] = float(np.std(intervals))
            else:
                peak_info['mean_interval'] = None
                peak_info['std_interval'] = None

            peak_info['statistics'] = {
                'amplitude': {
                    'mean': float(np.mean(amplitudes)),
                    'std': float(np.std(amplitudes)),
                    'min': float(np.min(amplitudes)),
                    'max': float(np.max(amplitudes))
                },
                'width': {
                    'mean': float(np.mean(widths)),
                    'std': float(np.std(widths)),
                    'min': float(np.min(widths)),
                    'max': float(np.max(widths))
                }
            }

        results['roi_peaks'].append(peak_info)

    # Calculate global statistics across all ROIs
    all_amplitudes = []
    all_widths = []
    all_intervals = []

    for roi_data in results['roi_peaks']:
        if roi_data['peaks']:
            all_amplitudes.extend([p['amplitude'] for p in roi_data['peaks']])
            all_widths.extend([p['width_seconds'] for p in roi_data['peaks']])
            if roi_data.get('mean_interval') is not None:
                # Collect individual intervals, not just means
                if len(roi_data['peaks']) > 1:
                    timings = [p['timing_seconds'] for p in roi_data['peaks']]
                    all_intervals.extend(np.diff(timings).tolist())

    results['global_statistics'] = {
        'total_peaks_found': sum([len(roi['peaks']) for roi in results['roi_peaks']]),
        'mean_peaks_per_roi': float(np.mean([len(roi['peaks']) for roi in results['roi_peaks']])),
        'amplitude': {
            'mean': float(np.mean(all_amplitudes)) if all_amplitudes else None,
            'std': float(np.std(all_amplitudes)) if all_amplitudes else None,
            'min': float(np.min(all_amplitudes)) if all_amplitudes else None,
            'max': float(np.max(all_amplitudes)) if all_amplitudes else None
        },
        'width': {
            'mean': float(np.mean(all_widths)) if all_widths else None,
            'std': float(np.std(all_widths)) if all_widths else None,
            'min': float(np.min(all_widths)) if all_widths else None,
            'max': float(np.max(all_widths)) if all_widths else None
        },
        'interval': {
            'mean': float(np.mean(all_intervals)) if all_intervals else None,
            'std': float(np.std(all_intervals)) if all_intervals else None
        }
    }

    return results


def save_peak_analysis(results, output_path):
    """
    Save peak analysis results to JSON file

    Parameters:
    -----------
    results : dict
        Peak analysis results from find_neural_peaks()
    output_path : str or Path
        Base output path (will add _neural_peaks.json)
    """
    output_path = Path(output_path)

    # Create output filename
    base_name = output_path.stem
    json_path = output_path.parent / f'{base_name}_neural_peaks.json'

    # Save to JSON
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    return json_path


def analyze_neural_peaks(neural_data, output_path, sampling_rate=1.25, num_peaks=12):
    """
    Main function to analyze neural activity peaks and save results

    Parameters:
    -----------
    neural_data : ndarray
        Neural activity data with shape (ROI, timepoints)
    output_path : str or Path
        Output path for saving results
    sampling_rate : float
        Sampling rate in Hz (default: 1.25 Hz for TR=0.8s)
    num_peaks : int
        Number of peaks to find per ROI (default: 12)

    Returns:
    --------
    dict : Peak analysis results
    """
    # Find peaks
    results = find_neural_peaks(neural_data, sampling_rate, num_peaks)

    # Save results
    json_path = save_peak_analysis(results, output_path)

    return results, json_path


if __name__ == '__main__':
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Analyze neural activity peaks from simulation data')
    parser.add_argument('--input', type=str, required=True,
                       help='Input neural data file (.npy)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: same as input)')
    parser.add_argument('--tr', type=float, default=0.8,
                       help='TR in seconds (default: 0.8)')
    parser.add_argument('--num-peaks', type=int, default=12,
                       help='Number of peaks to find per ROI (default: 12)')

    args = parser.parse_args()

    # Load data
    print(f"Loading neural data from {args.input}...")
    neural_data = np.load(args.input)

    # Calculate sampling rate from TR
    sampling_rate = 1.0 / args.tr

    # Set output path
    output_path = args.output if args.output else args.input

    # Analyze peaks
    results, json_path = analyze_neural_peaks(neural_data, output_path, sampling_rate, args.num_peaks)

    print(f"\nNeural Peak Analysis Summary:")
    print(f"  Total peaks found: {results['global_statistics']['total_peaks_found']}")
    print(f"  Mean peaks per ROI: {results['global_statistics']['mean_peaks_per_roi']:.1f}")
    print(f"  Results saved to: {json_path}")
    print("\nDone!")
