"""
Peak analysis for LFP-like neural signals
Computes peak width, timing, and amplitude for each ROI
"""

import numpy as np
from scipy import signal
from pathlib import Path


class PeakAnalyzer:
    """Analyze peaks in LFP-like neural signals"""

    def __init__(self, data, sampling_rate=16000.0):
        """
        Initialize peak analyzer

        Parameters:
        -----------
        data : np.ndarray
            LFP-like signal data with shape (ROI, timepoints)
        sampling_rate : float
            Sampling rate in Hz (default: 16000 Hz)
        """
        self.data = data
        self.sampling_rate = sampling_rate
        self.dt = 1000.0 / sampling_rate  # Convert to milliseconds
        self.num_rois = data.shape[0]
        self.num_timepoints = data.shape[1]

    def detect_peaks(self, roi_idx, height=None, prominence=None, distance=None, width=None,
                     max_peaks=12):
        """
        Detect peaks in signal for a specific ROI

        Parameters:
        -----------
        roi_idx : int
            ROI index
        height : float or None
            Minimum peak height (default: mean + 1*std)
        prominence : float or None
            Minimum peak prominence (default: 0.5*std)
        distance : int or None
            Minimum distance between peaks in samples
        width : tuple or None
            Min/max peak width in samples
        max_peaks : int or None
            Maximum number of peaks to return (selects top peaks by prominence)
            Default: 12 (similar to BOLD analysis)

        Returns:
        --------
        peaks : dict
            Dictionary with peak information
        """
        signal_data = self.data[roi_idx, :]

        # Auto-set thresholds if not provided
        if height is None:
            height = np.mean(signal_data) + 1.0 * np.std(signal_data)
        if prominence is None:
            prominence = 0.5 * np.std(signal_data)

        # Find peaks
        peak_indices, properties = signal.find_peaks(
            signal_data,
            height=height,
            prominence=prominence,
            distance=distance,
            width=width
        )

        # If too many peaks found, select top N by prominence
        if max_peaks is not None and len(peak_indices) > max_peaks:
            # Get prominences and select top peaks
            prominences = properties['prominences']
            top_peak_mask = np.argsort(prominences)[-max_peaks:]
            # Sort selected peaks by time order
            peak_indices = np.sort(peak_indices[top_peak_mask])

            # Re-calculate properties for selected peaks only
            peak_indices, properties = signal.find_peaks(
                signal_data,
                height=height,
                prominence=prominence,
                distance=distance,
                width=width
            )
            # Filter again to get only the selected peaks
            prominences = properties['prominences']
            if len(peak_indices) > max_peaks:
                top_peak_mask = np.argsort(prominences)[-max_peaks:]
                peak_indices = np.sort(peak_indices[top_peak_mask])

        # Calculate peak widths at half maximum (FWHM)
        if len(peak_indices) > 0:
            widths, width_heights, left_ips, right_ips = signal.peak_widths(
                signal_data,
                peak_indices,
                rel_height=0.5  # Full Width at Half Maximum
            )
            # Get prominences for selected peaks
            _, temp_properties = signal.find_peaks(
                signal_data,
                height=height,
                prominence=prominence
            )
            # Match prominences to our selected peaks
            prominences_out = np.zeros(len(peak_indices))
            for i, peak_idx in enumerate(peak_indices):
                mask = temp_properties['peak_heights'] == signal_data[peak_idx]
                if np.any(mask):
                    prominences_out[i] = temp_properties['prominences'][mask][0]
        else:
            widths = np.array([])
            width_heights = np.array([])
            left_ips = np.array([])
            right_ips = np.array([])
            prominences_out = np.array([])

        return {
            'indices': peak_indices,  # Peak positions in samples
            'times': peak_indices * self.dt,  # Peak times in ms
            'amplitudes': signal_data[peak_indices] if len(peak_indices) > 0 else np.array([]),
            'widths': widths * self.dt,  # Peak widths in ms (FWHM)
            'prominences': prominences_out,
            'left_bases': properties.get('left_bases', np.array([])),
            'right_bases': properties.get('right_bases', np.array([])),
        }

    def analyze_all_rois(self, **kwargs):
        """
        Analyze peaks for all ROIs

        Parameters:
        -----------
        **kwargs : optional parameters for peak detection
            height, prominence, distance, width

        Returns:
        --------
        results : dict
            Dictionary with peak analysis results for all ROIs
        """
        print(f"Analyzing peaks for {self.num_rois} ROIs...")

        results = {
            'peak_times': [],      # List of arrays, one per ROI
            'peak_amplitudes': [], # List of arrays, one per ROI
            'peak_widths': [],     # List of arrays, one per ROI
            'num_peaks': [],       # Number of peaks per ROI
        }

        for roi_idx in range(self.num_rois):
            if roi_idx % 10 == 0:
                print(f"  Processing ROI {roi_idx}/{self.num_rois}...")

            peaks = self.detect_peaks(roi_idx, **kwargs)

            results['peak_times'].append(peaks['times'])
            results['peak_amplitudes'].append(peaks['amplitudes'])
            results['peak_widths'].append(peaks['widths'])
            results['num_peaks'].append(len(peaks['indices']))

        print(f"Peak analysis complete!")
        return results

    def compute_summary_statistics(self, results):
        """
        Compute summary statistics for peak features

        Parameters:
        -----------
        results : dict
            Results from analyze_all_rois()

        Returns:
        --------
        summary : dict
            Summary statistics (ROI, feature_stat) format
        """
        num_rois = len(results['peak_times'])

        # Initialize arrays for summary statistics
        summary = {
            'num_peaks': np.array(results['num_peaks']),  # (ROI,)
            'mean_amplitude': np.zeros(num_rois),
            'std_amplitude': np.zeros(num_rois),
            'mean_width': np.zeros(num_rois),
            'std_width': np.zeros(num_rois),
            'mean_interval': np.zeros(num_rois),  # Mean time between peaks
            'std_interval': np.zeros(num_rois),
        }

        for roi_idx in range(num_rois):
            amplitudes = results['peak_amplitudes'][roi_idx]
            widths = results['peak_widths'][roi_idx]
            times = results['peak_times'][roi_idx]

            # Amplitude statistics
            if len(amplitudes) > 0:
                summary['mean_amplitude'][roi_idx] = np.mean(amplitudes)
                summary['std_amplitude'][roi_idx] = np.std(amplitudes)
            else:
                summary['mean_amplitude'][roi_idx] = 0.0
                summary['std_amplitude'][roi_idx] = 0.0

            # Width statistics
            if len(widths) > 0:
                summary['mean_width'][roi_idx] = np.mean(widths)
                summary['std_width'][roi_idx] = np.std(widths)
            else:
                summary['mean_width'][roi_idx] = 0.0
                summary['std_width'][roi_idx] = 0.0

            # Inter-peak interval statistics
            if len(times) > 1:
                intervals = np.diff(times)
                summary['mean_interval'][roi_idx] = np.mean(intervals)
                summary['std_interval'][roi_idx] = np.std(intervals)
            else:
                summary['mean_interval'][roi_idx] = 0.0
                summary['std_interval'][roi_idx] = 0.0

        return summary

    def save_results(self, results, summary, output_path):
        """
        Save peak analysis results to .npy files

        Parameters:
        -----------
        results : dict
            Results from analyze_all_rois()
        summary : dict
            Summary statistics from compute_summary_statistics()
        output_path : str or Path
            Base path for output files
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        base_name = output_path.stem

        print(f"\nSaving peak analysis results...")

        # Save summary statistics as arrays with shape (ROI,) or (ROI, n_stats)
        for key, value in summary.items():
            save_path = output_path.parent / f'{base_name}_peak_{key}.npy'
            np.save(save_path, value)
            print(f"  {save_path}")

        # Save detailed peak features (variable length per ROI)
        # Use object arrays to store variable-length arrays
        peak_times_arr = np.empty(self.num_rois, dtype=object)
        peak_amplitudes_arr = np.empty(self.num_rois, dtype=object)
        peak_widths_arr = np.empty(self.num_rois, dtype=object)

        for roi_idx in range(self.num_rois):
            peak_times_arr[roi_idx] = results['peak_times'][roi_idx]
            peak_amplitudes_arr[roi_idx] = results['peak_amplitudes'][roi_idx]
            peak_widths_arr[roi_idx] = results['peak_widths'][roi_idx]

        # Save detailed arrays
        np.save(output_path.parent / f'{base_name}_peak_times_all.npy', peak_times_arr)
        np.save(output_path.parent / f'{base_name}_peak_amplitudes_all.npy', peak_amplitudes_arr)
        np.save(output_path.parent / f'{base_name}_peak_widths_all.npy', peak_widths_arr)

        print(f"  {output_path.parent / f'{base_name}_peak_times_all.npy'}")
        print(f"  {output_path.parent / f'{base_name}_peak_amplitudes_all.npy'}")
        print(f"  {output_path.parent / f'{base_name}_peak_widths_all.npy'}")

        print(f"\nPeak analysis results saved successfully!")


def analyze_lfp_peaks(lfp_data, output_path, sampling_rate=16000.0, max_peaks=12, **peak_kwargs):
    """
    Convenience function to analyze LFP peaks and save results

    Parameters:
    -----------
    lfp_data : np.ndarray
        LFP-like signal data with shape (ROI, timepoints)
    output_path : str or Path
        Base path for output files
    sampling_rate : float
        Sampling rate in Hz
    max_peaks : int
        Maximum number of peaks per ROI (default: 12, similar to BOLD analysis)
    **peak_kwargs : optional parameters for peak detection

    Returns:
    --------
    summary : dict
        Summary statistics
    """
    print(f"  LFP peak analysis: limiting to {max_peaks} peaks per ROI")
    analyzer = PeakAnalyzer(lfp_data, sampling_rate)
    results = analyzer.analyze_all_rois(max_peaks=max_peaks, **peak_kwargs)
    summary = analyzer.compute_summary_statistics(results)
    analyzer.save_results(results, summary, output_path)

    return summary


if __name__ == '__main__':
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Analyze peaks in LFP-like neural signals')
    parser.add_argument('--input', type=str, required=True,
                       help='Input .npy file with raw neural data')
    parser.add_argument('--output', type=str, default=None,
                       help='Output base path (default: same as input)')
    parser.add_argument('--sampling-rate', type=float, default=16000.0,
                       help='Sampling rate in Hz (default: 16000)')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    data = np.load(args.input)
    print(f"Data shape: {data.shape}")

    # Set output path
    if args.output is None:
        args.output = args.input

    # Analyze peaks
    summary = analyze_lfp_peaks(data, args.output, args.sampling_rate)

    # Print summary
    print("\n" + "="*60)
    print("Peak Analysis Summary:")
    print(f"  Total ROIs: {len(summary['num_peaks'])}")
    print(f"  Mean peaks per ROI: {np.mean(summary['num_peaks']):.1f}")
    print(f"  Mean amplitude (across ROIs): {np.mean(summary['mean_amplitude']):.4f}")
    print(f"  Mean width (across ROIs): {np.mean(summary['mean_width']):.2f} ms")
    print("="*60)
