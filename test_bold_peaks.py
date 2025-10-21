"""
Test BOLD peak analysis with synthetic data
"""

import numpy as np
import sys
sys.path.insert(0, 'src')
from bold_peak_analysis import analyze_bold_peaks

# Create synthetic BOLD data
# 48 ROIs, 375 timepoints (5 minutes at TR=0.8s)
num_rois = 48
num_timepoints = 375
TR = 0.8
time = np.arange(num_timepoints) * TR

# Generate synthetic BOLD signals with peaks
np.random.seed(42)
bold_data = np.zeros((num_rois, num_timepoints))

for roi in range(num_rois):
    # Base oscillation with random frequency
    freq = 0.01 + np.random.rand() * 0.02  # 0.01-0.03 Hz
    signal = np.sin(2 * np.pi * freq * time)

    # Add random peaks
    for peak_idx in range(12):
        peak_time = np.random.uniform(0, time[-1])
        peak_width = np.random.uniform(5, 15)  # 5-15 seconds
        peak_amp = np.random.uniform(0.3, 1.0)

        gaussian = peak_amp * np.exp(-((time - peak_time) ** 2) / (2 * peak_width ** 2))
        signal += gaussian

    # Add noise
    signal += np.random.randn(num_timepoints) * 0.1

    bold_data[roi, :] = signal

# Save test data
np.save('results/test_bold_data.npy', bold_data)
print(f"Generated test BOLD data: {bold_data.shape}")
print(f"Saved to: results/test_bold_data.npy")

# Analyze peaks
sampling_rate = 1.0 / TR
results = analyze_bold_peaks(
    bold_data,
    'results/test_bold_data.npy',
    sampling_rate=sampling_rate,
    num_peaks=12
)

print("\n" + "="*60)
print("BOLD Peak Analysis Test Results")
print("="*60)
print(f"Total peaks found: {results['global_statistics']['total_peaks_found']}")
print(f"Mean peaks per ROI: {results['global_statistics']['mean_peaks_per_roi']:.1f}")
print(f"\nAmplitude statistics:")
print(f"  Mean: {results['global_statistics']['amplitude']['mean']:.4f}")
print(f"  Std:  {results['global_statistics']['amplitude']['std']:.4f}")
print(f"\nWidth statistics (seconds):")
print(f"  Mean: {results['global_statistics']['width']['mean']:.2f}")
print(f"  Std:  {results['global_statistics']['width']['std']:.2f}")
if results['global_statistics']['interval']['mean']:
    print(f"\nInter-peak interval (seconds):")
    print(f"  Mean: {results['global_statistics']['interval']['mean']:.2f}")
    print(f"  Std:  {results['global_statistics']['interval']['std']:.2f}")

# Show example ROI
print(f"\n" + "="*60)
print(f"Example: ROI 0")
print("="*60)
roi_0 = results['roi_peaks'][0]
print(f"Peaks found: {roi_0['num_peaks_found']}")
if roi_0['peaks']:
    print(f"\nFirst 3 peaks:")
    for peak in roi_0['peaks'][:3]:
        print(f"  Peak {peak['peak_number']}:")
        print(f"    Time: {peak['timing_seconds']:.2f} s")
        print(f"    Amplitude: {peak['amplitude']:.4f}")
        print(f"    Width: {peak['width_seconds']:.2f} s")

print("\nTest complete! Check results/test_bold_data_bold_peaks.json for full results.")
