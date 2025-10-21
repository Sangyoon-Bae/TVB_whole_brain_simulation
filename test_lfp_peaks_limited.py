"""
Test LFP peak analysis with limited peaks per ROI
"""

import numpy as np
import sys
sys.path.insert(0, 'src')
from peak_analysis import analyze_lfp_peaks

# Create synthetic LFP data (high resolution)
num_rois = 48
sampling_rate = 16000.0  # 16 kHz
duration = 5.0  # 5 seconds
num_timepoints = int(sampling_rate * duration)

print(f"Creating synthetic LFP data:")
print(f"  ROIs: {num_rois}")
print(f"  Sampling rate: {sampling_rate} Hz")
print(f"  Duration: {duration} seconds")
print(f"  Timepoints: {num_timepoints}")

# Generate synthetic LFP signals with many oscillations
np.random.seed(42)
lfp_data = np.zeros((num_rois, num_timepoints))
time = np.arange(num_timepoints) / sampling_rate

for roi in range(num_rois):
    # Multiple frequency components (realistic LFP)
    signal = 0.5 * np.sin(2 * np.pi * 10 * time)  # 10 Hz alpha
    signal += 0.3 * np.sin(2 * np.pi * 30 * time)  # 30 Hz beta
    signal += 0.2 * np.sin(2 * np.pi * 80 * time)  # 80 Hz gamma

    # Add random bursts
    for burst in range(20):
        burst_time = np.random.uniform(0, duration)
        burst_width = np.random.uniform(0.05, 0.2)
        burst_amp = np.random.uniform(1.0, 2.0)
        burst_freq = np.random.uniform(40, 100)

        gaussian = burst_amp * np.exp(-((time - burst_time) ** 2) / (2 * burst_width ** 2))
        burst_osc = gaussian * np.sin(2 * np.pi * burst_freq * time)
        signal += burst_osc

    # Add noise
    signal += np.random.randn(num_timepoints) * 0.1

    lfp_data[roi, :] = signal

# Save test data
np.save('results/test_lfp_data.npy', lfp_data)
print(f"\nSaved test LFP data: {lfp_data.shape}")

# Analyze peaks with limit
print("\n" + "="*60)
print("Analyzing LFP peaks with max_peaks=12")
print("="*60)

summary = analyze_lfp_peaks(
    lfp_data,
    'results/test_lfp_data.npy',
    sampling_rate=sampling_rate,
    max_peaks=12
)

print("\n" + "="*60)
print("LFP Peak Analysis Results (Limited)")
print("="*60)
print(f"Mean peaks per ROI: {np.mean(summary['num_peaks']):.1f}")
print(f"Max peaks in any ROI: {np.max(summary['num_peaks'])}")
print(f"Min peaks in any ROI: {np.min(summary['num_peaks'])}")
print(f"\nMean amplitude: {np.mean(summary['mean_amplitude']):.4f}")
print(f"Mean width: {np.mean(summary['mean_width']):.2f} ms")
print(f"Mean interval: {np.mean(summary['mean_interval']):.2f} ms")

# Check the actual peak counts
print("\nPeak counts per ROI:")
for i in range(min(10, num_rois)):
    print(f"  ROI {i}: {summary['num_peaks'][i]} peaks")

# Load and verify the saved data
import numpy as np
peak_amps = np.load('results/test_lfp_data_peak_amplitudes_all.npy', allow_pickle=True)
print(f"\n" + "="*60)
print("Verification:")
print(f"Total ROIs: {len(peak_amps)}")
for i in range(min(5, len(peak_amps))):
    print(f"  ROI {i}: {len(peak_amps[i])} peaks saved")

print("\n✓ Test complete!")
