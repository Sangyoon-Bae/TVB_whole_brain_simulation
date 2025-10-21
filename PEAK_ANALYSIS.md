# Peak Analysis Documentation

## Overview

The simulation automatically computes **peak features** from LFP-like neural signals, including:
- **Peak timing**: When peaks occur (in milliseconds)
- **Peak amplitude**: Height of each peak
- **Peak width**: Full Width at Half Maximum (FWHM) in milliseconds

## Automatic Peak Analysis

Peak analysis is **automatically performed** during simulation. No additional steps needed!

```bash
# Run simulation - peaks are analyzed automatically
python src/simulation.py --nodes 48 --timepoints 375 --tr 0.8
```

## Output Files

### Summary Statistics (per ROI)

These files have shape `(ROI,)` - one value per brain region:

1. **`*_peak_num_peaks.npy`**
   - Number of detected peaks per ROI
   - Shape: `(ROI,)` - e.g., `(48,)` or `(360,)`

2. **`*_peak_mean_amplitude.npy`**
   - Mean peak amplitude per ROI
   - Shape: `(ROI,)`

3. **`*_peak_std_amplitude.npy`**
   - Standard deviation of peak amplitudes per ROI
   - Shape: `(ROI,)`

4. **`*_peak_mean_width.npy`**
   - Mean peak width (FWHM) in milliseconds per ROI
   - Shape: `(ROI,)`

5. **`*_peak_std_width.npy`**
   - Standard deviation of peak widths per ROI
   - Shape: `(ROI,)`

6. **`*_peak_mean_interval.npy`**
   - Mean time between consecutive peaks in milliseconds per ROI
   - Shape: `(ROI,)`

7. **`*_peak_std_interval.npy`**
   - Standard deviation of inter-peak intervals per ROI
   - Shape: `(ROI,)`

### Detailed Peak Information (all peaks)

These files contain **all individual peaks** for each ROI (variable length):

1. **`*_peak_times_all.npy`**
   - Timing of each peak in milliseconds
   - Shape: `(ROI,)` object array, where each element is an array of peak times

2. **`*_peak_amplitudes_all.npy`**
   - Amplitude of each peak
   - Shape: `(ROI,)` object array, where each element is an array of amplitudes

3. **`*_peak_widths_all.npy`**
   - Width (FWHM) of each peak in milliseconds
   - Shape: `(ROI,)` object array, where each element is an array of widths

## Loading Peak Analysis Results

### Summary Statistics

```python
import numpy as np

# Load summary statistics
num_peaks = np.load('results/sim_48nodes_375tp_TR0.8_peak_num_peaks.npy')
mean_amplitude = np.load('results/sim_48nodes_375tp_TR0.8_peak_mean_amplitude.npy')
mean_width = np.load('results/sim_48nodes_375tp_TR0.8_peak_mean_width.npy')
mean_interval = np.load('results/sim_48nodes_375tp_TR0.8_peak_mean_interval.npy')

print(f"Shape: {num_peaks.shape}")  # (48,)
print(f"ROI 0 has {num_peaks[0]} peaks")
print(f"ROI 0 mean amplitude: {mean_amplitude[0]:.4f}")
print(f"ROI 0 mean width: {mean_width[0]:.2f} ms")
print(f"ROI 0 mean interval: {mean_interval[0]:.2f} ms")
```

### Detailed Peak Information

```python
import numpy as np

# Load detailed peak information
peak_times = np.load('results/sim_48nodes_375tp_TR0.8_peak_times_all.npy',
                     allow_pickle=True)
peak_amplitudes = np.load('results/sim_48nodes_375tp_TR0.8_peak_amplitudes_all.npy',
                          allow_pickle=True)
peak_widths = np.load('results/sim_48nodes_375tp_TR0.8_peak_widths_all.npy',
                      allow_pickle=True)

# Access peaks for specific ROI
roi_idx = 0
roi_0_times = peak_times[roi_idx]
roi_0_amps = peak_amplitudes[roi_idx]
roi_0_widths = peak_widths[roi_idx]

print(f"ROI 0 has {len(roi_0_times)} peaks")
print(f"First 5 peak times (ms): {roi_0_times[:5]}")
print(f"First 5 peak amplitudes: {roi_0_amps[:5]}")
print(f"First 5 peak widths (ms): {roi_0_widths[:5]}")
```

## Visualization Examples

### Plot Peak Width Distribution

```python
import numpy as np
import matplotlib.pyplot as plt

# Load peak widths
mean_width = np.load('results/sim_48nodes_375tp_TR0.8_peak_mean_width.npy')
std_width = np.load('results/sim_48nodes_375tp_TR0.8_peak_std_width.npy')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mean widths
axes[0].bar(range(len(mean_width)), mean_width)
axes[0].set_xlabel('ROI')
axes[0].set_ylabel('Mean Peak Width (ms)')
axes[0].set_title('Mean Peak Width per ROI')
axes[0].grid(True, alpha=0.3)

# Distribution
axes[1].hist(mean_width, bins=20, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Mean Peak Width (ms)')
axes[1].set_ylabel('Count')
axes[1].set_title('Distribution of Mean Peak Widths')
axes[1].axvline(np.mean(mean_width), color='red', linestyle='--',
               label=f'Overall mean: {np.mean(mean_width):.2f} ms')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/peak_widths.png', dpi=300)
plt.show()
```

### Plot Peak Timing Histogram

```python
import numpy as np
import matplotlib.pyplot as plt

# Load detailed peak times
peak_times = np.load('results/sim_48nodes_375tp_TR0.8_peak_times_all.npy',
                     allow_pickle=True)

# Flatten all peak times
all_times = np.concatenate([peak_times[i] for i in range(len(peak_times))])

# Plot
plt.figure(figsize=(12, 5))
plt.hist(all_times / 1000.0, bins=100, edgecolor='black', alpha=0.7)
plt.xlabel('Time (seconds)')
plt.ylabel('Number of Peaks')
plt.title(f'Peak Timing Distribution (Total: {len(all_times)} peaks)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/peak_timing_distribution.png', dpi=300)
plt.show()
```

### Plot Individual ROI Peaks

```python
import numpy as np
import matplotlib.pyplot as plt

# Load raw neural data and peak info
raw = np.load('results/sim_48nodes_375tp_TR0.8_raw_neural.npy')
peak_times = np.load('results/sim_48nodes_375tp_TR0.8_peak_times_all.npy',
                     allow_pickle=True)
peak_amps = np.load('results/sim_48nodes_375tp_TR0.8_peak_amplitudes_all.npy',
                    allow_pickle=True)

# Select ROI and time window
roi_idx = 0
time_start_ms = 0
time_end_ms = 5000  # First 5 seconds

dt = 0.0625  # ms
n_samples = int((time_end_ms - time_start_ms) / dt)
time_ms = np.arange(n_samples) * dt + time_start_ms

# Plot
plt.figure(figsize=(14, 5))
plt.plot(time_ms, raw[roi_idx, :n_samples], linewidth=0.5, alpha=0.7,
         label='LFP Signal')

# Mark detected peaks
roi_peaks = peak_times[roi_idx]
roi_amps = peak_amps[roi_idx]
mask = (roi_peaks >= time_start_ms) & (roi_peaks < time_end_ms)
plt.scatter(roi_peaks[mask], roi_amps[mask], color='red', s=50,
           zorder=5, label='Detected Peaks')

plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title(f'ROI {roi_idx} - LFP Signal with Detected Peaks')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'visualizations/roi_{roi_idx}_peaks.png', dpi=300)
plt.show()
```

### Correlation: Peak Width vs Amplitude

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load summary statistics
mean_width = np.load('results/sim_48nodes_375tp_TR0.8_peak_mean_width.npy')
mean_amp = np.load('results/sim_48nodes_375tp_TR0.8_peak_mean_amplitude.npy')

# Compute correlation
r, p = pearsonr(mean_width, mean_amp)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(mean_width, mean_amp, alpha=0.6, s=50)
plt.xlabel('Mean Peak Width (ms)')
plt.ylabel('Mean Peak Amplitude')
plt.title(f'Peak Width vs Amplitude\nr = {r:.3f}, p = {p:.3e}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/width_vs_amplitude.png', dpi=300)
plt.show()
```

## Manual Peak Analysis

You can also run peak analysis separately on existing raw neural data:

```bash
python src/peak_analysis.py --input results/sim_48nodes_375tp_TR0.8_raw_neural.npy
```

## Peak Detection Parameters

The peak detection algorithm uses these default parameters:

- **Height threshold**: mean + 1.0 × std
- **Prominence threshold**: 0.5 × std
- **Width measurement**: Full Width at Half Maximum (FWHM)

To customize peak detection, modify `src/peak_analysis.py`:

```python
# In peak_analysis.py, modify detect_peaks() defaults
height = np.mean(signal_data) + 2.0 * np.std(signal_data)  # More stringent
prominence = 1.0 * np.std(signal_data)  # Higher prominence required
```

## Understanding Peak Features

### Peak Width (FWHM)
- **Definition**: Full Width at Half Maximum
- **Units**: Milliseconds
- **Interpretation**: Broader peaks indicate slower neural events

### Peak Amplitude
- **Definition**: Peak height above baseline
- **Units**: Same as input signal (arbitrary units from simulation)
- **Interpretation**: Larger amplitudes indicate stronger neural activation

### Peak Timing
- **Definition**: Time of peak occurrence
- **Units**: Milliseconds from simulation start
- **Interpretation**: Reveals temporal patterns of neural activity

### Inter-peak Interval
- **Definition**: Time between consecutive peaks
- **Units**: Milliseconds
- **Interpretation**: Indicates rhythmic/oscillatory properties
  - Shorter intervals → higher frequency oscillations
  - Regular intervals → periodic activity

## Complete Analysis Workflow

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Load all peak features
num_peaks = np.load('results/sim_48nodes_375tp_TR0.8_peak_num_peaks.npy')
mean_amp = np.load('results/sim_48nodes_375tp_TR0.8_peak_mean_amplitude.npy')
mean_width = np.load('results/sim_48nodes_375tp_TR0.8_peak_mean_width.npy')
mean_interval = np.load('results/sim_48nodes_375tp_TR0.8_peak_mean_interval.npy')

# 2. Create summary report
print("="*60)
print("PEAK ANALYSIS SUMMARY")
print("="*60)
print(f"Total ROIs: {len(num_peaks)}")
print(f"\nPeak Counts:")
print(f"  Mean: {np.mean(num_peaks):.1f} ± {np.std(num_peaks):.1f}")
print(f"  Range: {np.min(num_peaks):.0f} - {np.max(num_peaks):.0f}")
print(f"\nPeak Amplitudes:")
print(f"  Mean: {np.mean(mean_amp):.4f} ± {np.std(mean_amp):.4f}")
print(f"\nPeak Widths (ms):")
print(f"  Mean: {np.mean(mean_width):.2f} ± {np.std(mean_width):.2f}")
print(f"\nInter-peak Intervals (ms):")
print(f"  Mean: {np.mean(mean_interval):.2f} ± {np.std(mean_interval):.2f}")
print(f"  Frequency: {1000.0/np.mean(mean_interval):.2f} Hz")
print("="*60)

# 3. Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].bar(range(len(num_peaks)), num_peaks, alpha=0.7)
axes[0, 0].set_title('Number of Peaks per ROI')
axes[0, 0].set_xlabel('ROI')
axes[0, 0].set_ylabel('Count')

axes[0, 1].hist(mean_amp, bins=20, alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Mean Peak Amplitude Distribution')
axes[0, 1].set_xlabel('Amplitude')
axes[0, 1].set_ylabel('Frequency')

axes[1, 0].hist(mean_width, bins=20, alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Mean Peak Width Distribution')
axes[1, 0].set_xlabel('Width (ms)')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(mean_interval, bins=20, alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Mean Inter-Peak Interval Distribution')
axes[1, 1].set_xlabel('Interval (ms)')
axes[1, 1].set_ylabel('Frequency')

for ax in axes.flat:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/peak_analysis_summary.png', dpi=300)
plt.show()
```

## Storage Requirements

Peak analysis files are relatively small:

- Summary statistics: ~1-10 KB per file
- Detailed peak info: Depends on number of peaks
  - Typical: 1-5 MB per file
  - High peak count: Up to 50 MB

Much smaller than raw neural data!
