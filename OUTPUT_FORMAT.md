# Output Format Documentation

## Overview

Each simulation produces **three types of signals** saved as separate `.npy` files, plus metadata with HRF parameters.

## Output Files

When you run a simulation like:
```bash
python src/simulation.py --nodes 48 --timepoints 375 --tr 0.8 --output results/my_sim.npy
```

You get these files:

### 1. Raw Neural Activity (LFP-like)
**File**: `results/my_sim_raw_neural.npy`

- **Description**: High temporal resolution neural activity (similar to local field potentials)
- **Shape**: `(ROI, many_timepoints)`
  - For 48 nodes, 375 TR timepoints: approximately `(48, 4,800,000)` timepoints
  - For 360 nodes, 375 TR timepoints: approximately `(360, 4,800,000)` timepoints
- **Sampling rate**: 16,000 Hz (dt = 0.0625 ms = 2^-4 ms)
- **Duration**: Same total duration as simulation
- **Use case**: Detailed neural dynamics analysis, high-frequency oscillations, LFP analysis

```python
import numpy as np

# Load raw neural activity
raw = np.load('results/my_sim_raw_neural.npy')
print(raw.shape)  # (48, ~4800000) for 300 second simulation

# Access single ROI high-res signal
roi_0_lfp = raw[0, :]  # LFP-like signal for ROI 0
```

### 2. Downsampled Neural Activity
**File**: `results/my_sim_neural.npy`

- **Description**: Neural activity downsampled to TR (temporal average)
- **Shape**: `(ROI, timepoints)`
  - For 48 nodes, 375 timepoints: `(48, 375)`
  - For 360 nodes, 375 timepoints: `(360, 375)`
- **Sampling rate**: 1/TR (e.g., 1.25 Hz for TR=0.8s)
- **Duration**: TR × timepoints seconds
- **Use case**: Standard fMRI-resolution neural activity, functional connectivity

```python
import numpy as np

# Load downsampled neural activity
neural = np.load('results/my_sim_neural.npy')
print(neural.shape)  # (48, 375)

# Compute functional connectivity
fc = np.corrcoef(neural)
```

### 3. BOLD Signal with HRF
**File**: `results/my_sim_bold.npy`

- **Description**: BOLD fMRI signal with hemodynamic response function (HRF) applied
- **Shape**: `(ROI, timepoints)`
  - For 48 nodes, 375 timepoints: `(48, 375)`
  - For 360 nodes, 375 timepoints: `(360, 375)`
- **Sampling rate**: 1/TR (e.g., 1.25 Hz for TR=0.8s)
- **HRF model**: Balloon-Windkessel hemodynamic model
- **Use case**: fMRI simulation, BOLD functional connectivity, comparison with real fMRI data

```python
import numpy as np

# Load BOLD signal
bold = np.load('results/my_sim_bold.npy')
print(bold.shape)  # (48, 375)

# Compute BOLD functional connectivity
bold_fc = np.corrcoef(bold)
```

### 4. Metadata with HRF Parameters
**File**: `results/my_sim_metadata.json`

```json
{
  "num_nodes": 48,
  "num_timepoints": 375,
  "TR": 0.8,
  "simulation_length_ms": 300000.0,
  "model_type": "wong_wang",
  "timestamp": "2025-01-15 12:30:45.123456",
  "integration_dt_ms": 0.0625,
  "sampling_frequency_hz": 16000.0,
  "outputs": {
    "raw_neural": "my_sim_raw_neural.npy",
    "neural": "my_sim_neural.npy",
    "bold": "my_sim_bold.npy"
  },
  "data_format": "(ROI, timepoints)",
  "signal_types": {
    "raw_neural": "High-resolution neural activity (LFP-like)",
    "neural": "Downsampled neural activity at TR=0.8s",
    "bold": "BOLD signal with HRF convolution at TR=0.8s"
  },
  "hrf_parameters": {
    "tau_s": 0.65,
    "tau_f": 0.41,
    "tau_o": 0.98,
    "alpha": 0.32,
    "E0": 0.4,
    "V0": 0.02,
    "TE": 0.04,
    "v0": 40.3,
    "r0": 25.0,
    "period_ms": 800.0
  }
}
```

### 5. Connectivity Matrix
**File**: `results/connectivity_48nodes.npy`

- **Description**: Structural connectivity (anatomical) matrix
- **Shape**: `(ROI, ROI)`
  - For 48 nodes: `(48, 48)`
  - For 360 nodes: `(360, 360)`

```python
import numpy as np

# Load connectivity
conn = np.load('results/connectivity_48nodes.npy')
print(conn.shape)  # (48, 48)
```

## HRF Parameters Explained

The Balloon-Windkessel hemodynamic model parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_s` | 0.65 s | Signal decay time constant |
| `tau_f` | 0.41 s | Feedback regulation time constant |
| `tau_o` | 0.98 s | Oxygen metabolism time constant |
| `alpha` | 0.32 | Grubb's exponent (vessel stiffness) |
| `E0` | 0.4 | Resting oxygen extraction fraction |
| `V0` | 0.02 | Resting blood volume fraction |
| `TE` | 0.04 s | Echo time (scanner parameter) |
| `v0` | 40.3 s^-1 | Frequency offset at rest |
| `r0` | 25.0 s^-1 | Slope of intravascular relaxation rate |

These parameters can be modified in `src/simulation.py` by accessing the BOLD monitor object.

## Data Dimensions Summary

### Example 1: 48 nodes, 375 timepoints, TR=0.8s

```
raw_neural:  (48, ~4,800,000)  @ 16,000 Hz
neural:      (48, 375)         @ 1.25 Hz
bold:        (48, 375)         @ 1.25 Hz
```

### Example 2: 360 nodes, 375 timepoints, TR=0.8s

```
raw_neural:  (360, ~4,800,000)  @ 16,000 Hz
neural:      (360, 375)         @ 1.25 Hz
bold:        (360, 375)         @ 1.25 Hz
```

### Example 3: 48 nodes, 3000 timepoints, TR=0.1s

```
raw_neural:  (48, ~4,800,000)  @ 16,000 Hz
neural:      (48, 3000)        @ 10 Hz
bold:        (48, 3000)        @ 10 Hz
```

## Loading All Outputs

```python
import numpy as np
import json

# Load all signal types
raw_neural = np.load('results/my_sim_raw_neural.npy')
neural = np.load('results/my_sim_neural.npy')
bold = np.load('results/my_sim_bold.npy')
connectivity = np.load('results/connectivity_48nodes.npy')

# Load metadata with HRF parameters
with open('results/my_sim_metadata.json') as f:
    metadata = json.load(f)

print("Raw neural shape:", raw_neural.shape)
print("Neural shape:", neural.shape)
print("BOLD shape:", bold.shape)
print("Connectivity shape:", connectivity.shape)
print("\nHRF tau_s:", metadata['hrf_parameters']['tau_s'])
```

## Signal Comparison

```python
import numpy as np
import matplotlib.pyplot as plt

# Load signals
raw = np.load('results/my_sim_raw_neural.npy')
neural = np.load('results/my_sim_neural.npy')
bold = np.load('results/my_sim_bold.npy')

# Select one ROI
roi_idx = 0

# Plot comparison
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

# Raw neural (show first 10 seconds only)
dt = 0.0625  # ms
time_raw = np.arange(160000) * dt / 1000  # Convert to seconds, first 10s
axes[0].plot(time_raw, raw[roi_idx, :160000], linewidth=0.5)
axes[0].set_ylabel('Raw Neural')
axes[0].set_title(f'ROI {roi_idx} - Signal Comparison')
axes[0].set_xlim([0, 10])

# Downsampled neural
TR = 0.8
time_neural = np.arange(len(neural[roi_idx])) * TR
axes[1].plot(time_neural, neural[roi_idx], linewidth=1.5)
axes[1].set_ylabel('Neural (downsampled)')

# BOLD
axes[2].plot(time_neural, bold[roi_idx], linewidth=1.5, color='red')
axes[2].set_ylabel('BOLD')
axes[2].set_xlabel('Time (seconds)')

plt.tight_layout()
plt.savefig('visualizations/signal_comparison.png', dpi=300)
plt.show()
```

## Storage Considerations

**Raw neural data can be very large!**

- 48 nodes, 300s simulation: ~1.8 GB
- 360 nodes, 300s simulation: ~13.8 GB

If storage is a concern:
1. Use shorter simulation durations for raw data
2. Process raw data immediately and delete
3. Only save neural and BOLD (much smaller)
4. Downsample raw data before saving

## Memory-Efficient Loading

```python
import numpy as np

# Load only a subset of raw data
# (load entire file, then slice - numpy loads lazily)
raw = np.load('results/my_sim_raw_neural.npy', mmap_mode='r')

# Access only first 10 seconds for ROI 0
dt = 0.0625  # ms
n_samples = int(10000 / dt)  # 10 seconds
roi_0_snippet = raw[0, :n_samples]

# Now you can work with just this small array
print(roi_0_snippet.shape)  # (160000,) instead of (4800000,)
```
