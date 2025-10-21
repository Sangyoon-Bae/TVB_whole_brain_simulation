# Complete Output Files Reference

## Quick Overview

Running a single simulation like:
```bash
python src/simulation.py --nodes 48 --timepoints 375 --tr 0.8
```

Produces **18+ output files** automatically!

## All Output Files

### 1. Neural Signals (3 files)

| File | Description | Shape | Sampling |
|------|-------------|-------|----------|
| `*_raw_neural.npy` | LFP-like high-res neural | `(48, ~4.8M)` | 16 kHz |
| `*_neural.npy` | Downsampled neural | `(48, 375)` | 1.25 Hz |
| `*_bold.npy` | BOLD with HRF | `(48, 375)` | 1.25 Hz |

### 2. Peak Features - Summary (7 files)

| File | Description | Shape |
|------|-------------|-------|
| `*_peak_num_peaks.npy` | Number of peaks per ROI | `(48,)` |
| `*_peak_mean_amplitude.npy` | Mean peak amplitude per ROI | `(48,)` |
| `*_peak_std_amplitude.npy` | Std peak amplitude per ROI | `(48,)` |
| `*_peak_mean_width.npy` | Mean peak width (FWHM) in ms | `(48,)` |
| `*_peak_std_width.npy` | Std peak width per ROI | `(48,)` |
| `*_peak_mean_interval.npy` | Mean inter-peak interval in ms | `(48,)` |
| `*_peak_std_interval.npy` | Std inter-peak interval | `(48,)` |

### 3. Peak Features - Detailed (3 files)

| File | Description | Type |
|------|-------------|------|
| `*_peak_times_all.npy` | All peak times (ms) | Object array `(48,)` |
| `*_peak_amplitudes_all.npy` | All peak amplitudes | Object array `(48,)` |
| `*_peak_widths_all.npy` | All peak widths (ms) | Object array `(48,)` |

### 4. Metadata & Connectivity (2 files)

| File | Description | Content |
|------|-------------|---------|
| `*_metadata.json` | Simulation metadata + HRF params | JSON |
| `connectivity_48nodes.npy` | Structural connectivity | `(48, 48)` |

## Complete Example Output

```
results/
├── sim_48nodes_375tp_TR0.8_raw_neural.npy          # (48, 4800000)
├── sim_48nodes_375tp_TR0.8_neural.npy              # (48, 375)
├── sim_48nodes_375tp_TR0.8_bold.npy                # (48, 375)
├── sim_48nodes_375tp_TR0.8_peak_num_peaks.npy      # (48,)
├── sim_48nodes_375tp_TR0.8_peak_mean_amplitude.npy # (48,)
├── sim_48nodes_375tp_TR0.8_peak_std_amplitude.npy  # (48,)
├── sim_48nodes_375tp_TR0.8_peak_mean_width.npy     # (48,)
├── sim_48nodes_375tp_TR0.8_peak_std_width.npy      # (48,)
├── sim_48nodes_375tp_TR0.8_peak_mean_interval.npy  # (48,)
├── sim_48nodes_375tp_TR0.8_peak_std_interval.npy   # (48,)
├── sim_48nodes_375tp_TR0.8_peak_times_all.npy      # (48,) object
├── sim_48nodes_375tp_TR0.8_peak_amplitudes_all.npy # (48,) object
├── sim_48nodes_375tp_TR0.8_peak_widths_all.npy     # (48,) object
├── sim_48nodes_375tp_TR0.8_metadata.json           # JSON
└── connectivity_48nodes.npy                         # (48, 48)
```

**Total: 15 files (18 for 360 nodes due to separate connectivity)**

## Load Everything at Once

```python
import numpy as np
import json
from pathlib import Path

def load_simulation_results(base_path):
    """Load all simulation outputs"""
    base = Path(base_path)
    stem = base.stem
    parent = base.parent

    results = {}

    # Neural signals
    results['raw_neural'] = np.load(parent / f'{stem}_raw_neural.npy')
    results['neural'] = np.load(parent / f'{stem}_neural.npy')
    results['bold'] = np.load(parent / f'{stem}_bold.npy')

    # Peak summary statistics
    results['peaks'] = {
        'num_peaks': np.load(parent / f'{stem}_peak_num_peaks.npy'),
        'mean_amplitude': np.load(parent / f'{stem}_peak_mean_amplitude.npy'),
        'std_amplitude': np.load(parent / f'{stem}_peak_std_amplitude.npy'),
        'mean_width': np.load(parent / f'{stem}_peak_mean_width.npy'),
        'std_width': np.load(parent / f'{stem}_peak_std_width.npy'),
        'mean_interval': np.load(parent / f'{stem}_peak_mean_interval.npy'),
        'std_interval': np.load(parent / f'{stem}_peak_std_interval.npy'),
    }

    # Peak detailed info
    results['peak_details'] = {
        'times': np.load(parent / f'{stem}_peak_times_all.npy',
                        allow_pickle=True),
        'amplitudes': np.load(parent / f'{stem}_peak_amplitudes_all.npy',
                             allow_pickle=True),
        'widths': np.load(parent / f'{stem}_peak_widths_all.npy',
                         allow_pickle=True),
    }

    # Metadata
    with open(parent / f'{stem}_metadata.json') as f:
        results['metadata'] = json.load(f)

    # Connectivity
    num_nodes = results['metadata']['num_nodes']
    results['connectivity'] = np.load(parent / f'connectivity_{num_nodes}nodes.npy')

    return results

# Usage
data = load_simulation_results('results/sim_48nodes_375tp_TR0.8.npy')

print(f"Raw neural shape: {data['raw_neural'].shape}")
print(f"Neural shape: {data['neural'].shape}")
print(f"BOLD shape: {data['bold'].shape}")
print(f"Number of peaks per ROI: {data['peaks']['num_peaks']}")
print(f"Mean peak width: {data['peaks']['mean_width'].mean():.2f} ms")
print(f"HRF tau_s: {data['metadata']['hrf_parameters']['tau_s']}")
```

## File Size Reference

For a **48-node, 375 timepoint (300s) simulation**:

| File Category | Approximate Size |
|---------------|------------------|
| Raw neural | 1.8 GB |
| Neural + BOLD | 300 KB |
| Peak summaries (7 files) | 3 KB |
| Peak details (3 files) | 1-5 MB |
| Metadata + Connectivity | 20 KB |
| **Total** | **~1.81 GB** |

For a **360-node simulation**, multiply by ~7.5.

## What to Keep vs Delete

### Keep Always
- `*_neural.npy` - Downsampled neural activity
- `*_bold.npy` - BOLD signal
- `*_peak_*.npy` - All peak features (small files)
- `*_metadata.json` - Metadata with HRF params
- `connectivity_*.npy` - Structural connectivity

**Storage: ~6 MB for 48 nodes, ~45 MB for 360 nodes**

### Optional (Large)
- `*_raw_neural.npy` - LFP-like signals

**Storage: ~1.8 GB for 48 nodes, ~13.8 GB for 360 nodes**

If storage is limited, process raw neural data immediately and delete, keeping only the peak features and downsampled signals.

## Quick Analysis Script

```python
import numpy as np
import matplotlib.pyplot as plt

# Load key outputs
neural = np.load('results/sim_48nodes_375tp_TR0.8_neural.npy')
bold = np.load('results/sim_48nodes_375tp_TR0.8_bold.npy')
peak_width = np.load('results/sim_48nodes_375tp_TR0.8_peak_mean_width.npy')
peak_amp = np.load('results/sim_48nodes_375tp_TR0.8_peak_mean_amplitude.npy')

# Compute functional connectivity
fc_neural = np.corrcoef(neural)
fc_bold = np.corrcoef(bold)

# Plot summary
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Neural FC
im0 = axes[0, 0].imshow(fc_neural, cmap='RdBu_r', vmin=-1, vmax=1)
axes[0, 0].set_title('Neural FC')
plt.colorbar(im0, ax=axes[0, 0])

# BOLD FC
im1 = axes[0, 1].imshow(fc_bold, cmap='RdBu_r', vmin=-1, vmax=1)
axes[0, 1].set_title('BOLD FC')
plt.colorbar(im1, ax=axes[0, 1])

# Peak widths
axes[1, 0].bar(range(len(peak_width)), peak_width)
axes[1, 0].set_title('Peak Width per ROI')
axes[1, 0].set_xlabel('ROI')
axes[1, 0].set_ylabel('Width (ms)')

# Peak amplitudes
axes[1, 1].scatter(peak_width, peak_amp, alpha=0.6)
axes[1, 1].set_title('Peak Width vs Amplitude')
axes[1, 1].set_xlabel('Width (ms)')
axes[1, 1].set_ylabel('Amplitude')

plt.tight_layout()
plt.savefig('visualizations/quick_summary.png', dpi=300)
plt.show()

print(f"Mean neural FC: {fc_neural[np.triu_indices_from(fc_neural, k=1)].mean():.3f}")
print(f"Mean BOLD FC: {fc_bold[np.triu_indices_from(fc_bold, k=1)].mean():.3f}")
print(f"Mean peak width: {peak_width.mean():.2f} ms")
print(f"Mean peak amplitude: {peak_amp.mean():.4f}")
```

## Documentation Index

- **`README.md`** - Main project documentation
- **`QUICKSTART.md`** - Quick start guide
- **`OUTPUT_FORMAT.md`** - Detailed signal format documentation
- **`PEAK_ANALYSIS.md`** - Peak analysis features and usage
- **`USAGE_EXAMPLES.md`** - Python code examples
- **`CLAUDE.md`** - AI assistant development guide
- **`COMPLETE_OUTPUTS.md`** - This file

## Support

For questions about specific outputs, see the detailed documentation files above.
