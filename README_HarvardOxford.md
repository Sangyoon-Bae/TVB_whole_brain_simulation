# HarvardOxford Atlas TVB Simulation

TVB simulation using the HarvardOxford cortical atlas with 48 regions of interest (ROIs).

## Overview

This implementation provides whole-brain simulations using the HarvardOxford cortical parcellation atlas, which includes 48 bilateral cortical regions (24 per hemisphere).

## Atlas Details

**HarvardOxford Cortical Atlas (48 ROIs)**

Bilateral regions organized by hemisphere:
- **Left Hemisphere (24 regions)**: Frontal, temporal, parietal, and occipital cortex
- **Right Hemisphere (24 regions)**: Mirror structure of left hemisphere

Includes regions such as:
- Frontal: Superior/Middle/Inferior Frontal Gyrus, Precentral Gyrus
- Temporal: Superior/Middle/Inferior Temporal Gyrus, Temporal Pole
- Parietal: Postcentral Gyrus, Superior Parietal Lobule, Supramarginal/Angular Gyrus
- Occipital: Lateral Occipital Cortex, Intracalcarine Cortex
- Insular Cortex

## Installation

Ensure you have the required dependencies:

```bash
pip install nibabel tvb-library tvb-data numpy scipy
```

## Usage

### Step 1: Generate Connectivity Matrices

First, extract ROI information from the HarvardOxford atlas and generate connectivity matrices:

```bash
python src/harvard_oxford_loader.py --threshold 25 --method distance
```

**Parameters:**
- `--threshold`: Probability threshold for atlas (0, 25, or 50). Default: 25
- `--method`: Connectivity generation method
  - `distance`: Distance-based connectivity (recommended)
  - `uniform`: Uniform random connectivity
  - `default`: Downsampled from TVB default connectivity
- `--output`: Output directory (default: `data/HarvardOxford`)

**Generated Files:**
- `ho_weights_48.npy`: Connectivity weights matrix (48x48)
- `ho_tract_lengths_48.npy`: Tract lengths matrix (48x48)
- `ho_centers_48.npy`: ROI centers in MNI coordinates (48x3)
- `ho_connectivity_48_metadata.json`: Metadata with region labels

### Step 2: Run TVB Simulation

Once connectivity is generated, run the brain simulation:

```bash
# Default: 375 timepoints, TR=0.8s, Wong-Wang model
python src/harvard_oxford_simulation.py

# Custom parameters
python src/harvard_oxford_simulation.py --timepoints 375 --tr 0.8 --model wong_wang

# Fast test simulation (100 timepoints)
python src/harvard_oxford_simulation.py --timepoints 100 --tr 0.8 --output results/ho_test.npy
```

**Parameters:**
- `--timepoints`: Number of timepoints (default: 375)
- `--tr`: Repetition time in seconds (default: 0.8)
- `--model`: Neural mass model
  - `wong_wang`: Reduced Wong-Wang model (default)
  - `kuramoto`: Kuramoto oscillator model
  - `generic_2d_oscillator`: Generic 2D oscillator
- `--connectivity-dir`: Directory with connectivity files (default: `data/HarvardOxford`)
- `--output`: Output file path

**Note**: If connectivity files don't exist, the script will automatically generate them using distance-based method.

### Output Files

Simulation generates the following files:

```
results/
├── ho_sim_48nodes_375tp_TR0.8_neural.npy    # Neural activity (48, 375)
├── ho_sim_48nodes_375tp_TR0.8_bold.npy      # BOLD signal (48, 375)
├── ho_sim_48nodes_375tp_TR0.8_metadata.json # Simulation metadata
└── ho_connectivity_48nodes.npy              # Connectivity used
```

**Data Format**: All `.npy` files have shape `(ROI, timepoints)` = `(48, N)`

## Loading Results in Python

```python
import numpy as np
import json

# Load neural activity
neural_data = np.load('results/ho_sim_48nodes_375tp_TR0.8_neural.npy')
print(f"Neural data shape: {neural_data.shape}")  # (48, 375)

# Load BOLD signal
bold_data = np.load('results/ho_sim_48nodes_375tp_TR0.8_bold.npy')
print(f"BOLD data shape: {bold_data.shape}")  # (48, 375)

# Load metadata including region labels
with open('results/ho_sim_48nodes_375tp_TR0.8_metadata.json') as f:
    metadata = json.load(f)
    region_labels = metadata['region_labels']
    print(f"Number of regions: {len(region_labels)}")
    print(f"First 5 regions: {region_labels[:5]}")

# Access specific ROI time series
roi_0_neural = neural_data[0, :]  # Left Frontal Pole neural activity
roi_0_bold = bold_data[0, :]      # Left Frontal Pole BOLD signal
```

## Region Labels

The 48 cortical regions follow this naming convention:

```python
[
  "L_Frontal_Pole", "L_Insular_Cortex", "L_Superior_Frontal_Gyrus", ...
  "R_Frontal_Pole", "R_Insular_Cortex", "R_Superior_Frontal_Gyrus", ...
]
```

Full list available in `harvard_oxford_loader.py` (`CORTICAL_LABELS`)

## Connectivity Methods

### Distance-Based (Recommended)

```bash
python src/harvard_oxford_loader.py --method distance
```

- Weights inversely proportional to Euclidean distance
- Tract lengths = actual distances between ROI centers
- Physiologically plausible
- Best for realistic simulations

### Uniform Random

```bash
python src/harvard_oxford_loader.py --method uniform
```

- Random uniform weights
- Good for testing and exploring parameter space

### TVB Default Downsampled

```bash
python src/harvard_oxford_loader.py --method default
```

- Uses TVB's built-in connectivity downsampled to 48 nodes
- Not specific to HarvardOxford anatomy

## Simulation Parameters

**Default Configuration:**
- Integration timestep: dt = 2^-4 ms (~0.0625 ms)
- Noise: Additive Gaussian noise, σ = 0.001
- Coupling strength: a = 0.0152 (linear coupling)
- Conduction speed: 3.0 m/s
- Monitor: TemporalAverage + BOLD at TR

**Recommended Configurations:**

1. **fMRI-like** (functional connectivity studies):
   ```bash
   python src/harvard_oxford_simulation.py --timepoints 375 --tr 0.8
   # Duration: 300 seconds (5 minutes)
   ```

2. **High temporal resolution** (detailed dynamics):
   ```bash
   python src/harvard_oxford_simulation.py --timepoints 3000 --tr 0.1
   # Duration: 300 seconds, can downsample later
   ```

3. **Quick test**:
   ```bash
   python src/harvard_oxford_simulation.py --timepoints 100 --tr 0.8
   # Duration: 80 seconds (~2-3 minutes to run)
   ```

## Performance

- **48-node simulation**: ~2-5 minutes on standard laptop
- **Memory usage**: ~200-300 MB
- **Disk usage**: ~1-2 MB per simulation output

## Comparison with Other Atlases

| Atlas | Regions | Coverage | Use Case |
|-------|---------|----------|----------|
| **HarvardOxford** | 48 | Cortical only | Standard cortical parcellation |
| Desikan-Killiany | 68 | Cortical only | Widely used in neuroimaging |
| HCP-MMP1 | 360 | Cortical only | High-resolution parcellation |

## Example Workflow

```bash
# 1. Generate connectivity
python src/harvard_oxford_loader.py --threshold 25 --method distance

# 2. Run simulation
python src/harvard_oxford_simulation.py --timepoints 375 --tr 0.8

# 3. Analyze results
python
>>> import numpy as np
>>> data = np.load('results/ho_sim_48nodes_375tp_TR0.8_neural.npy')
>>> print(f"Mean activity: {data.mean():.4f}")
>>> print(f"Std activity: {data.std():.4f}")
```

## Visualization

Use the existing visualization tools:

```bash
python src/visualization.py \
  --input results/ho_sim_48nodes_375tp_TR0.8_neural.npy \
  --output visualizations/ho_sim_48
```

## Files Structure

```
brain_simulation_tvb/
├── src/
│   ├── harvard_oxford_loader.py      # Atlas loader & connectivity generator
│   ├── harvard_oxford_simulation.py  # Main simulation script
│   └── visualization.py              # Visualization tools (existing)
├── data/
│   └── HarvardOxford/
│       ├── HarvardOxford-cort-*.nii.gz  # Atlas files
│       ├── ho_weights_48.npy            # Generated connectivity
│       ├── ho_tract_lengths_48.npy      # Generated tract lengths
│       ├── ho_centers_48.npy            # ROI centers
│       └── ho_connectivity_48_metadata.json
└── results/
    └── ho_sim_*.npy                   # Simulation outputs
```

## Troubleshooting

**Issue: "Connectivity files not found"**
```bash
# Solution: Generate connectivity first
python src/harvard_oxford_loader.py --method distance
```

**Issue: "ModuleNotFoundError: No module named 'nibabel'"**
```bash
# Solution: Install nibabel
pip install nibabel
```

**Issue: Simulation runs out of memory**
```bash
# Solution: Reduce number of timepoints
python src/harvard_oxford_simulation.py --timepoints 100
```

## References

- HarvardOxford Atlas: Part of FSL (FMRIB Software Library)
- TVB Documentation: https://www.thevirtualbrain.org/
- Paper: Sanz Leon et al. (2013), Front. Neuroinform. 7:10

## Notes

- The HarvardOxford atlas is probability-based. We use maximum probability maps with threshold=25% for ROI definition.
- Distance-based connectivity is a simplification. For empirical connectivity, consider using DTI/DWI-derived structural connectivity.
- ROI centers are computed as center of mass in MNI space.
