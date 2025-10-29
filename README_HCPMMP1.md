# HCP-MMP1 Atlas TVB Simulation (360 ROIs)

TVB whole-brain simulation using the HCP-MMP1 (Human Connectome Project Multi-Modal Parcellation 1.0) atlas with 360 cortical regions.

## Overview

This implementation provides high-resolution whole-brain simulations using the HCP-MMP1 cortical parcellation, which includes 360 bilateral cortical regions (180 per hemisphere).

## Atlas Details

**HCP-MMP1 Atlas (360 ROIs)**

The HCP-MMP1 parcellation is a state-of-the-art cortical parcellation based on:
- Multi-modal MRI data (architecture, function, connectivity, topography)
- 180 regions per hemisphere (360 total)
- High spatial resolution for detailed brain dynamics

**Organization:**
- **Left Hemisphere**: ROIs 0-179
- **Right Hemisphere**: ROIs 180-359

## Installation

Ensure you have the required dependencies:

```bash
pip install nibabel tvb-library tvb-data numpy scipy
```

## Usage

### Step 1: Generate Connectivity Matrices

Extract ROI information from the HCP-MMP1 atlas and generate connectivity matrices:

```bash
python src/hcpmmp1_loader.py --method distance
```

**Parameters:**
- `--method`: Connectivity generation method
  - `distance`: Distance-based connectivity (recommended)
  - `uniform`: Uniform random connectivity
  - `structured`: Structured with hemispheric bias
- `--output`: Output directory (default: `data/HCPMMP1`)

**Generated Files:**
- `mmp_weights_360.npy`: Connectivity weights matrix (360x360)
- `mmp_tract_lengths_360.npy`: Tract lengths matrix (360x360)
- `mmp_centers_360.npy`: ROI centers in MNI coordinates (360x3)
- `mmp_connectivity_360_metadata.json`: Metadata with region labels

### Step 2: Run TVB Simulation

Once connectivity is generated, run the brain simulation:

```bash
# Default: 375 timepoints, TR=0.8s, Wong-Wang model
python src/hcpmmp1_simulation.py

# Custom parameters
python src/hcpmmp1_simulation.py --timepoints 375 --tr 0.8 --model wong_wang

# Quick test (50 timepoints) - takes ~5-10 minutes
python src/hcpmmp1_simulation.py --timepoints 50 --tr 0.8 --output results/mmp_test.npy

# Full simulation (375 timepoints) - takes ~20-40 minutes
python src/hcpmmp1_simulation.py --timepoints 375 --tr 0.8
```

**Parameters:**
- `--timepoints`: Number of timepoints (default: 375)
- `--tr`: Repetition time in seconds (default: 0.8)
- `--model`: Neural mass model
  - `wong_wang`: Reduced Wong-Wang model (default)
  - `kuramoto`: Kuramoto oscillator model
  - `generic_2d_oscillator`: Generic 2D oscillator
- `--connectivity-dir`: Directory with connectivity files (default: `data/HCPMMP1`)
- `--output`: Output file path

**Note**: If connectivity files don't exist, the script will automatically generate them using distance-based method.

### Output Files

Simulation generates the following files:

```
results/
├── mmp_sim_360nodes_375tp_TR0.8_neural.npy    # Neural activity (360, 375)
├── mmp_sim_360nodes_375tp_TR0.8_bold.npy      # BOLD signal (360, 375)
├── mmp_sim_360nodes_375tp_TR0.8_metadata.json # Simulation metadata
└── mmp_connectivity_360nodes.npy              # Connectivity used
```

**Data Format**: All `.npy` files have shape `(ROI, timepoints)` = `(360, N)`

## Loading Results in Python

```python
import numpy as np
import json

# Load neural activity
neural_data = np.load('results/mmp_sim_360nodes_375tp_TR0.8_neural.npy')
print(f"Neural data shape: {neural_data.shape}")  # (360, 375)

# Load BOLD signal
bold_data = np.load('results/mmp_sim_360nodes_375tp_TR0.8_bold.npy')
print(f"BOLD data shape: {bold_data.shape}")  # (360, 375)

# Load metadata including region labels
with open('results/mmp_sim_360nodes_375tp_TR0.8_metadata.json') as f:
    metadata = json.load(f)
    region_labels = metadata['region_labels']
    print(f"Number of regions: {len(region_labels)}")
    print(f"Hemispheres: {metadata['hemispheres']}")

# Access specific ROI time series
roi_0_neural = neural_data[0, :]  # Left hemisphere ROI 0 neural activity
roi_180_bold = bold_data[180, :]  # Right hemisphere ROI 0 BOLD signal

# Analyze hemisphere separately
left_hemisphere = neural_data[0:180, :]   # Left: ROIs 0-179
right_hemisphere = neural_data[180:360, :] # Right: ROIs 180-359
```

## Region Labels

The 360 cortical regions follow this naming convention:

```python
[
  "L_MMP_001", "L_MMP_002", ..., "L_MMP_180",  # Left hemisphere
  "R_MMP_181", "R_MMP_182", ..., "R_MMP_360"   # Right hemisphere
]
```

Region labels are stored in the metadata JSON file.

## Connectivity Methods

### Distance-Based (Recommended)

```bash
python src/hcpmmp1_loader.py --method distance
```

- Weights inversely proportional to Euclidean distance
- Tract lengths = actual distances between ROI centers
- Physiologically plausible
- Best for realistic simulations

### Structured Connectivity

```bash
python src/hcpmmp1_loader.py --method structured
```

- Distance-based with hemispheric organization
- Stronger intra-hemispheric connections (1.5x boost)
- Weaker inter-hemispheric connections
- Models known hemispheric asymmetries

### Uniform Random

```bash
python src/hcpmmp1_loader.py --method uniform
```

- Random uniform weights
- Good for testing and parameter exploration

## Simulation Parameters

**Default Configuration:**
- Integration timestep: dt = 2^-4 ms (~0.0625 ms)
- Noise: Additive Gaussian noise, σ = 0.001
- Coupling strength: a = 0.0152 (linear coupling)
- Conduction speed: 3.0 m/s
- Monitor: TemporalAverage + BOLD at TR

**Recommended Configurations:**

1. **Quick test** (~5-10 minutes):
   ```bash
   python src/hcpmmp1_simulation.py --timepoints 50 --tr 0.8
   # Duration: 40 seconds
   ```

2. **Standard fMRI-like** (~20-30 minutes):
   ```bash
   python src/hcpmmp1_simulation.py --timepoints 375 --tr 0.8
   # Duration: 300 seconds (5 minutes)
   ```

3. **High temporal resolution** (~60-90 minutes):
   ```bash
   python src/hcpmmp1_simulation.py --timepoints 3000 --tr 0.1
   # Duration: 300 seconds, can downsample later
   ```

## Performance

- **360-node simulation**: ~10-30 minutes on standard laptop (for 375 timepoints)
- **Memory usage**: ~1-2 GB
- **Disk usage**: ~5-10 MB per simulation output
- **Scales as O(n²)** where n = number of nodes

## Comparison with Other Atlases

| Atlas | Regions | Coverage | Resolution | Use Case |
|-------|---------|----------|------------|----------|
| HarvardOxford | 48 | Cortical | Standard | Fast prototyping |
| Desikan-Killiany | 68 | Cortical | Standard | Clinical studies |
| TVB 192 | 192 | Cortical + Subcortical | Medium | General purpose |
| **HCP-MMP1** | **360** | **Cortical** | **High** | **High-res research** |

## Example Workflow

```bash
# 1. Generate connectivity (run once)
python src/hcpmmp1_loader.py --method distance

# 2. Run quick test simulation
python src/hcpmmp1_simulation.py --timepoints 50 --tr 0.8 --output results/test.npy

# 3. Run full simulation
python src/hcpmmp1_simulation.py --timepoints 375 --tr 0.8

# 4. Analyze results in Python
python
>>> import numpy as np
>>> data = np.load('results/mmp_sim_360nodes_375tp_TR0.8_neural.npy')
>>> print(f"Shape: {data.shape}")  # (360, 375)
>>> print(f"Mean activity: {data.mean():.4f}")
>>>
>>> # Hemisphere analysis
>>> left = data[0:180, :]
>>> right = data[180:360, :]
>>> print(f"Left mean: {left.mean():.4f}")
>>> print(f"Right mean: {right.mean():.4f}")
```

## Visualization

Use the existing visualization tools (may need adjustment for 360 ROIs):

```bash
python src/visualization.py \
  --input results/mmp_sim_360nodes_375tp_TR0.8_neural.npy \
  --output visualizations/mmp_sim_360
```

## Files Structure

```
brain_simulation_tvb/
├── src/
│   ├── hcpmmp1_loader.py          # Atlas loader & connectivity generator
│   ├── hcpmmp1_simulation.py      # Main simulation script
│   └── visualization.py           # Visualization tools (existing)
├── data/
│   └── HCPMMP1/
│       ├── MMP_in_MNI_corr.nii.gz        # Atlas file (user-provided)
│       ├── mmp_weights_360.npy           # Generated connectivity
│       ├── mmp_tract_lengths_360.npy     # Generated tract lengths
│       ├── mmp_centers_360.npy           # ROI centers
│       └── mmp_connectivity_360_metadata.json
└── results/
    └── mmp_sim_*.npy                # Simulation outputs
```

## Troubleshooting

**Issue: "Connectivity files not found"**
```bash
# Solution: Generate connectivity first
python src/hcpmmp1_loader.py --method distance
```

**Issue: "ModuleNotFoundError: No module named 'nibabel'"**
```bash
# Solution: Install nibabel
pip install nibabel
```

**Issue: Simulation runs out of memory**
```bash
# Solution: Reduce number of timepoints
python src/hcpmmp1_simulation.py --timepoints 100
```

**Issue: Simulation takes too long**
```bash
# Solution: Use fewer timepoints for testing
python src/hcpmmp1_simulation.py --timepoints 50
```

## Advanced Usage

### Hemisphere-Specific Analysis

```python
import numpy as np

# Load data
data = np.load('results/mmp_sim_360nodes_375tp_TR0.8_neural.npy')

# Separate hemispheres
left_hemisphere = data[0:180, :]    # ROIs 0-179
right_hemisphere = data[180:360, :] # ROIs 180-359

# Compute functional connectivity per hemisphere
from scipy.stats import pearsonr

# Intra-hemispheric FC
left_fc = np.corrcoef(left_hemisphere)    # (180, 180)
right_fc = np.corrcoef(right_hemisphere)  # (180, 180)

# Inter-hemispheric FC
inter_fc = np.corrcoef(data)  # (360, 360)
inter_fc_LR = inter_fc[0:180, 180:360]  # Left-Right connections

print(f"Left intra-hemispheric FC mean: {left_fc.mean():.4f}")
print(f"Right intra-hemispheric FC mean: {right_fc.mean():.4f}")
print(f"Inter-hemispheric FC mean: {inter_fc_LR.mean():.4f}")
```

### Custom Connectivity

```python
# Load and modify connectivity before simulation
import numpy as np

weights = np.load('data/HCPMMP1/mmp_weights_360.npy')

# Example: Increase connection strength by 20%
weights_modified = weights * 1.2

# Save modified connectivity
np.save('data/HCPMMP1/mmp_weights_360_modified.npy', weights_modified)

# Use in simulation by replacing the file
```

## Performance Tips

1. **Start with small timepoints** for testing (50-100)
2. **Use distance-based connectivity** for realistic results
3. **Monitor memory usage** - 360 ROIs can use 1-2GB RAM
4. **Run overnight** for long simulations (375+ timepoints)
5. **Save intermediate results** if running very long simulations

## References

- Glasser et al. (2016). "A multi-modal parcellation of human cerebral cortex." Nature 536:171-178
- HCP-MMP1 Atlas: https://balsa.wustl.edu/study/show/RVVG
- TVB Documentation: https://www.thevirtualbrain.org/

## Notes

- The HCP-MMP1 atlas is based on multi-modal neuroimaging data from the Human Connectome Project
- Distance-based connectivity is a simplification. For empirical connectivity, consider using DTI/DWI-derived structural connectivity
- ROI centers are computed as center of mass in MNI space from the atlas
- For more details on the parcellation, see Glasser et al. (2016)
