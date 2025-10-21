# Brain Simulation with The Virtual Brain (TVB)

A comprehensive whole-brain simulation project using The Virtual Brain (TVB) framework with HCP-MMP1 atlas parcellation.

## Overview

This project implements large-scale brain network simulations using:
- **The Virtual Brain (TVB)**: Neural mass modeling framework
- **HCP-MMP1 Atlas**: High-resolution cortical parcellation (48 and 360 nodes)
- **Connectivity**: Structural connectivity matrices
- **Neural Mass Models**: Configurable dynamics models
- **Output Format**: NumPy arrays with shape (ROI, timepoints)

## Features

- 48-node and 360-node whole brain simulations
- Configurable number of timepoints (375 or 3000) and TR (0.8s or 0.1s)
- HCP-MMP1 atlas parcellation
- Multiple neural mass models (Wong-Wang, Kuramoto, etc.)
- Time series analysis and visualization
- Functional connectivity analysis
- Output as .npy files with (ROI, timepoint) format

## Installation

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## Project Structure

```
brain_simulation_tvb/
├── src/                    # Source code
│   ├── simulation.py       # Main simulation script
│   ├── visualization.py    # Visualization utilities
│   └── analysis.py         # Analysis functions
├── data/                   # Data files (connectivity, parcellation)
├── results/                # Simulation outputs (.npy + .json)
├── visualizations/         # Generated plots
├── notebooks/              # Jupyter notebooks for exploration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Usage

### Basic Simulation

Run a 48-node simulation with 375 timepoints and TR=0.8s:
```bash
python src/simulation.py --nodes 48 --timepoints 375 --tr 0.8
```

Run a 360-node simulation with 3000 timepoints and TR=0.1s:
```bash
python src/simulation.py --nodes 360 --timepoints 3000 --tr 0.1
```

Custom output path:
```bash
python src/simulation.py --nodes 48 --timepoints 375 --tr 0.8 --output results/my_sim.npy
```

### Output Format

Each simulation produces **three types of neural signals**:

1. **Raw Neural Activity** (`*_raw_neural.npy`): High-resolution LFP-like signals @ 16 kHz
   - Shape: `(ROI, ~4,800,000)` for 300s simulation
   - **Automatic peak analysis**: Width, timing, amplitude computed automatically

2. **Downsampled Neural** (`*_neural.npy`): Neural activity at TR resolution
   - Shape: `(ROI, timepoints)` - e.g., `(48, 375)` or `(360, 3000)`

3. **BOLD Signal** (`*_bold.npy`): fMRI signal with HRF convolution
   - Shape: `(ROI, timepoints)` - e.g., `(48, 375)` or `(360, 3000)`

Plus:
- **Peak Features**: Automatically computed from LFP signals
  - Peak width (FWHM), timing, amplitude per ROI
  - See `PEAK_ANALYSIS.md` for details
- **Metadata**: `.json` file with HRF parameters and simulation info
- **Connectivity**: Structural connectivity matrix `(ROI, ROI)`

See `OUTPUT_FORMAT.md` for signal documentation and `PEAK_ANALYSIS.md` for peak analysis details.

### Load Results in Python

```python
import numpy as np
import json

# Load different signal types
raw_neural = np.load('results/sim_48nodes_375tp_TR0.8_raw_neural.npy')  # LFP-like
neural = np.load('results/sim_48nodes_375tp_TR0.8_neural.npy')  # Downsampled
bold = np.load('results/sim_48nodes_375tp_TR0.8_bold.npy')  # BOLD with HRF

print(f"Raw shape: {raw_neural.shape}")  # (48, ~4,800,000)
print(f"Neural shape: {neural.shape}")    # (48, 375)
print(f"BOLD shape: {bold.shape}")        # (48, 375)

# Load metadata with HRF parameters
with open('results/sim_48nodes_375tp_TR0.8_metadata.json') as f:
    metadata = json.load(f)
    print(f"HRF tau_s: {metadata['hrf_parameters']['tau_s']}")
```

### Visualization

Visualize simulation results:
```bash
python src/visualization.py --input results/sim_48nodes_375tp_TR0.8.npy --output visualizations/
```

### Interactive Exploration

Use Jupyter notebooks for interactive exploration:
```bash
jupyter notebook notebooks/example_simulation.ipynb
```

## Data Specifications

### Standard Configurations

1. **Configuration 1** (fMRI-like):
   - Nodes: 48 or 360
   - Timepoints: 375
   - TR: 0.8 seconds
   - Total duration: 300 seconds (5 minutes)

2. **Configuration 2** (High temporal resolution):
   - Nodes: 48 or 360
   - Timepoints: 3000
   - TR: 0.1 seconds
   - Total duration: 300 seconds (5 minutes)

### Downsampling

For Configuration 2, you can downsample to fMRI-like resolution:
```python
import numpy as np
from scipy import signal

# Load high-res data
data = np.load('results/sim_360nodes_3000tp_TR0.1.npy')

# Downsample from TR=0.1s to TR=0.8s (factor of 8)
data_downsampled = signal.decimate(data, q=8, axis=1)
# Result: (360, 375)
```

## HCP-MMP1 Atlas

The HCP-MMP1 (Human Connectome Project Multi-Modal Parcellation) is a high-resolution cortical parcellation:
- **360 regions**: Full parcellation (180 per hemisphere)
- **48 regions**: Reduced parcellation for faster simulations
- Based on multimodal neuroimaging data

## References

- Sanz Leon, P., et al. (2013). The Virtual Brain: a simulator of primate brain network dynamics. Frontiers in Neuroinformatics, 7, 10.
- Glasser, M. F., et al. (2016). A multi-modal parcellation of human cerebral cortex. Nature, 536(7615), 171-178.
