# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a computational neuroscience project implementing whole-brain simulations using **The Virtual Brain (TVB)** framework. The project simulates neural dynamics across large-scale brain networks using the HCP-MMP1 (Human Connectome Project Multi-Modal Parcellation) atlas with both 48-node and 360-node parcellations.

**Output Format**: All results are saved as NumPy `.npy` files with shape **(ROI, timepoints)**, along with `.json` metadata files.

## Architecture

### Core Components

1. **Simulation Engine (`src/simulation.py`)**
   - `BrainSimulation` class orchestrates entire simulation pipeline
   - Configures TVB components: connectivity, neural mass models, coupling, integration
   - Supports multiple neural mass models: Wong-Wang, Kuramoto, Generic 2D Oscillator
   - Uses Heun stochastic integrator with additive noise
   - Monitor: TemporalAverage with user-specified TR
   - **Output**: `.npy` file with shape (ROI, timepoints) + `.json` metadata

2. **Visualization (`src/visualization.py`)**
   - `SimulationVisualizer` class handles all plotting
   - Loads `.npy` files and associated `.json` metadata
   - Time series plots for regional activity
   - Structural connectivity matrices (linear and log-scale)
   - Functional connectivity (correlation matrices and distributions)
   - Power spectrum analysis (frequency domain)
   - `create_summary_report()` generates comprehensive visualization suite

3. **Analysis (`src/analysis.py`)**
   - `SimulationAnalyzer` class for quantitative metrics
   - Loads `.npy` files for analysis
   - Functional connectivity computation
   - Global synchrony (Kuramoto order parameter using Hilbert transform)
   - Metastability (standard deviation of synchrony over time)
   - Summary statistics aggregation

### Data Flow

```
Simulation → .npy + .json Storage → Visualization/Analysis
     ↓
Connectivity (structural)
Neural Mass Model (dynamics)
Monitor (temporal average at TR)
     ↓
Results: (ROI, timepoints) array
Metadata: JSON with parameters
Connectivity: Separate .npy file
```

### Key Design Patterns

- **Modular architecture**: Simulation, visualization, and analysis are independent
- **NumPy storage**: All results in `.npy` format with separate `.json` metadata
- **Flexible dimensions**: User specifies num_timepoints and TR directly
- **Standard output format**: Always (ROI, timepoints) for easy integration
- **CLI + Jupyter**: Supports both command-line workflows and interactive notebooks

## Commands

### Installation and Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Simulations

```bash
# 48-node, 375 timepoints, TR=0.8s (fMRI-like, fast ~2-5 min)
python src/simulation.py --nodes 48 --timepoints 375 --tr 0.8

# 360-node, 375 timepoints, TR=0.8s (full parcellation, ~10-20 min)
python src/simulation.py --nodes 360 --timepoints 375 --tr 0.8

# High temporal resolution: 3000 timepoints, TR=0.1s
python src/simulation.py --nodes 48 --timepoints 3000 --tr 0.1

# Alternative models
python src/simulation.py --nodes 48 --timepoints 375 --tr 0.8 --model kuramoto
python src/simulation.py --nodes 48 --timepoints 375 --tr 0.8 --model generic_2d_oscillator

# Custom output path
python src/simulation.py --nodes 48 --timepoints 375 --tr 0.8 --output results/custom.npy
```

### Visualization

```bash
# Generate complete visualization report
python src/visualization.py --input results/sim_48nodes_375tp_TR0.8.npy --output visualizations/sim_48
```

### Interactive Analysis

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/example_simulation.ipynb
```

### Loading Results in Python

```python
import numpy as np
import json

# Load time series data (ROI, timepoints)
data = np.load('results/sim_48nodes_375tp_TR0.8.npy')
print(data.shape)  # (48, 375)

# Access individual ROI time series
roi_0 = data[0, :]  # Time series for ROI 0

# Load metadata
with open('results/sim_48nodes_375tp_TR0.8.json') as f:
    metadata = json.load(f)
    print(f"TR: {metadata['TR']}, Model: {metadata['model_type']}")

# Load connectivity
conn = np.load('results/connectivity_48nodes.npy')
print(conn.shape)  # (48, 48)
```

## Important Implementation Details

### Output Format

**Critical**: All simulations output arrays with shape **(ROI, timepoints)**, not (timepoints, ROI).

- **ROI dimension** (first axis): Number of brain regions (48 or 360)
- **Timepoints dimension** (second axis): Number of time samples (e.g., 375, 3000)
- **Data type**: float64
- **Format**: NumPy binary (.npy)

### Simulation Parameters

- **Integration timestep**: dt = 2^-4 ms (~0.0625 ms) - fixed for numerical stability
- **Noise**: Additive Gaussian noise with σ = 0.001
- **Coupling strength**: a = 0.0152 (linear coupling)
- **Monitor**: TemporalAverage with period = TR * 1000 ms
- **Total duration**: num_timepoints * TR seconds

### Common Configurations

1. **fMRI-like** (recommended for functional connectivity studies):
   - Timepoints: 375
   - TR: 0.8 seconds
   - Duration: 300 seconds (5 minutes)

2. **High temporal resolution** (for detailed dynamics):
   - Timepoints: 3000
   - TR: 0.1 seconds
   - Duration: 300 seconds
   - Can downsample later using scipy.signal.decimate

### File Naming Convention

Default output: `sim_{nodes}nodes_{timepoints}tp_TR{tr}.npy`
- Example: `sim_48nodes_375tp_TR0.8.npy`

Associated files:
- Metadata: Same name with `.json` extension
- Connectivity: `connectivity_{nodes}nodes.npy`

### Node Configurations

- **48 nodes**: Downsampled connectivity via linear indexing for faster prototyping
- **360 nodes**: Full HCP-MMP1 parcellation (currently uses TVB default connectivity)

**NOTE**: For true HCP-MMP1 360-node simulations, replace connectivity loading in `setup_connectivity()` with actual HCP-MMP1 structural connectivity data.

### Neural Mass Models

- **Wong-Wang** (default): Reduced mean-field model for decision-making circuits
- **Kuramoto**: Oscillator model for phase synchronization
- **Generic2dOscillator**: General-purpose 2D dynamical system

## Development Workflow

### Adding New Neural Mass Models

1. Import model from `tvb.simulator.models`
2. Add case to `setup_model()` in `BrainSimulation` class
3. Update CLI choices in `simulation.py` main()

### Modifying Output Dimensions

All output handling is in `save_results()` method of `BrainSimulation` class. The code ensures:
1. Extracts time series from TemporalAverage monitor
2. Reshapes to (time, nodes)
3. Transposes to **(ROI, timepoints)**
4. Pads or truncates to exact num_timepoints

### Custom Connectivity

Replace `connectivity.Connectivity.from_file()` in `setup_connectivity()`:
```python
# Load custom connectivity
weights = np.load('path/to/weights.npy')
tract_lengths = np.load('path/to/tract_lengths.npy')
conn = connectivity.Connectivity(
    weights=weights,
    tract_lengths=tract_lengths,
    region_labels=region_labels,
    centres=centres
)
conn.configure()
```

### Downsampling High-Resolution Data

```python
from scipy import signal

# Load high-res data (e.g., 3000 timepoints at TR=0.1s)
data = np.load('results/sim_48nodes_3000tp_TR0.1.npy')

# Downsample to TR=0.8s (factor of 8)
data_downsampled = signal.decimate(data, q=8, axis=1)
# Result: (48, 375)
```

## File Organization

- `src/`: Core source code (simulation, visualization, analysis)
- `data/`: Custom connectivity matrices and parcellation data (user-provided)
- `results/`: Simulation output `.npy` and `.json` files
- `visualizations/`: Generated plots and figures
- `notebooks/`: Jupyter notebooks for interactive exploration

## Performance Considerations

- 48-node simulations: ~2-5 minutes on standard laptop
- 360-node simulations: ~10-30 minutes (scales with O(n²) for connectivity)
- Memory usage: ~200MB for 48 nodes, ~1-2GB for 360 nodes
- High timepoints (3000) take ~2-3x longer than 375 timepoints

## Common Issues

1. **TVB Import Errors**: Ensure TVB is properly installed: `pip install tvb-library tvb-data`
2. **Memory Issues**: Reduce num_timepoints or use fewer nodes
3. **Shape Mismatch**: Always check data.shape - should be (ROI, timepoints)
4. **Connectivity Shape Mismatch**: Verify connectivity matrix matches num_nodes
5. **Visualization Errors**: Check that .json metadata file exists alongside .npy file

## Key Differences from HDF5 Version

This version uses **NumPy arrays** instead of HDF5:
- Simpler file format (.npy + .json)
- Standard (ROI, timepoints) shape
- No h5py dependency needed
- Easier integration with other tools
- Direct numpy.load() access
