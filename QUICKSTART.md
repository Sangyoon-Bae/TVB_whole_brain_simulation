# Quick Start Guide

## Setup (First Time)

1. **Navigate to project directory**:
   ```bash
   cd /mnt/c/Users/stell/brain_simulation_tvb
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Run Your First Simulation

### Option 1: Command Line (Quick)

Run a 48-node simulation with 375 timepoints and TR=0.8s (takes ~2-5 minutes):
```bash
python src/simulation.py --nodes 48 --timepoints 375 --tr 0.8
```

Output files:
- `results/sim_48nodes_375tp_TR0.8_raw_neural.npy` - High-res neural (LFP-like)
- `results/sim_48nodes_375tp_TR0.8_neural.npy` - Downsampled neural (48, 375)
- `results/sim_48nodes_375tp_TR0.8_bold.npy` - BOLD signal with HRF (48, 375)
- `results/sim_48nodes_375tp_TR0.8_metadata.json` - Metadata + HRF parameters
- `results/connectivity_48nodes.npy` - Structural connectivity

Then visualize it:
```bash
python src/visualization.py --input results/sim_48nodes_375tp_TR0.8.npy --output visualizations/
```

### Option 2: High Temporal Resolution

Run with 3000 timepoints and TR=0.1s:
```bash
python src/simulation.py --nodes 48 --timepoints 3000 --tr 0.1
```

### Option 3: 360-node Simulation

Full HCP-MMP1 parcellation (takes longer):
```bash
python src/simulation.py --nodes 360 --timepoints 375 --tr 0.8
```

### Option 4: Jupyter Notebook (Interactive)

```bash
jupyter notebook notebooks/example_simulation.ipynb
```

Then run all cells to see the complete workflow!

## Load and Use Results

```python
import numpy as np
import json

# Load different signal types
raw = np.load('results/sim_48nodes_375tp_TR0.8_raw_neural.npy')  # LFP-like
neural = np.load('results/sim_48nodes_375tp_TR0.8_neural.npy')  # Downsampled
bold = np.load('results/sim_48nodes_375tp_TR0.8_bold.npy')  # BOLD+HRF

print(raw.shape)     # (48, ~4,800,000) @ 16 kHz
print(neural.shape)  # (48, 375) @ 1.25 Hz
print(bold.shape)    # (48, 375) @ 1.25 Hz

# Load metadata with HRF parameters
with open('results/sim_48nodes_375tp_TR0.8_metadata.json') as f:
    meta = json.load(f)
    print(f"TR: {meta['TR']}s")
    print(f"HRF tau_s: {meta['hrf_parameters']['tau_s']}")

# Access individual ROI
roi_0_bold = bold[0, :]  # BOLD time series for ROI 0
print(roi_0_bold.shape)  # (375,)
```

## Common Commands

```bash
# 48 nodes, 375 timepoints, TR=0.8s (default, fMRI-like)
python src/simulation.py --nodes 48 --timepoints 375 --tr 0.8

# 360 nodes, 375 timepoints, TR=0.8s (full parcellation)
python src/simulation.py --nodes 360 --timepoints 375 --tr 0.8

# 48 nodes, 3000 timepoints, TR=0.1s (high temporal resolution)
python src/simulation.py --nodes 48 --timepoints 3000 --tr 0.1

# Different neural mass model
python src/simulation.py --nodes 48 --timepoints 375 --tr 0.8 --model kuramoto

# Custom output path
python src/simulation.py --nodes 48 --timepoints 375 --tr 0.8 --output results/custom_name.npy
```

## Output Files

After running a simulation, you'll get:

1. **Raw Neural Activity** (`.npy`): LFP-like high-resolution signals
   - Shape: `(ROI, ~4,800,000)` @ 16 kHz
   - Example: `(48, 4,800,000)` for 300s simulation

2. **Downsampled Neural** (`.npy`): Neural activity at TR
   - Shape: `(ROI, timepoints)`
   - Example: `(48, 375)` or `(360, 3000)`

3. **BOLD Signal** (`.npy`): fMRI with HRF convolution
   - Shape: `(ROI, timepoints)`
   - Example: `(48, 375)` or `(360, 3000)`

4. **Metadata** (`.json`): HRF parameters + simulation info
   - HRF: tau_s, tau_f, tau_o, alpha, E0, V0, TE, v0, r0
   - Simulation: TR, model_type, sampling rates, etc.

5. **Connectivity** (`.npy`): Structural connectivity matrix
   - Shape: `(ROI, ROI)`

## Next Steps

- Try different neural mass models: `--model kuramoto` or `--model generic_2d_oscillator`
- Explore visualizations in `visualizations/` folder
- Check functional connectivity patterns
- Read `CLAUDE.md` for detailed architecture info
- Customize simulations by editing `src/simulation.py`
