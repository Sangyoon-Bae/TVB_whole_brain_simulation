# Usage Examples

## Quick Examples

### Example 1: Basic 48-node Simulation

```bash
# Run simulation
python src/simulation.py --nodes 48 --timepoints 375 --tr 0.8

# Load and inspect results
python -c "
import numpy as np
import json

data = np.load('results/sim_48nodes_375tp_TR0.8.npy')
print(f'Shape: {data.shape}')  # (48, 375)
print(f'Mean: {data.mean():.4f}')
print(f'Std: {data.std():.4f}')

with open('results/sim_48nodes_375tp_TR0.8.json') as f:
    meta = json.load(f)
    print(f'Model: {meta[\"model_type\"]}')
    print(f'TR: {meta[\"TR\"]}s')
"
```

### Example 2: 360-node Full Parcellation

```bash
# Run 360-node simulation
python src/simulation.py --nodes 360 --timepoints 375 --tr 0.8

# Visualize
python src/visualization.py --input results/sim_360nodes_375tp_TR0.8.npy --output visualizations/sim_360
```

### Example 3: High Temporal Resolution

```bash
# Run with 3000 timepoints at TR=0.1s
python src/simulation.py --nodes 48 --timepoints 3000 --tr 0.1

# Downsample to fMRI resolution
python -c "
from scipy import signal
import numpy as np

data = np.load('results/sim_48nodes_3000tp_TR0.1.npy')
print(f'Original shape: {data.shape}')  # (48, 3000)

# Downsample by factor of 8: TR 0.1s -> 0.8s
data_ds = signal.decimate(data, q=8, axis=1)
print(f'Downsampled shape: {data_ds.shape}')  # (48, 375)

np.save('results/sim_48nodes_375tp_TR0.8_downsampled.npy', data_ds)
"
```

## Python Integration Examples

### Loading and Basic Analysis

```python
import numpy as np
import json
import matplotlib.pyplot as plt

# Load data
data = np.load('results/sim_48nodes_375tp_TR0.8.npy')
meta_path = 'results/sim_48nodes_375tp_TR0.8.json'

with open(meta_path) as f:
    metadata = json.load(f)

print(f"Nodes: {metadata['num_nodes']}")
print(f"Timepoints: {metadata['num_timepoints']}")
print(f"TR: {metadata['TR']}s")
print(f"Shape: {data.shape}")

# Plot single ROI time series
plt.figure(figsize=(12, 4))
plt.plot(data[0, :])
plt.xlabel('Timepoint')
plt.ylabel('Activity')
plt.title('ROI 0 Time Series')
plt.show()
```

### Computing Functional Connectivity

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = np.load('results/sim_48nodes_375tp_TR0.8.npy')

# Compute functional connectivity (correlation between ROIs)
fc = np.corrcoef(data)

# Plot FC matrix
plt.figure(figsize=(10, 8))
sns.heatmap(fc, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={'label': 'Correlation'})
plt.title('Functional Connectivity Matrix')
plt.xlabel('ROI')
plt.ylabel('ROI')
plt.tight_layout()
plt.savefig('visualizations/my_fc_matrix.png', dpi=300)
plt.show()

# Summary statistics
fc_triu = fc[np.triu_indices_from(fc, k=1)]
print(f"Mean FC: {fc_triu.mean():.4f}")
print(f"Std FC: {fc_triu.std():.4f}")
```

### Using the Analysis Module

```python
from src.analysis import SimulationAnalyzer

# Load and analyze
analyzer = SimulationAnalyzer('results/sim_48nodes_375tp_TR0.8.npy')

# Get summary statistics
stats = analyzer.summary_statistics()
print("Summary Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value:.4f}")

# Compute specific metrics
fc = analyzer.compute_functional_connectivity()
metastability = analyzer.compute_metastability()

print(f"\nMetastability: {metastability:.4f}")
```

### Using the Visualization Module

```python
from src.visualization import SimulationVisualizer
import matplotlib.pyplot as plt

# Create visualizer
vis = SimulationVisualizer('results/sim_48nodes_375tp_TR0.8.npy')

# Plot specific ROIs
vis.plot_time_series(regions=[0, 5, 10, 15, 20])
plt.show()

# Plot functional connectivity
fig, fc = vis.plot_functional_connectivity()
plt.show()

# Generate complete report
vis.create_summary_report('visualizations/my_report')
```

### Batch Processing Multiple Simulations

```python
import numpy as np
from pathlib import Path

# Run multiple simulations with different models
models = ['wong_wang', 'kuramoto', 'generic_2d_oscillator']

for model in models:
    print(f"Running {model} simulation...")

    # Run via command line
    import subprocess
    subprocess.run([
        'python', 'src/simulation.py',
        '--nodes', '48',
        '--timepoints', '375',
        '--tr', '0.8',
        '--model', model,
        '--output', f'results/sim_48_{model}.npy'
    ])

# Load and compare results
results = {}
for model in models:
    data = np.load(f'results/sim_48_{model}.npy')
    results[model] = data
    print(f"{model}: mean={data.mean():.4f}, std={data.std():.4f}")
```

### Export to Other Formats

```python
import numpy as np
import pandas as pd

# Load data
data = np.load('results/sim_48nodes_375tp_TR0.8.npy')

# Convert to pandas DataFrame
df = pd.DataFrame(data.T, columns=[f'ROI_{i}' for i in range(data.shape[0])])
df.index.name = 'timepoint'

# Save as CSV
df.to_csv('results/sim_48nodes_375tp_TR0.8.csv')

# Save as Excel
df.to_excel('results/sim_48nodes_375tp_TR0.8.xlsx')

print(f"Exported to CSV and Excel")
print(f"DataFrame shape: {df.shape}")
```

## Advanced Examples

### Custom Connectivity Matrix

```python
import numpy as np
from src.simulation import BrainSimulation

# Create custom connectivity (example: random)
num_nodes = 48
custom_weights = np.random.rand(num_nodes, num_nodes)
custom_weights = (custom_weights + custom_weights.T) / 2  # Make symmetric
np.fill_diagonal(custom_weights, 0)  # Zero diagonal

# Save custom connectivity
np.save('data/custom_connectivity_48.npy', custom_weights)

# Modify simulation.py to load this custom connectivity
# Then run simulation
```

### Time-Frequency Analysis

```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Load data
data = np.load('results/sim_48nodes_3000tp_TR0.1.npy')
TR = 0.1

# Select one ROI
roi_ts = data[0, :]

# Compute spectrogram
f, t, Sxx = signal.spectrogram(roi_ts, fs=1/TR, nperseg=256)

# Plot
plt.figure(figsize=(12, 6))
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Spectrogram - ROI 0')
plt.colorbar(label='Power (dB)')
plt.ylim([0, 2])  # Focus on low frequencies
plt.tight_layout()
plt.savefig('visualizations/spectrogram_roi0.png', dpi=300)
plt.show()
```

### Dynamic Functional Connectivity

```python
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Load data
data = np.load('results/sim_48nodes_375tp_TR0.8.npy')

# Sliding window parameters
window_size = 50
step = 5

num_windows = (data.shape[1] - window_size) // step + 1
dfc_variance = []

for i in range(num_windows):
    start = i * step
    end = start + window_size

    # Compute FC for this window
    window_data = data[:, start:end]
    fc = np.corrcoef(window_data)

    # Store variance of FC
    fc_triu = fc[np.triu_indices_from(fc, k=1)]
    dfc_variance.append(np.std(fc_triu))

# Plot dynamic FC variance
plt.figure(figsize=(12, 4))
plt.plot(dfc_variance)
plt.xlabel('Window Index')
plt.ylabel('FC Std Dev')
plt.title('Dynamic Functional Connectivity Variability')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/dynamic_fc.png', dpi=300)
plt.show()
```
