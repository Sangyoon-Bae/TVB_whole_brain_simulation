"""
Visualization utilities for brain simulation results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import argparse


class SimulationVisualizer:
    """Visualize TVB simulation results"""

    def __init__(self, results_path):
        """
        Initialize visualizer with simulation results

        Parameters:
        -----------
        results_path : str
            Path to .npy file containing simulation results
        """
        self.results_path = Path(results_path)
        self.data = None
        self.metadata = None
        self.load_results()

    def load_results(self):
        """Load simulation results from .npy file"""
        print(f"Loading results from {self.results_path}...")

        # Load time series data (ROI, timepoints)
        self.data = np.load(self.results_path)

        # Load metadata from .json file
        metadata_path = self.results_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            # Infer metadata from data shape
            self.metadata = {
                'num_nodes': self.data.shape[0],
                'num_timepoints': self.data.shape[1],
                'TR': 'unknown',
                'model_type': 'unknown'
            }

        print(f"Loaded {self.metadata['num_nodes']}-node simulation")
        print(f"Data shape: {self.data.shape} (ROI, timepoints)")
        print(f"TR: {self.metadata.get('TR', 'unknown')} seconds")

    def plot_time_series(self, regions=None, save_path=None):
        """
        Plot time series for selected regions

        Parameters:
        -----------
        regions : list
            List of region indices to plot (None = first 10)
        save_path : str
            Path to save figure
        """
        # Data shape is (ROI, timepoints)
        num_regions, num_timepoints = self.data.shape

        # Select regions to plot
        if regions is None:
            regions = list(range(min(10, num_regions)))

        # Create figure
        fig, axes = plt.subplots(len(regions), 1, figsize=(14, 2*len(regions)),
                                sharex=True)
        if len(regions) == 1:
            axes = [axes]

        TR = self.metadata.get('TR', 1.0)
        time = np.arange(num_timepoints) * TR  # Time in seconds

        for idx, (ax, region) in enumerate(zip(axes, regions)):
            ax.plot(time, self.data[region, :], linewidth=0.8, alpha=0.8)
            ax.set_ylabel(f'ROI {region}', fontsize=10)
            ax.grid(True, alpha=0.3)

            if idx == 0:
                ax.set_title(f'{self.metadata["num_nodes"]}-node Simulation\n'
                           f'Model: {self.metadata.get("model_type", "unknown")} | '
                           f'TR: {TR}s',
                           fontsize=12, fontweight='bold')

        axes[-1].set_xlabel('Time (seconds)', fontsize=11)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved time series plot to {save_path}")

        return fig

    def plot_connectivity_matrix(self, save_path=None):
        """
        Plot structural connectivity matrix

        Parameters:
        -----------
        save_path : str
            Path to save figure
        """
        # Try to load connectivity file
        conn_path = self.results_path.parent / f'connectivity_{self.metadata["num_nodes"]}nodes.npy'
        if not conn_path.exists():
            print(f"Connectivity file not found: {conn_path}")
            return

        weights = np.load(conn_path)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot connectivity weights
        im1 = axes[0].imshow(weights, cmap='hot', aspect='auto')
        axes[0].set_title(f'Structural Connectivity Weights\n'
                         f'{self.metadata["num_nodes"]} nodes',
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Region', fontsize=10)
        axes[0].set_ylabel('Region', fontsize=10)
        plt.colorbar(im1, ax=axes[0], label='Connection Strength')

        # Plot log-scale connectivity
        weights_nonzero = weights.copy()
        weights_nonzero[weights_nonzero == 0] = np.nan
        im2 = axes[1].imshow(np.log10(weights_nonzero), cmap='viridis', aspect='auto')
        axes[1].set_title('Log-scale Connectivity Weights',
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Region', fontsize=10)
        axes[1].set_ylabel('Region', fontsize=10)
        plt.colorbar(im2, ax=axes[1], label='log10(Strength)')

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved connectivity matrix plot to {save_path}")

        return fig

    def plot_functional_connectivity(self, save_path=None):
        """
        Compute and plot functional connectivity (correlation matrix)

        Parameters:
        -----------
        save_path : str
            Path to save figure
        """
        # Data shape is (ROI, timepoints)
        # Compute correlation across timepoints
        fc = np.corrcoef(self.data)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Full FC matrix
        im1 = axes[0].imshow(fc, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        axes[0].set_title(f'Functional Connectivity\n'
                         f'{self.metadata["num_nodes"]} nodes - {monitor} Monitor',
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Region', fontsize=10)
        axes[0].set_ylabel('Region', fontsize=10)
        plt.colorbar(im1, ax=axes[0], label='Correlation')

        # FC histogram
        fc_triu = fc[np.triu_indices_from(fc, k=1)]
        axes[1].hist(fc_triu, bins=50, alpha=0.7, edgecolor='black')
        axes[1].set_title('FC Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Correlation Coefficient', fontsize=10)
        axes[1].set_ylabel('Count', fontsize=10)
        axes[1].axvline(fc_triu.mean(), color='red', linestyle='--',
                       label=f'Mean: {fc_triu.mean():.3f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved functional connectivity plot to {save_path}")

        return fig, fc

    def plot_power_spectrum(self, regions=None, save_path=None):
        """
        Plot power spectrum for selected regions

        Parameters:
        -----------
        regions : list
            Region indices to plot
        save_path : str
            Path to save figure
        """
        if regions is None:
            regions = list(range(min(5, self.data.shape[0])))

        TR = self.metadata.get('TR', 1.0)
        sampling_freq = 1.0 / TR  # Hz

        fig, ax = plt.subplots(figsize=(12, 6))

        for region in regions:
            # Compute FFT on time series for this ROI
            fft_vals = np.fft.rfft(self.data[region, :])
            fft_freq = np.fft.rfftfreq(self.data.shape[1], d=TR)
            power = np.abs(fft_vals)**2

            ax.semilogy(fft_freq, power, alpha=0.7, label=f'ROI {region}')

        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('Power', fontsize=11)
        ax.set_title(f'Power Spectrum\n'
                    f'{self.metadata["num_nodes"]} nodes | TR: {TR}s',
                    fontsize=12, fontweight='bold')
        ax.set_xlim([0, min(0.5 * sampling_freq, 0.5)])  # Nyquist frequency
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved power spectrum plot to {save_path}")

        return fig

    def create_summary_report(self, output_dir):
        """
        Create comprehensive visualization report

        Parameters:
        -----------
        output_dir : str
            Directory to save all plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*60)
        print("Creating visualization report...")
        print("="*60 + "\n")

        # Time series
        self.plot_time_series(save_path=output_dir / 'time_series.png')
        plt.close()

        # Connectivity
        self.plot_connectivity_matrix(save_path=output_dir / 'connectivity_matrix.png')
        plt.close()

        # Functional connectivity
        self.plot_functional_connectivity(save_path=output_dir / 'functional_connectivity.png')
        plt.close()

        # Power spectrum
        self.plot_power_spectrum(save_path=output_dir / 'power_spectrum.png')
        plt.close()

        print("\n" + "="*60)
        print(f"Visualization report saved to {output_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Visualize brain simulation results')
    parser.add_argument('--input', type=str, required=True,
                       help='Input .npy file with simulation results')
    parser.add_argument('--output', type=str, default='visualizations',
                       help='Output directory for plots')

    args = parser.parse_args()

    # Create visualizer and generate report
    vis = SimulationVisualizer(args.input)
    vis.create_summary_report(args.output)


if __name__ == '__main__':
    main()
