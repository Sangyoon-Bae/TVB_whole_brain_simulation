"""
Brain Simulation with The Virtual Brain (TVB)
Implements whole-brain simulations with HCP-MMP1 atlas parcellation
"""

import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime
from scipy import signal

from tvb.simulator.lab import *
from tvb.datatypes import connectivity
from peak_analysis import analyze_lfp_peaks
from bold_peak_analysis import analyze_bold_peaks
from neural_peak_analysis import analyze_neural_peaks


class BrainSimulation:
    """Whole-brain simulation using TVB framework"""

    def __init__(self, num_nodes=68, num_timepoints=375, TR=0.8, model_type='wong_wang'):
        """
        Initialize brain simulation

        Parameters:
        -----------
        num_nodes : int
            Number of brain regions (48, 68, or 360)
            - 48: Downsampled from default TVB connectivity
            - 68: Desikan-Killiany atlas (cortical regions)
            - 360: HCP-MMP1 full parcellation
        num_timepoints : int
            Number of timepoints to generate (375 or 3000)
        TR : float
            Repetition time in seconds (0.8 or 0.1)
        model_type : str
            Neural mass model type ('wong_wang', 'kuramoto', 'generic_2d_oscillator')
        """
        self.num_nodes = num_nodes
        self.num_timepoints = num_timepoints
        self.TR = TR
        self.simulation_length = num_timepoints * TR * 1000  # Convert to milliseconds
        self.model_type = model_type
        self.simulator = None
        self.results = None

    def setup_connectivity(self):
        """Setup structural connectivity"""
        print(f"Setting up connectivity for {self.num_nodes} nodes...")

        if self.num_nodes == 68:
            # Use Desikan-Killiany atlas (68 cortical regions)
            print("Using Desikan-Killiany atlas (68 cortical regions)")
            conn = connectivity.Connectivity.from_file('connectivity_68.zip')

        elif self.num_nodes == 48:
            # Use reduced connectivity (downsample from default 76 regions)
            print("Using downsampled connectivity (48 regions)")
            conn = connectivity.Connectivity.from_file()
            indices = np.linspace(0, len(conn.weights) - 1, 48, dtype=int)
            conn.weights = conn.weights[indices][:, indices]
            conn.tract_lengths = conn.tract_lengths[indices][:, indices]
            conn.region_labels = conn.region_labels[indices]
            conn.centres = conn.centres[indices]

        elif self.num_nodes == 360:
            # For HCP-MMP1 360 nodes, we would need the actual connectivity matrix
            # For now, we'll use TVB's default 76-region connectivity
            # In production, load actual HCP-MMP1 connectivity here
            print("Note: Using TVB default connectivity (76 regions).")
            print("      For true HCP-MMP1 360 nodes, load actual HCP-MMP1 connectivity matrix.")
            conn = connectivity.Connectivity.from_file()

        else:
            raise ValueError(f"Unsupported number of nodes: {self.num_nodes}. Choose 48, 68, or 360.")

        conn.configure()
        return conn

    def setup_model(self):
        """Setup neural mass model"""
        print(f"Setting up {self.model_type} model...")

        if self.model_type == 'wong_wang':
            model = models.ReducedWongWang()
        elif self.model_type == 'kuramoto':
            model = models.Kuramoto()
        elif self.model_type == 'generic_2d_oscillator':
            model = models.Generic2dOscillator()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return model

    def setup_coupling(self):
        """Setup coupling between regions"""
        coupling_func = coupling.Linear(a=np.array([0.0152]))
        return coupling_func

    def setup_integrator(self):
        """Setup integration scheme"""
        # Heun stochastic integrator with noise
        heunint = integrators.HeunStochastic(
            dt=2**-4,
            noise=noise.Additive(nsig=np.array([0.001]))
        )
        return heunint

    def setup_monitors(self):
        """Setup monitors to record simulation output"""
        # Raw neural activity (LFP-like, high temporal resolution)
        mon_raw = monitors.Raw()

        # Temporal average with TR period (downsampled neural activity)
        mon_tavg = monitors.TemporalAverage(period=self.TR * 1000.0)  # Convert to ms

        # BOLD signal with HRF (for fMRI comparison)
        mon_bold = monitors.Bold(period=self.TR * 1000.0)  # Sample at TR

        return [mon_raw, mon_tavg, mon_bold]

    def run_simulation(self):
        """Run the brain simulation"""
        print("\n" + "="*60)
        print(f"Starting {self.num_nodes}-node brain simulation")
        print(f"Duration: {self.simulation_length} ms")
        print(f"Model: {self.model_type}")
        print("="*60 + "\n")

        # Setup components
        connectivity = self.setup_connectivity()
        model = self.setup_model()
        coupling_func = self.setup_coupling()
        integrator = self.setup_integrator()
        monitors = self.setup_monitors()

        # Initialize simulator
        self.simulator = simulator.Simulator(
            model=model,
            connectivity=connectivity,
            coupling=coupling_func,
            integrator=integrator,
            monitors=monitors
        )

        self.simulator.configure()

        print("Running simulation...")
        print("This may take several minutes...\n")

        # Run simulation
        results = {}
        chunk_count = 0
        for data in self.simulator(simulation_length=self.simulation_length):
            chunk_count += 1
            if chunk_count % 50 == 0:
                print(f"  Progress: {chunk_count} chunks processed...")

            for i, monitor in enumerate(self.simulator.monitors):
                monitor_name = monitor.__class__.__name__
                if data[i] is not None:
                    if monitor_name not in results:
                        results[monitor_name] = []
                    results[monitor_name].append(data[i])

        # Concatenate results
        print(f"Concatenating {len(results)} result streams...")
        for key in results:
            results[key] = np.concatenate([x[1] for x in results[key] if x is not None])
            print(f"  {key}: {results[key].shape}")

        self.results = results
        print("Simulation completed!")
        return results

    def save_results(self, output_path):
        """
        Save simulation results to .npy files with shape (ROI, timepoint)
        Saves multiple signal types: neural activity (LFP-like), downsampled, and BOLD
        Also saves HRF parameters and metadata as .json file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        base_name = output_path.stem

        print(f"\nSaving results to {output_path.parent}...")

        # Helper function to process and save time series
        def process_and_save(data, suffix, expected_timepoints=None):
            """Process time series and save with given suffix"""
            # Extract first state variable and first mode
            if len(data.shape) == 4:
                ts_data = data[:, :, 0, 0]  # (time, nodes)
            elif len(data.shape) == 3:
                ts_data = data[:, :, 0]  # (time, nodes)
            else:
                ts_data = data

            # Truncate or pad if expected timepoints specified
            if expected_timepoints is not None:
                if ts_data.shape[0] > expected_timepoints:
                    ts_data = ts_data[:expected_timepoints, :]
                elif ts_data.shape[0] < expected_timepoints:
                    print(f"  Warning: {suffix} has {ts_data.shape[0]} timepoints, expected {expected_timepoints}")
                    padded = np.zeros((expected_timepoints, self.num_nodes))
                    padded[:ts_data.shape[0], :] = ts_data
                    ts_data = padded

            # Transpose to (ROI, timepoint) format
            ts_data = ts_data.T

            # Save
            save_path = output_path.parent / f'{base_name}_{suffix}.npy'
            np.save(save_path, ts_data)

            return ts_data, save_path

        # 1. RAW neural activity (LFP-like signals) - SKIPPED to save disk space
        # Note: Raw neural data is NOT saved to disk to conserve storage
        if 'Raw' in self.results:
            print(f"  Raw neural activity: [NOT SAVED - too large]")
            print(f"    Shape would be: (ROI, {self.results['Raw'].shape[0]}) - skipping to save disk space")

        # 2. Save DOWNSAMPLED neural activity (at TR)
        if 'TemporalAverage' in self.results:
            tavg_data = self.results['TemporalAverage']
            tavg_ts, tavg_path = process_and_save(tavg_data, 'neural', self.num_timepoints)
            print(f"  Downsampled neural: {tavg_path}")
            print(f"    Shape: {tavg_ts.shape} (ROI, timepoints)")
            print(f"    TR: {self.TR} seconds")

            # Analyze neural activity peaks (12 peaks per ROI)
            try:
                sampling_rate = 1.0 / self.TR  # Hz
                peak_results, json_path = analyze_neural_peaks(
                    tavg_ts,
                    output_path,
                    sampling_rate=sampling_rate,
                    num_peaks=12
                )
                print(f"    Neural peak analysis saved to: {json_path}")
                print(f"    Target: 12 peaks per ROI")
                print(f"    Actual: {peak_results['global_statistics']['mean_peaks_per_roi']:.1f} peaks per ROI")
            except Exception as e:
                print(f"    Warning: Neural peak analysis failed: {e}")

        # 3. Save BOLD signal with HRF
        if 'Bold' in self.results:
            bold_data = self.results['Bold']
            bold_ts, bold_path = process_and_save(bold_data, 'bold', self.num_timepoints)
            print(f"  BOLD (with HRF): {bold_path}")
            print(f"    Shape: {bold_ts.shape} (ROI, timepoints)")
            print(f"    TR: {self.TR} seconds")

            # Analyze BOLD peaks (12 peaks per ROI)
            try:
                sampling_rate = 1.0 / self.TR  # Hz
                peak_results = analyze_bold_peaks(
                    bold_ts,
                    output_path,
                    sampling_rate=sampling_rate,
                    num_peaks=12
                )
                print(f"    BOLD peak analysis complete!")
                print(f"    Target: 12 peaks per ROI")
                print(f"    Actual: {peak_results['global_statistics']['mean_peaks_per_roi']:.1f} peaks per ROI")
            except Exception as e:
                print(f"    Warning: BOLD peak analysis failed: {e}")

        # Save metadata including HRF parameters
        metadata_path = output_path.parent / f'{base_name}_metadata.json'

        # Get HRF parameters from BOLD monitor
        hrf_params = {}
        if hasattr(self.simulator, 'monitors'):
            for mon in self.simulator.monitors:
                if isinstance(mon, monitors.Bold):
                    # Safely extract available attributes
                    hrf_params = {}
                    attr_map = {
                        'tau_s': 'Signal decay time constant',
                        'tau_f': 'Feedback time constant',
                        'tau_o': 'Oxidative metabolism time constant',
                        'alpha': "Grubb's exponent",
                        'E0': 'Resting oxygen extraction fraction',
                        'V0': 'Resting blood volume fraction',
                        'TE': 'Echo time',
                        'v0': 'Frequency offset at rest',
                        'r0': 'Slope of intravascular relaxation rate',
                        'period': 'Sampling period in ms'
                    }
                    for attr_name in attr_map:
                        if hasattr(mon, attr_name):
                            hrf_params[attr_name] = float(getattr(mon, attr_name))
                    break

        metadata = {
            'num_nodes': int(self.num_nodes),
            'num_timepoints': int(self.num_timepoints),
            'TR': float(self.TR),
            'simulation_length_ms': float(self.simulation_length),
            'model_type': self.model_type,
            'timestamp': str(datetime.now()),
            'integration_dt_ms': float(2**-4),
            'sampling_frequency_hz': float(1000 / (2**-4)),
            'outputs': {
                'neural': f'{base_name}_neural.npy',
                'bold': f'{base_name}_bold.npy'
            },
            'data_format': '(ROI, timepoints)',
            'signal_types': {
                'neural': f'Downsampled neural activity at TR={self.TR}s',
                'bold': f'BOLD signal with HRF convolution at TR={self.TR}s'
            },
            'note': 'Raw neural activity NOT saved to conserve disk space',
            'hrf_parameters': hrf_params if hrf_params else 'Not available'
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save connectivity separately
        if self.simulator:
            conn_path = output_path.parent / f'connectivity_{self.num_nodes}nodes.npy'
            np.save(conn_path, self.simulator.connectivity.weights)
            print(f"  Connectivity: {conn_path}")

        print(f"\n  Metadata (with HRF params): {metadata_path}")
        print(f"\nAll results saved successfully!")
        return output_path

    def get_time_series(self):
        """Extract time series in (ROI, timepoint) format"""
        if self.results is None:
            raise ValueError("No simulation results available. Run simulation first.")

        time_series = self.results['TemporalAverage']

        if len(time_series.shape) == 4:
            ts_data = time_series[:, :, 0, 0].T  # Transpose to (ROI, time)
        elif len(time_series.shape) == 3:
            ts_data = time_series[:, :, 0].T
        else:
            ts_data = time_series.T

        return ts_data


def main():
    parser = argparse.ArgumentParser(description='Run whole-brain simulation with TVB')
    parser.add_argument('--nodes', type=int, default=68, choices=[48, 68, 360],
                       help='Number of brain regions: 48 (downsampled), 68 (Desikan-Killiany), or 360 (HCP-MMP1)')
    parser.add_argument('--timepoints', type=int, default=375,
                       help='Number of timepoints (default: 375)')
    parser.add_argument('--tr', type=float, default=0.8,
                       help='Repetition time (TR) in seconds (default: 0.8)')
    parser.add_argument('--model', type=str, default='wong_wang',
                       choices=['wong_wang', 'kuramoto', 'generic_2d_oscillator'],
                       help='Neural mass model type')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (.npy)')

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = f"results/sim_{args.nodes}nodes_{args.timepoints}tp_TR{args.tr}.npy"

    # Run simulation
    sim = BrainSimulation(
        num_nodes=args.nodes,
        num_timepoints=args.timepoints,
        TR=args.tr,
        model_type=args.model
    )

    sim.run_simulation()
    sim.save_results(args.output)

    print("\n" + "="*60)
    print("Simulation Summary:")
    print(f"  Nodes (ROI): {args.nodes}")
    print(f"  Timepoints: {args.timepoints}")
    print(f"  TR: {args.tr} seconds")
    print(f"  Total duration: {args.timepoints * args.tr} seconds")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()
