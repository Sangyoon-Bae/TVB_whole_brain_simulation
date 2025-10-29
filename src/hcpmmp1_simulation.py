"""
TVB Simulation with HCP-MMP1 Atlas (360 ROIs)
Whole-brain simulation using HCP-MMP1 cortical parcellation
"""

import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime

from tvb.simulator.lab import *
from tvb.datatypes import connectivity


class HCPMMP1Simulation:
    """
    Whole-brain simulation using HCP-MMP1 360 ROI atlas
    """

    def __init__(self, num_timepoints=375, TR=0.8, model_type='wong_wang',
                 connectivity_dir='data/HCPMMP1'):
        """
        Initialize HCP-MMP1 brain simulation

        Parameters:
        -----------
        num_timepoints : int
            Number of timepoints to generate (default: 375)
        TR : float
            Repetition time in seconds (default: 0.8)
        model_type : str
            Neural mass model ('wong_wang', 'kuramoto', 'generic_2d_oscillator')
        connectivity_dir : str
            Directory containing HCP-MMP1 connectivity files
        """
        self.num_nodes = 360  # HCP-MMP1 atlas
        self.num_timepoints = num_timepoints
        self.TR = TR
        self.simulation_length = num_timepoints * TR * 1000  # Convert to ms
        self.model_type = model_type
        self.connectivity_dir = Path(connectivity_dir)
        self.simulator = None
        self.results = None
        self.region_labels = None

    def load_connectivity(self):
        """Load HCP-MMP1 connectivity from saved files"""
        print(f"Loading HCP-MMP1 connectivity from {self.connectivity_dir}...")

        # Load connectivity matrices
        weights_file = self.connectivity_dir / 'mmp_weights_360.npy'
        tract_file = self.connectivity_dir / 'mmp_tract_lengths_360.npy'
        centers_file = self.connectivity_dir / 'mmp_centers_360.npy'
        metadata_file = self.connectivity_dir / 'mmp_connectivity_360_metadata.json'

        if not all([f.exists() for f in [weights_file, tract_file, centers_file]]):
            raise FileNotFoundError(
                f"Connectivity files not found in {self.connectivity_dir}.\n"
                f"Run: python src/hcpmmp1_loader.py --method distance"
            )

        # Load arrays
        weights = np.load(weights_file)
        tract_lengths = np.load(tract_file)
        centers = np.load(centers_file)

        # Load metadata if available
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.region_labels = np.array(metadata.get('regions', [f"ROI_{i}" for i in range(360)]))
        else:
            self.region_labels = np.array([f"MMP_ROI_{i}" for i in range(360)])

        print(f"  Loaded connectivity: {weights.shape}")
        print(f"  Number of regions: {len(self.region_labels)}")

        # Create TVB connectivity object
        conn = connectivity.Connectivity(
            weights=weights,
            tract_lengths=tract_lengths,
            region_labels=self.region_labels,
            centres=centers,
            speed=np.array([3.0])  # Conduction speed in m/s
        )

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
        heunint = integrators.HeunStochastic(
            dt=2**-4,
            noise=noise.Additive(nsig=np.array([0.001]))
        )
        return heunint

    def setup_monitors(self):
        """Setup monitors to record simulation output"""
        # Raw neural activity
        mon_raw = monitors.Raw()

        # Temporal average with TR period
        mon_tavg = monitors.TemporalAverage(period=self.TR * 1000.0)

        # BOLD signal
        mon_bold = monitors.Bold(period=self.TR * 1000.0)

        return [mon_raw, mon_tavg, mon_bold]

    def run_simulation(self):
        """Run the brain simulation"""
        print("\n" + "="*60)
        print(f"HCP-MMP1 Brain Simulation (360 cortical ROIs)")
        print(f"Duration: {self.simulation_length} ms")
        print(f"Model: {self.model_type}")
        print("="*60 + "\n")

        # Setup components
        connectivity = self.load_connectivity()
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
        print("This may take 10-30 minutes for 360 ROIs...\n")

        # Run simulation
        results = {}
        chunk_count = 0

        for data in self.simulator(simulation_length=self.simulation_length):
            chunk_count += 1
            if chunk_count % 100 == 0:
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
        """Save simulation results to .npy files with shape (ROI, timepoint)"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        base_name = output_path.stem

        print(f"\nSaving results to {output_path.parent}...")

        def process_and_save(data, suffix, expected_timepoints=None):
            """Process time series and save with given suffix"""
            # Extract first state variable
            if len(data.shape) == 4:
                ts_data = data[:, :, 0, 0]  # (time, nodes)
            elif len(data.shape) == 3:
                ts_data = data[:, :, 0]
            else:
                ts_data = data

            # Truncate or pad
            if expected_timepoints is not None:
                if ts_data.shape[0] > expected_timepoints:
                    ts_data = ts_data[:expected_timepoints, :]
                elif ts_data.shape[0] < expected_timepoints:
                    print(f"  Warning: {suffix} has {ts_data.shape[0]} timepoints, expected {expected_timepoints}")
                    padded = np.zeros((expected_timepoints, self.num_nodes))
                    padded[:ts_data.shape[0], :] = ts_data
                    ts_data = padded

            # Transpose to (ROI, timepoint)
            ts_data = ts_data.T

            # Save
            save_path = output_path.parent / f'{base_name}_{suffix}.npy'
            np.save(save_path, ts_data)

            return ts_data, save_path

        # Save neural activity
        if 'TemporalAverage' in self.results:
            tavg_data = self.results['TemporalAverage']
            tavg_ts, tavg_path = process_and_save(tavg_data, 'neural', self.num_timepoints)
            print(f"  Neural activity: {tavg_path}")
            print(f"    Shape: {tavg_ts.shape} (ROI, timepoints)")

        # Save BOLD signal
        if 'Bold' in self.results:
            bold_data = self.results['Bold']
            bold_ts, bold_path = process_and_save(bold_data, 'bold', self.num_timepoints)
            print(f"  BOLD signal: {bold_path}")
            print(f"    Shape: {bold_ts.shape} (ROI, timepoints)")

        # Save metadata
        metadata_path = output_path.parent / f'{base_name}_metadata.json'

        metadata = {
            'atlas': 'HCP-MMP1-360',
            'description': 'Human Connectome Project Multi-Modal Parcellation 1.0',
            'num_nodes': int(self.num_nodes),
            'num_timepoints': int(self.num_timepoints),
            'TR': float(self.TR),
            'simulation_length_ms': float(self.simulation_length),
            'model_type': self.model_type,
            'timestamp': str(datetime.now()),
            'integration_dt_ms': float(2**-4),
            'region_labels': self.region_labels.tolist() if self.region_labels is not None else [],
            'hemispheres': {
                'left': 'ROIs 0-179',
                'right': 'ROIs 180-359'
            },
            'outputs': {
                'neural': f'{base_name}_neural.npy',
                'bold': f'{base_name}_bold.npy'
            },
            'data_format': '(ROI, timepoints)',
            'connectivity_source': str(self.connectivity_dir)
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save connectivity
        if self.simulator:
            conn_path = output_path.parent / 'mmp_connectivity_360nodes.npy'
            np.save(conn_path, self.simulator.connectivity.weights)
            print(f"  Connectivity: {conn_path}")

        print(f"  Metadata: {metadata_path}")
        print(f"\nAll results saved successfully!")

        return output_path


def main():
    """Run HCP-MMP1 simulation"""
    parser = argparse.ArgumentParser(
        description='Run TVB simulation with HCP-MMP1 360 cortical ROIs'
    )
    parser.add_argument('--timepoints', type=int, default=375,
                       help='Number of timepoints (default: 375)')
    parser.add_argument('--tr', type=float, default=0.8,
                       help='Repetition time in seconds (default: 0.8)')
    parser.add_argument('--model', type=str, default='wong_wang',
                       choices=['wong_wang', 'kuramoto', 'generic_2d_oscillator'],
                       help='Neural mass model type')
    parser.add_argument('--connectivity-dir', type=str, default='data/HCPMMP1',
                       help='Directory containing HCP-MMP1 connectivity files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (.npy)')

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = f"results/mmp_sim_360nodes_{args.timepoints}tp_TR{args.tr}.npy"

    # Check if connectivity exists
    conn_dir = Path(args.connectivity_dir)
    if not (conn_dir / 'mmp_weights_360.npy').exists():
        print("="*60)
        print("HCP-MMP1 connectivity not found!")
        print("Generating connectivity matrices first...")
        print("="*60 + "\n")

        # Generate connectivity
        import hcpmmp1_loader
        atlas = hcpmmp1_loader.HCPMMP1Atlas(data_dir=args.connectivity_dir)
        atlas.load_atlas()
        atlas.save_connectivity(output_dir=args.connectivity_dir, method='distance')
        print("\n" + "="*60 + "\n")

    # Run simulation
    sim = HCPMMP1Simulation(
        num_timepoints=args.timepoints,
        TR=args.tr,
        model_type=args.model,
        connectivity_dir=args.connectivity_dir
    )

    sim.run_simulation()
    sim.save_results(args.output)

    print("\n" + "="*60)
    print("Simulation Summary:")
    print(f"  Atlas: HCP-MMP1 (360 ROIs)")
    print(f"  Timepoints: {args.timepoints}")
    print(f"  TR: {args.tr} seconds")
    print(f"  Total duration: {args.timepoints * args.tr} seconds")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()
