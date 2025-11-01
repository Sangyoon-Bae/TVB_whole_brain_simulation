"""
TVB Simulation with HarvardOxford Atlas (48 ROIs)
Whole-brain simulation using HarvardOxford cortical parcellation
WITH PARALLEL COMPUTING SUPPORT (32 CPUs)
"""

import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
import time

from tvb.simulator.lab import *
from tvb.datatypes import connectivity


class HarvardOxfordSimulation:
    """
    Whole-brain simulation using HarvardOxford 48 ROI atlas
    """

    def __init__(self, num_timepoints=375, TR=0.8, model_type='wong_wang',
                 connectivity_dir='data/HarvardOxford', n_workers=32):
        """
        Initialize HarvardOxford brain simulation

        Parameters:
        -----------
        num_timepoints : int
            Number of timepoints to generate (default: 375)
        TR : float
            Repetition time in seconds (default: 0.8)
        model_type : str
            Neural mass model ('wong_wang', 'kuramoto', 'generic_2d_oscillator')
        connectivity_dir : str
            Directory containing HarvardOxford connectivity files
        n_workers : int
            Number of parallel workers for ensemble simulations (default: 32)
        """
        self.num_nodes = 48  # HarvardOxford cortical atlas
        self.num_timepoints = num_timepoints
        self.TR = TR
        self.simulation_length = num_timepoints * TR * 1000  # Convert to ms
        self.model_type = model_type
        self.connectivity_dir = Path(connectivity_dir)
        self.n_workers = min(n_workers, cpu_count())  # Don't exceed available CPUs
        self.simulator = None
        self.results = None
        self.region_labels = None

    def load_connectivity(self):
        """Load HarvardOxford connectivity from saved files"""
        print(f"Loading HarvardOxford connectivity from {self.connectivity_dir}...")

        # Load connectivity matrices
        weights_file = self.connectivity_dir / 'ho_weights_48.npy'
        tract_file = self.connectivity_dir / 'ho_tract_lengths_48.npy'
        centers_file = self.connectivity_dir / 'ho_centers_48.npy'
        metadata_file = self.connectivity_dir / 'ho_connectivity_48_metadata.json'

        if not all([f.exists() for f in [weights_file, tract_file, centers_file]]):
            raise FileNotFoundError(
                f"Connectivity files not found in {self.connectivity_dir}.\n"
                f"Run: python src/harvard_oxford_loader.py --method distance"
            )

        # Load arrays
        weights = np.load(weights_file)
        tract_lengths = np.load(tract_file)
        centers = np.load(centers_file)

        # Load metadata if available
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.region_labels = np.array(metadata.get('regions', [f"ROI_{i}" for i in range(48)]))
        else:
            self.region_labels = np.array([f"HO_ROI_{i}" for i in range(48)])

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
        print(f"HarvardOxford Brain Simulation (48 cortical ROIs)")
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
            'atlas': 'HarvardOxford-cortical-48',
            'num_nodes': int(self.num_nodes),
            'num_timepoints': int(self.num_timepoints),
            'TR': float(self.TR),
            'simulation_length_ms': float(self.simulation_length),
            'model_type': self.model_type,
            'timestamp': str(datetime.now()),
            'integration_dt_ms': float(2**-4),
            'region_labels': self.region_labels.tolist() if self.region_labels is not None else [],
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
            conn_path = output_path.parent / 'ho_connectivity_48nodes.npy'
            np.save(conn_path, self.simulator.connectivity.weights)
            print(f"  Connectivity: {conn_path}")

        print(f"  Metadata: {metadata_path}")
        print(f"\nAll results saved successfully!")

        return output_path

    def run_ensemble_parallel(self, n_realizations=32, average=True, save_all=False):
        """
        Run ensemble of simulations in parallel with different noise realizations

        This method uses parallel computing (32 CPUs by default) to run multiple
        simulations with different random seeds simultaneously.

        Parameters:
        -----------
        n_realizations : int
            Number of simulations to run with different noise (default: 32)
        average : bool
            If True, return averaged results across realizations (default: True)
        save_all : bool
            If True, save all individual realizations (default: False)

        Returns:
        --------
        results : dict
            Ensemble simulation results (averaged or all realizations)
        """
        print("\n" + "="*70)
        print(f"PARALLEL ENSEMBLE SIMULATION (HarvardOxford 48 ROIs)")
        print("="*70)
        print(f"Number of realizations: {n_realizations}")
        print(f"Parallel workers: {self.n_workers}")
        print(f"Total CPU cores available: {cpu_count()}")
        print(f"Average results: {average}")
        print("="*70)
        print()

        # Create arguments for each realization
        args_list = [
            (i, self.num_timepoints, self.TR, self.simulation_length,
             self.model_type, str(self.connectivity_dir), self.num_nodes)
            for i in range(n_realizations)
        ]

        # Run simulations in parallel
        start_time = time.time()
        print(f"Starting {n_realizations} simulations in parallel with {self.n_workers} workers...")

        with Pool(processes=self.n_workers) as pool:
            all_results = pool.map(_run_single_realization_ho, args_list)

        elapsed_time = time.time() - start_time

        print(f"\n✓ All {n_realizations} simulations completed!")
        print(f"  Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        print(f"  Average time per simulation: {elapsed_time/n_realizations:.1f}s")
        print(f"  Speedup vs sequential: ~{n_realizations/(elapsed_time/(elapsed_time/n_realizations)):.1f}x")

        # Process results
        if average:
            print("\nAveraging results across realizations...")
            averaged_results = {}

            for key in all_results[0].keys():
                # Stack all realizations
                stacked = np.array([r[key] for r in all_results])
                # Average across realizations (axis 0)
                averaged_results[key] = np.mean(stacked, axis=0)
                print(f"  {key}: {averaged_results[key].shape}")

            self.results = averaged_results
            return averaged_results
        else:
            self.results = all_results
            return all_results


def _run_single_realization_ho(args):
    """
    Worker function to run a single simulation realization

    This function is defined at module level for multiprocessing compatibility

    Parameters:
    -----------
    args : tuple
        (seed, num_timepoints, TR, simulation_length, model_type, connectivity_dir, num_nodes)

    Returns:
    --------
    results : dict
        Simulation results for this realization
    """
    seed, num_timepoints, TR, simulation_length, model_type, connectivity_dir, num_nodes = args

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Load connectivity
    conn_dir = Path(connectivity_dir)
    weights = np.load(conn_dir / 'ho_weights_48.npy')
    tract_lengths = np.load(conn_dir / 'ho_tract_lengths_48.npy')
    centers = np.load(conn_dir / 'ho_centers_48.npy')

    metadata_file = conn_dir / 'ho_connectivity_48_metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            region_labels = np.array(metadata.get('regions', [f"ROI_{i}" for i in range(48)]))
    else:
        region_labels = np.array([f"HO_ROI_{i}" for i in range(48)])

    conn = connectivity.Connectivity(
        weights=weights,
        tract_lengths=tract_lengths,
        region_labels=region_labels,
        centres=centers,
        speed=np.array([3.0])
    )
    conn.configure()

    # Setup model
    if model_type == 'wong_wang':
        model = models.ReducedWongWang()
    elif model_type == 'kuramoto':
        model = models.Kuramoto()
    elif model_type == 'generic_2d_oscillator':
        model = models.Generic2dOscillator()

    # Setup coupling
    coupling_func = coupling.Linear(a=np.array([0.0152]))

    # Setup integrator with seed-specific noise
    heunint = integrators.HeunStochastic(
        dt=2**-4,
        noise=noise.Additive(nsig=np.array([0.001]))
    )

    # Setup monitors
    mon_raw = monitors.Raw()
    mon_tavg = monitors.TemporalAverage(period=TR * 1000.0)
    mon_bold = monitors.Bold(period=TR * 1000.0)

    # Initialize simulator
    sim = simulator.Simulator(
        model=model,
        connectivity=conn,
        coupling=coupling_func,
        integrator=heunint,
        monitors=[mon_raw, mon_tavg, mon_bold]
    )
    sim.configure()

    # Run simulation (suppress output)
    results = {}
    for data in sim(simulation_length=simulation_length):
        for i, monitor in enumerate(sim.monitors):
            monitor_name = monitor.__class__.__name__
            if data[i] is not None:
                if monitor_name not in results:
                    results[monitor_name] = []
                results[monitor_name].append(data[i])

    # Concatenate and extract data
    final_results = {}
    for key in results:
        concatenated = np.concatenate([x[1] for x in results[key] if x is not None])

        # Extract first state variable
        if len(concatenated.shape) == 4:
            ts_data = concatenated[:, :, 0, 0]
        elif len(concatenated.shape) == 3:
            ts_data = concatenated[:, :, 0]
        else:
            ts_data = concatenated

        # Truncate/pad to expected timepoints
        if ts_data.shape[0] > num_timepoints:
            ts_data = ts_data[:num_timepoints, :]
        elif ts_data.shape[0] < num_timepoints:
            padded = np.zeros((num_timepoints, num_nodes))
            padded[:ts_data.shape[0], :] = ts_data
            ts_data = padded

        # Transpose to (ROI, timepoints)
        final_results[key] = ts_data.T

    return final_results


def main():
    """Run HarvardOxford simulation"""
    parser = argparse.ArgumentParser(
        description='Run TVB simulation with HarvardOxford 48 cortical ROIs (with parallel computing)'
    )
    parser.add_argument('--timepoints', type=int, default=375,
                       help='Number of timepoints (default: 375)')
    parser.add_argument('--tr', type=float, default=0.8,
                       help='Repetition time in seconds (default: 0.8)')
    parser.add_argument('--model', type=str, default='wong_wang',
                       choices=['wong_wang', 'kuramoto', 'generic_2d_oscillator'],
                       help='Neural mass model type')
    parser.add_argument('--connectivity-dir', type=str, default='data/HarvardOxford',
                       help='Directory containing HarvardOxford connectivity files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (.npy)')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel computing (32 CPUs)')
    parser.add_argument('--n-realizations', type=int, default=32,
                       help='Number of ensemble realizations for parallel mode (default: 32)')
    parser.add_argument('--n-workers', type=int, default=32,
                       help='Number of parallel workers (default: 32)')

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = f"results/ho_sim_48nodes_{args.timepoints}tp_TR{args.tr}.npy"

    # Check if connectivity exists
    conn_dir = Path(args.connectivity_dir)
    if not (conn_dir / 'ho_weights_48.npy').exists():
        print("="*60)
        print("HarvardOxford connectivity not found!")
        print("Generating connectivity matrices first...")
        print("="*60 + "\n")

        # Generate connectivity
        import harvard_oxford_loader
        atlas = harvard_oxford_loader.HarvardOxfordAtlas(data_dir=args.connectivity_dir)
        atlas.load_atlas(threshold=25)
        atlas.save_connectivity(output_dir=args.connectivity_dir, method='distance')
        print("\n" + "="*60 + "\n")

    # Run simulation
    sim = HarvardOxfordSimulation(
        num_timepoints=args.timepoints,
        TR=args.tr,
        model_type=args.model,
        connectivity_dir=args.connectivity_dir,
        n_workers=args.n_workers
    )

    if args.parallel:
        # Run parallel ensemble simulation
        print(f"\n🚀 PARALLEL MODE: Using {args.n_workers} CPUs")
        sim.run_ensemble_parallel(n_realizations=args.n_realizations, average=True)
    else:
        # Run single simulation (original mode)
        print("\n📌 SEQUENTIAL MODE: Using single CPU")
        sim.run_simulation()

    sim.save_results(args.output)

    print("\n" + "="*60)
    print("Simulation Summary:")
    print(f"  Atlas: HarvardOxford cortical (48 ROIs)")
    print(f"  Mode: {'PARALLEL (ensemble)' if args.parallel else 'SEQUENTIAL (single)'}")
    if args.parallel:
        print(f"  Realizations: {args.n_realizations}")
        print(f"  Workers (CPUs): {args.n_workers}")
    print(f"  Timepoints: {args.timepoints}")
    print(f"  TR: {args.tr} seconds")
    print(f"  Total duration: {args.timepoints * args.tr} seconds")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()
