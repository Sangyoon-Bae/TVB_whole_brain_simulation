"""
Analysis utilities for brain simulation results
"""

import numpy as np
import json
from pathlib import Path
from scipy import signal, stats


class SimulationAnalyzer:
    """Analyze TVB simulation results"""

    def __init__(self, results_path):
        """Load simulation results"""
        self.results_path = Path(results_path)
        self.data = None
        self.metadata = None
        self.load_results()

    def load_results(self):
        """Load simulation results from .npy file"""
        # Load time series data (ROI, timepoints)
        self.data = np.load(self.results_path)

        # Load metadata from .json file
        metadata_path = self.results_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'num_nodes': self.data.shape[0],
                'num_timepoints': self.data.shape[1],
                'TR': 'unknown',
                'model_type': 'unknown'
            }

    def compute_functional_connectivity(self):
        """Compute functional connectivity matrix"""
        # Data shape is (ROI, timepoints)
        # Compute correlation between ROIs across time
        fc = np.corrcoef(self.data)
        return fc

    def compute_global_synchrony(self):
        """Compute global synchrony (Kuramoto order parameter)"""
        # Data shape is (ROI, timepoints)
        # Transpose to (timepoints, ROI) for Hilbert transform
        ts_data = self.data.T

        # Compute instantaneous phase using Hilbert transform
        analytic_signal = signal.hilbert(ts_data, axis=0)
        phases = np.angle(analytic_signal)

        # Compute Kuramoto order parameter
        R = np.abs(np.mean(np.exp(1j * phases), axis=1))

        return R

    def compute_metastability(self):
        """Compute metastability (std of global synchrony)"""
        R = self.compute_global_synchrony()
        return np.std(R)

    def summary_statistics(self):
        """Compute summary statistics"""
        fc = self.compute_functional_connectivity()
        fc_triu = fc[np.triu_indices_from(fc, k=1)]

        stats_dict = {
            'mean_fc': float(np.mean(fc_triu)),
            'std_fc': float(np.std(fc_triu)),
            'metastability': float(self.compute_metastability()),
            'mean_signal': float(np.mean(self.data)),
            'std_signal': float(np.std(self.data))
        }

        return stats_dict
