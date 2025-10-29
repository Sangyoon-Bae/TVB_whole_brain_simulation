"""
HarvardOxford Atlas Loader for TVB Simulations
Loads HarvardOxford cortical atlas and prepares connectivity for TVB
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import json


class HarvardOxfordAtlas:
    """
    Loads and processes HarvardOxford cortical atlas
    The cortical atlas contains 48 regions (bilateral)
    """

    # HarvardOxford cortical atlas labels (48 regions)
    CORTICAL_LABELS = [
        # Left hemisphere (1-24)
        "L_Frontal_Pole", "L_Insular_Cortex", "L_Superior_Frontal_Gyrus",
        "L_Middle_Frontal_Gyrus", "L_Inferior_Frontal_Gyrus_pars_triangularis",
        "L_Inferior_Frontal_Gyrus_pars_opercularis", "L_Precentral_Gyrus",
        "L_Temporal_Pole", "L_Superior_Temporal_Gyrus_anterior",
        "L_Superior_Temporal_Gyrus_posterior", "L_Middle_Temporal_Gyrus_anterior",
        "L_Middle_Temporal_Gyrus_posterior", "L_Middle_Temporal_Gyrus_temporooccipital",
        "L_Inferior_Temporal_Gyrus_anterior", "L_Inferior_Temporal_Gyrus_posterior",
        "L_Inferior_Temporal_Gyrus_temporooccipital", "L_Postcentral_Gyrus",
        "L_Superior_Parietal_Lobule", "L_Supramarginal_Gyrus_anterior",
        "L_Supramarginal_Gyrus_posterior", "L_Angular_Gyrus",
        "L_Lateral_Occipital_Cortex_superior", "L_Lateral_Occipital_Cortex_inferior",
        "L_Intracalcarine_Cortex",

        # Right hemisphere (25-48)
        "R_Frontal_Pole", "R_Insular_Cortex", "R_Superior_Frontal_Gyrus",
        "R_Middle_Frontal_Gyrus", "R_Inferior_Frontal_Gyrus_pars_triangularis",
        "R_Inferior_Frontal_Gyrus_pars_opercularis", "R_Precentral_Gyrus",
        "R_Temporal_Pole", "R_Superior_Temporal_Gyrus_anterior",
        "R_Superior_Temporal_Gyrus_posterior", "R_Middle_Temporal_Gyrus_anterior",
        "R_Middle_Temporal_Gyrus_posterior", "R_Middle_Temporal_Gyrus_temporooccipital",
        "R_Inferior_Temporal_Gyrus_anterior", "R_Inferior_Temporal_Gyrus_posterior",
        "R_Inferior_Temporal_Gyrus_temporooccipital", "R_Postcentral_Gyrus",
        "R_Superior_Parietal_Lobule", "R_Supramarginal_Gyrus_anterior",
        "R_Supramarginal_Gyrus_posterior", "R_Angular_Gyrus",
        "R_Lateral_Occipital_Cortex_superior", "R_Lateral_Occipital_Cortex_inferior",
        "R_Intracalcarine_Cortex"
    ]

    def __init__(self, data_dir='data/HarvardOxford'):
        """Initialize atlas loader with data directory"""
        self.data_dir = Path(data_dir)
        self.atlas_img = None
        self.atlas_data = None
        self.num_regions = 48

    def load_atlas(self, threshold=25):
        """
        Load HarvardOxford cortical atlas

        Parameters:
        -----------
        threshold : int
            Probability threshold (0, 25, or 50)
        """
        atlas_file = self.data_dir / f'HarvardOxford-cort-maxprob-thr{threshold}-2mm.nii.gz'

        if not atlas_file.exists():
            raise FileNotFoundError(f"Atlas file not found: {atlas_file}")

        print(f"Loading HarvardOxford cortical atlas (threshold={threshold}%)...")
        self.atlas_img = nib.load(str(atlas_file))
        self.atlas_data = self.atlas_img.get_fdata()

        # Get unique labels
        unique_labels = sorted(set(self.atlas_data.flatten().astype(int)))
        print(f"  Atlas shape: {self.atlas_data.shape}")
        print(f"  Unique regions: {len(unique_labels)} (including background)")
        print(f"  Region labels: {unique_labels}")

        return self.atlas_data

    def extract_roi_centers(self):
        """
        Extract center of mass for each ROI
        Returns array of shape (48, 3) with MNI coordinates
        """
        if self.atlas_data is None:
            self.load_atlas()

        centers = []
        affine = self.atlas_img.affine

        for roi_idx in range(1, self.num_regions + 1):
            # Find voxels belonging to this ROI
            roi_mask = (self.atlas_data == roi_idx)

            if roi_mask.sum() == 0:
                # No voxels for this ROI, use default position
                centers.append([0, 0, 0])
                continue

            # Calculate center of mass in voxel coordinates
            voxel_coords = np.array(np.where(roi_mask))
            center_voxel = voxel_coords.mean(axis=1)

            # Convert to MNI coordinates using affine transform
            center_mni = nib.affines.apply_affine(affine, center_voxel)
            centers.append(center_mni)

        centers = np.array(centers)
        print(f"  Extracted {len(centers)} ROI centers")
        return centers

    def create_connectivity_matrix(self, method='distance'):
        """
        Create connectivity matrix for 48 ROIs

        Parameters:
        -----------
        method : str
            'distance' - based on Euclidean distance between ROIs
            'uniform' - uniform random connectivity
            'default' - use TVB default and downsample

        Returns:
        --------
        weights : ndarray (48, 48)
            Connectivity weights matrix
        tract_lengths : ndarray (48, 48)
            Tract length matrix
        centers : ndarray (48, 3)
            ROI centers in MNI space
        """
        centers = self.extract_roi_centers()

        if method == 'distance':
            # Distance-based connectivity
            print("Creating distance-based connectivity matrix...")

            # Calculate pairwise distances
            n_roi = len(centers)
            distances = np.zeros((n_roi, n_roi))

            for i in range(n_roi):
                for j in range(n_roi):
                    distances[i, j] = np.linalg.norm(centers[i] - centers[j])

            # Tract lengths = distances
            tract_lengths = distances

            # Weights: inverse distance (closer regions = stronger connection)
            # Avoid division by zero on diagonal
            weights = np.zeros_like(distances)
            mask = distances > 0
            weights[mask] = 1.0 / distances[mask]

            # Normalize weights
            weights = weights / weights.max() * 0.05

            # Zero out diagonal
            np.fill_diagonal(weights, 0)

        elif method == 'uniform':
            # Uniform random connectivity
            print("Creating uniform random connectivity matrix...")
            n_roi = self.num_regions
            weights = np.random.uniform(0.001, 0.05, (n_roi, n_roi))
            weights = (weights + weights.T) / 2  # Make symmetric
            np.fill_diagonal(weights, 0)

            # Random tract lengths
            tract_lengths = np.random.uniform(10, 100, (n_roi, n_roi))
            tract_lengths = (tract_lengths + tract_lengths.T) / 2
            np.fill_diagonal(tract_lengths, 0)

        elif method == 'default':
            # Use TVB default and downsample to 48 nodes
            print("Using downsampled TVB default connectivity...")
            from tvb.datatypes import connectivity

            conn = connectivity.Connectivity.from_file()
            n_orig = len(conn.weights)

            # Create indices for 48 regions
            indices = np.linspace(0, n_orig - 1, 48, dtype=int)

            # Downsample
            weights = conn.weights[indices][:, indices]
            tract_lengths = conn.tract_lengths[indices][:, indices]
            centers = conn.centres[indices]

        print(f"  Connectivity matrix shape: {weights.shape}")
        print(f"  Tract lengths shape: {tract_lengths.shape}")
        print(f"  Weight range: [{weights[weights>0].min():.6f}, {weights.max():.6f}]")
        print(f"  Tract length range: [{tract_lengths[tract_lengths>0].min():.2f}, {tract_lengths.max():.2f}] mm")

        return weights, tract_lengths, centers

    def save_connectivity(self, output_dir='data/HarvardOxford', method='distance'):
        """
        Save connectivity matrices and metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        weights, tract_lengths, centers = self.create_connectivity_matrix(method=method)

        # Save arrays
        np.save(output_dir / 'ho_weights_48.npy', weights)
        np.save(output_dir / 'ho_tract_lengths_48.npy', tract_lengths)
        np.save(output_dir / 'ho_centers_48.npy', centers)

        # Save metadata
        metadata = {
            'num_regions': self.num_regions,
            'atlas': 'HarvardOxford-cortical',
            'regions': self.CORTICAL_LABELS,
            'connectivity_method': method,
            'weights_file': 'ho_weights_48.npy',
            'tract_lengths_file': 'ho_tract_lengths_48.npy',
            'centers_file': 'ho_centers_48.npy',
            'shape': list(weights.shape),
            'notes': 'Connectivity for TVB simulations using HarvardOxford 48 cortical ROIs'
        }

        with open(output_dir / 'ho_connectivity_48_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nSaved connectivity files to {output_dir}:")
        print(f"  - ho_weights_48.npy")
        print(f"  - ho_tract_lengths_48.npy")
        print(f"  - ho_centers_48.npy")
        print(f"  - ho_connectivity_48_metadata.json")

        return weights, tract_lengths, centers


def main():
    """Generate HarvardOxford connectivity matrices"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate HarvardOxford connectivity for TVB')
    parser.add_argument('--threshold', type=int, default=25, choices=[0, 25, 50],
                       help='Probability threshold for atlas (default: 25)')
    parser.add_argument('--method', type=str, default='distance',
                       choices=['distance', 'uniform', 'default'],
                       help='Connectivity generation method')
    parser.add_argument('--output', type=str, default='data/HarvardOxford',
                       help='Output directory')

    args = parser.parse_args()

    # Create atlas loader
    atlas = HarvardOxfordAtlas()

    # Load atlas
    atlas.load_atlas(threshold=args.threshold)

    # Generate and save connectivity
    atlas.save_connectivity(output_dir=args.output, method=args.method)

    print("\nHarvardOxford connectivity generation complete!")
    print("Use this connectivity in TVB simulations with:")
    print("  python src/harvard_oxford_simulation.py --nodes 48")


if __name__ == '__main__':
    main()
