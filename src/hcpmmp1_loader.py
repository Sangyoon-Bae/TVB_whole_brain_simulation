"""
HCP-MMP1 Atlas Loader for TVB Simulations
Loads HCP-MMP1 (Human Connectome Project Multi-Modal Parcellation 1.0) atlas
360 cortical regions parcellation
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import json


class HCPMMP1Atlas:
    """
    Loads and processes HCP-MMP1 atlas
    The HCP-MMP1 atlas contains 360 cortical regions (180 per hemisphere)
    """

    def __init__(self, data_dir='data/HCPMMP1'):
        """Initialize atlas loader with data directory"""
        self.data_dir = Path(data_dir)
        self.atlas_img = None
        self.atlas_data = None
        self.num_regions = 360

    def load_atlas(self):
        """
        Load HCP-MMP1 atlas from NIfTI file
        """
        atlas_file = self.data_dir / 'MMP_in_MNI_corr.nii.gz'

        if not atlas_file.exists():
            raise FileNotFoundError(f"Atlas file not found: {atlas_file}")

        print(f"Loading HCP-MMP1 atlas (360 cortical regions)...")
        self.atlas_img = nib.load(str(atlas_file))
        self.atlas_data = self.atlas_img.get_fdata()

        # Get unique labels
        unique_labels = sorted(set(self.atlas_data.flatten()))
        unique_labels = [int(x) for x in unique_labels if x > 0]  # Exclude background (0)

        print(f"  Atlas shape: {self.atlas_data.shape}")
        print(f"  Unique regions: {len(unique_labels)} (excluding background)")
        print(f"  Label range: {min(unique_labels)} - {max(unique_labels)}")

        return self.atlas_data

    def extract_roi_centers(self):
        """
        Extract center of mass for each ROI
        Returns array of shape (360, 3) with MNI coordinates
        """
        if self.atlas_data is None:
            self.load_atlas()

        centers = []
        affine = self.atlas_img.affine

        # Get all unique ROI labels (excluding 0 = background)
        unique_labels = sorted(set(self.atlas_data.flatten()))
        roi_labels = [int(x) for x in unique_labels if x > 0]

        print(f"Extracting ROI centers for {len(roi_labels)} regions...")

        for roi_label in roi_labels:
            # Find voxels belonging to this ROI
            roi_mask = (self.atlas_data == roi_label)

            if roi_mask.sum() == 0:
                # No voxels for this ROI, use default position
                print(f"  Warning: ROI {roi_label} has no voxels")
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

        # Ensure we have exactly 360 centers
        if len(centers) != 360:
            print(f"  Warning: Expected 360 ROIs but found {len(centers)}")
            if len(centers) < 360:
                # Pad with zeros if needed
                padding = np.zeros((360 - len(centers), 3))
                centers = np.vstack([centers, padding])
            else:
                # Truncate if too many
                centers = centers[:360]

        return centers

    def generate_region_labels(self):
        """
        Generate region labels for HCP-MMP1 360 ROIs
        Format: L_ROI_001, L_ROI_002, ..., R_ROI_001, R_ROI_002, ...
        """
        labels = []

        # Left hemisphere (1-180)
        for i in range(1, 181):
            labels.append(f"L_MMP_{i:03d}")

        # Right hemisphere (181-360)
        for i in range(181, 361):
            labels.append(f"R_MMP_{i:03d}")

        return labels

    def create_connectivity_matrix(self, method='distance'):
        """
        Create connectivity matrix for 360 ROIs

        Parameters:
        -----------
        method : str
            'distance' - based on Euclidean distance between ROIs
            'uniform' - uniform random connectivity
            'structured' - structured connectivity with hemispheric bias

        Returns:
        --------
        weights : ndarray (360, 360)
            Connectivity weights matrix
        tract_lengths : ndarray (360, 360)
            Tract length matrix
        centers : ndarray (360, 3)
            ROI centers in MNI space
        """
        centers = self.extract_roi_centers()

        if method == 'distance':
            # Distance-based connectivity
            print("Creating distance-based connectivity matrix for 360 ROIs...")

            # Calculate pairwise distances
            n_roi = len(centers)
            distances = np.zeros((n_roi, n_roi))

            for i in range(n_roi):
                for j in range(n_roi):
                    distances[i, j] = np.linalg.norm(centers[i] - centers[j])

            # Tract lengths = distances
            tract_lengths = distances

            # Weights: inverse distance (closer regions = stronger connection)
            weights = np.zeros_like(distances)
            mask = distances > 0
            weights[mask] = 1.0 / distances[mask]

            # Normalize weights
            weights = weights / weights.max() * 0.05

            # Zero out diagonal
            np.fill_diagonal(weights, 0)

        elif method == 'uniform':
            # Uniform random connectivity
            print("Creating uniform random connectivity matrix for 360 ROIs...")
            n_roi = self.num_regions
            weights = np.random.uniform(0.001, 0.05, (n_roi, n_roi))
            weights = (weights + weights.T) / 2  # Make symmetric
            np.fill_diagonal(weights, 0)

            # Random tract lengths
            tract_lengths = np.random.uniform(10, 150, (n_roi, n_roi))
            tract_lengths = (tract_lengths + tract_lengths.T) / 2
            np.fill_diagonal(tract_lengths, 0)

        elif method == 'structured':
            # Structured connectivity with hemispheric organization
            print("Creating structured connectivity matrix for 360 ROIs...")

            # Calculate distances
            n_roi = len(centers)
            distances = np.zeros((n_roi, n_roi))
            for i in range(n_roi):
                for j in range(n_roi):
                    distances[i, j] = np.linalg.norm(centers[i] - centers[j])

            tract_lengths = distances

            # Base weights on inverse distance
            weights = np.zeros_like(distances)
            mask = distances > 0
            weights[mask] = 1.0 / distances[mask]

            # Strengthen intra-hemispheric connections
            # Left hemisphere: 0-179, Right hemisphere: 180-359
            left_idx = np.arange(180)
            right_idx = np.arange(180, 360)

            # Boost intra-hemispheric weights by 50%
            weights[np.ix_(left_idx, left_idx)] *= 1.5
            weights[np.ix_(right_idx, right_idx)] *= 1.5

            # Normalize
            weights = weights / weights.max() * 0.05
            np.fill_diagonal(weights, 0)

        print(f"  Connectivity matrix shape: {weights.shape}")
        print(f"  Tract lengths shape: {tract_lengths.shape}")
        print(f"  Weight range: [{weights[weights>0].min():.6f}, {weights.max():.6f}]")
        print(f"  Tract length range: [{tract_lengths[tract_lengths>0].min():.2f}, {tract_lengths.max():.2f}] mm")

        return weights, tract_lengths, centers

    def save_connectivity(self, output_dir='data/HCPMMP1', method='distance'):
        """
        Save connectivity matrices and metadata for 360 ROIs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        weights, tract_lengths, centers = self.create_connectivity_matrix(method=method)
        region_labels = self.generate_region_labels()

        # Save arrays
        np.save(output_dir / 'mmp_weights_360.npy', weights)
        np.save(output_dir / 'mmp_tract_lengths_360.npy', tract_lengths)
        np.save(output_dir / 'mmp_centers_360.npy', centers)

        # Save metadata
        metadata = {
            'num_regions': self.num_regions,
            'atlas': 'HCP-MMP1',
            'description': 'Human Connectome Project Multi-Modal Parcellation 1.0',
            'regions': region_labels,
            'connectivity_method': method,
            'weights_file': 'mmp_weights_360.npy',
            'tract_lengths_file': 'mmp_tract_lengths_360.npy',
            'centers_file': 'mmp_centers_360.npy',
            'shape': list(weights.shape),
            'hemispheres': {
                'left': '0-179',
                'right': '180-359'
            },
            'notes': 'Connectivity for TVB simulations using HCP-MMP1 360 cortical ROIs'
        }

        with open(output_dir / 'mmp_connectivity_360_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nSaved connectivity files to {output_dir}:")
        print(f"  - mmp_weights_360.npy")
        print(f"  - mmp_tract_lengths_360.npy")
        print(f"  - mmp_centers_360.npy")
        print(f"  - mmp_connectivity_360_metadata.json")

        return weights, tract_lengths, centers


def main():
    """Generate HCP-MMP1 connectivity matrices"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate HCP-MMP1 connectivity for TVB')
    parser.add_argument('--method', type=str, default='distance',
                       choices=['distance', 'uniform', 'structured'],
                       help='Connectivity generation method')
    parser.add_argument('--output', type=str, default='data/HCPMMP1',
                       help='Output directory')

    args = parser.parse_args()

    # Create atlas loader
    atlas = HCPMMP1Atlas()

    # Load atlas
    atlas.load_atlas()

    # Generate and save connectivity
    atlas.save_connectivity(output_dir=args.output, method=args.method)

    print("\nHCP-MMP1 connectivity generation complete!")
    print("Use this connectivity in TVB simulations with:")
    print("  python src/hcpmmp1_simulation.py --nodes 360")


if __name__ == '__main__':
    main()
