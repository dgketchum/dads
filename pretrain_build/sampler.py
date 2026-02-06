"""
Samples grid cells for each epoch of pre-training.

Key design choice: the graph structure is rebuilt each epoch with
a fresh random sample of grid cells. This acts as data augmentation
and prevents the model from overfitting to a fixed spatial configuration.

The sampler maintains:
    - Current epoch's sampled cell indices
    - Precomputed neighbor relationships for those cells
    - Edge attributes (terrain differences, bearing, distance)
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from pretrain_build.grid_index import GridIndex
from pretrain_build.config import PretrainConfig


@dataclass
class EpochSample:
    """
    Holds the sampled graph structure for one epoch.

    Attributes:
        target_indices: (N,) indices into GridIndex for target cells
        neighbor_map: target_idx -> list of neighbor indices
        bearing_map: target_idx -> list of bearings (degrees) from neighbor to target
        distance_map: target_idx -> list of distances (km)
        terrain_delta_map: target_idx -> (k, D) array of terrain differences
    """
    target_indices: np.ndarray
    neighbor_map: Dict[int, List[int]] = field(default_factory=dict)
    bearing_map: Dict[int, List[float]] = field(default_factory=dict)
    distance_map: Dict[int, List[float]] = field(default_factory=dict)
    terrain_delta_map: Dict[int, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.target_indices)

    def get_neighbors(self, target_idx: int) -> Tuple[List[int], List[float], List[float], np.ndarray]:
        """
        Get neighbor info for a target cell.

        Args:
            target_idx: Index into GridIndex

        Returns:
            (neighbor_indices, bearings, distances, terrain_deltas)
        """
        return (
            self.neighbor_map.get(target_idx, []),
            self.bearing_map.get(target_idx, []),
            self.distance_map.get(target_idx, []),
            self.terrain_delta_map.get(target_idx, np.array([])),
        )


class EpochSampler:
    """
    Generates a new graph sample each epoch.

    The key insight is that by randomizing the graph structure across epochs,
    the model learns more generalizable spatial patterns rather than overfitting
    to a fixed set of neighbor relationships.

    Usage:
        sampler = EpochSampler(grid_index, config)
        for epoch in range(n_epochs):
            sample = sampler.new_epoch(seed=epoch)
            dataset = PretrainDataset(sample, ...)
            train_one_epoch(dataset)
    """

    def __init__(self, grid_index: GridIndex, config: PretrainConfig):
        """
        Initialize sampler with grid index and configuration.

        Args:
            grid_index: GridIndex with valid cells
            config: PretrainConfig with sampling parameters
        """
        self.index = grid_index
        self.config = config

    def new_epoch(self, seed: Optional[int] = None) -> EpochSample:
        """
        Generate a new random sample of grid cells with graph structure.

        Args:
            seed: Random seed for reproducibility

        Returns:
            EpochSample with target cells and neighbor relationships
        """
        rng = np.random.default_rng(seed)

        # Sample target cells
        n_cells = min(self.config.n_cells_per_epoch, len(self.index))
        target_indices = self.index.sample_cells(
            n=n_cells,
            strategy=self.config.sampling_strategy,
            min_spacing_km=self.config.min_cell_spacing_km,
            rng=rng,
        )

        # Build set for O(1) lookup
        target_set = set(target_indices.tolist())

        # Build neighbor relationships
        neighbor_map = {}
        bearing_map = {}
        distance_map = {}
        terrain_delta_map = {}

        # Query more neighbors than needed for filtering
        query_k = self.config.n_neighbors * self.config.neighbor_pool_factor

        valid_targets = []

        for tgt_idx in target_indices:
            tgt_lat, tgt_lon = self.index.coords[tgt_idx]
            tgt_terrain = self.index.terrain[tgt_idx]

            # Query potential neighbors
            nbr_idx, nbr_dist = self.index.query_neighbors(tgt_lat, tgt_lon, k=query_k)

            # Filter: must be in sampled set, not self, within max distance
            valid_neighbors = []
            valid_distances = []

            for ni, nd in zip(nbr_idx, nbr_dist):
                if ni == tgt_idx:
                    continue
                if ni not in target_set:
                    continue
                if nd > self.config.max_distance_km:
                    continue
                valid_neighbors.append(int(ni))
                valid_distances.append(float(nd))
                if len(valid_neighbors) >= self.config.n_neighbors:
                    break

            if len(valid_neighbors) < self.config.n_neighbors:
                # Not enough neighbors - skip this target
                continue

            # Compute bearings
            bearings = []
            for ni in valid_neighbors:
                nbr_lat, nbr_lon = self.index.coords[ni]
                bearing = _compute_bearing(nbr_lat, nbr_lon, tgt_lat, tgt_lon)
                bearings.append(bearing)

            # Compute terrain deltas (target - neighbor)
            nbr_terrains = self.index.terrain[valid_neighbors]
            terrain_delta = tgt_terrain - nbr_terrains  # (k, D)

            neighbor_map[int(tgt_idx)] = valid_neighbors
            bearing_map[int(tgt_idx)] = bearings
            distance_map[int(tgt_idx)] = valid_distances
            terrain_delta_map[int(tgt_idx)] = terrain_delta

            valid_targets.append(int(tgt_idx))

        return EpochSample(
            target_indices=np.array(valid_targets),
            neighbor_map=neighbor_map,
            bearing_map=bearing_map,
            distance_map=distance_map,
            terrain_delta_map=terrain_delta_map,
        )

    def create_fixed_sample(
        self,
        target_indices: np.ndarray,
        seed: Optional[int] = None,
    ) -> EpochSample:
        """
        Create sample with fixed target cells (useful for validation).

        Args:
            target_indices: Fixed array of target cell indices
            seed: Random seed for neighbor selection

        Returns:
            EpochSample with specified targets
        """
        rng = np.random.default_rng(seed)

        target_set = set(target_indices.tolist())
        query_k = self.config.n_neighbors * self.config.neighbor_pool_factor

        neighbor_map = {}
        bearing_map = {}
        distance_map = {}
        terrain_delta_map = {}

        valid_targets = []

        for tgt_idx in target_indices:
            tgt_lat, tgt_lon = self.index.coords[tgt_idx]
            tgt_terrain = self.index.terrain[tgt_idx]

            nbr_idx, nbr_dist = self.index.query_neighbors(tgt_lat, tgt_lon, k=query_k)

            valid_neighbors = []
            valid_distances = []

            for ni, nd in zip(nbr_idx, nbr_dist):
                if ni == tgt_idx:
                    continue
                if ni not in target_set:
                    continue
                if nd > self.config.max_distance_km:
                    continue
                valid_neighbors.append(int(ni))
                valid_distances.append(float(nd))
                if len(valid_neighbors) >= self.config.n_neighbors:
                    break

            if len(valid_neighbors) < self.config.n_neighbors:
                continue

            bearings = []
            for ni in valid_neighbors:
                nbr_lat, nbr_lon = self.index.coords[ni]
                bearing = _compute_bearing(nbr_lat, nbr_lon, tgt_lat, tgt_lon)
                bearings.append(bearing)

            nbr_terrains = self.index.terrain[valid_neighbors]
            terrain_delta = tgt_terrain - nbr_terrains

            neighbor_map[int(tgt_idx)] = valid_neighbors
            bearing_map[int(tgt_idx)] = bearings
            distance_map[int(tgt_idx)] = valid_distances
            terrain_delta_map[int(tgt_idx)] = terrain_delta

            valid_targets.append(int(tgt_idx))

        return EpochSample(
            target_indices=np.array(valid_targets),
            neighbor_map=neighbor_map,
            bearing_map=bearing_map,
            distance_map=distance_map,
            terrain_delta_map=terrain_delta_map,
        )


def _compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute bearing from (lat1, lon1) to (lat2, lon2) in degrees.

    Args:
        lat1, lon1: Source coordinates
        lat2, lon2: Destination coordinates

    Returns:
        Bearing in degrees (0-360, with 0=north)
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad

    x = np.sin(dlon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)

    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360.0) % 360.0


def compute_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute haversine distance between two points.

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        Distance in kilometers
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return 6371.0 * c  # Earth radius in km


class ValidationSampler:
    """
    Sampler for validation that holds out entire regions.

    Unlike training which uses random sampling, validation should test
    generalization to unseen regions rather than unseen cells within
    the same region.
    """

    def __init__(
        self,
        grid_index: GridIndex,
        config: PretrainConfig,
        holdout_bounds: Tuple[float, float, float, float],
    ):
        """
        Initialize validation sampler with holdout region.

        Args:
            grid_index: GridIndex with valid cells
            config: PretrainConfig
            holdout_bounds: (west, south, east, north) for validation region
        """
        self.index = grid_index
        self.config = config
        self.holdout_bounds = holdout_bounds

        # Identify cells in holdout region
        w, s, e, n = holdout_bounds
        mask = (
            (grid_index.coords[:, 0] >= s) &
            (grid_index.coords[:, 0] <= n) &
            (grid_index.coords[:, 1] >= w) &
            (grid_index.coords[:, 1] <= e)
        )
        self.val_indices = np.where(mask)[0]
        self.train_indices = np.where(~mask)[0]

        print(f"[ValidationSampler] Holdout region: {len(self.val_indices)} cells")
        print(f"[ValidationSampler] Training region: {len(self.train_indices)} cells")

    def get_train_indices(self) -> np.ndarray:
        """Get indices of training cells (outside holdout region)."""
        return self.train_indices

    def get_val_indices(self) -> np.ndarray:
        """Get indices of validation cells (inside holdout region)."""
        return self.val_indices

    def create_train_sample(self, seed: Optional[int] = None) -> EpochSample:
        """Create training sample from non-holdout region."""
        sampler = EpochSampler(self.index, self.config)

        # Sample from training indices only
        rng = np.random.default_rng(seed)
        n = min(self.config.n_cells_per_epoch, len(self.train_indices))

        if self.config.sampling_strategy == 'uniform':
            sampled = rng.choice(self.train_indices, size=n, replace=False)
        else:
            # Use poisson disk on training subset
            available = set(self.train_indices.tolist())
            sampled = []
            candidates = list(available)
            rng.shuffle(candidates)

            for idx in candidates:
                if idx not in available:
                    continue
                if len(sampled) >= n:
                    break

                sampled.append(idx)
                available.discard(idx)

                if self.config.min_cell_spacing_km:
                    lat, lon = self.index.coords[idx]
                    nbr_idx, nbr_dist = self.index.query_neighbors(
                        lat, lon, k=min(500, len(available) + 1)
                    )
                    for ni, nd in zip(nbr_idx, nbr_dist):
                        if nd < self.config.min_cell_spacing_km and ni in available:
                            available.discard(ni)

            sampled = np.array(sampled)

        return sampler.create_fixed_sample(sampled, seed=seed)

    def create_val_sample(self, seed: Optional[int] = None) -> EpochSample:
        """Create validation sample from holdout region."""
        sampler = EpochSampler(self.index, self.config)
        return sampler.create_fixed_sample(self.val_indices, seed=seed)


if __name__ == '__main__':
    # Example usage
    from pretrain_build.config import PretrainConfig
    from pretrain_build.grid_index import GridIndex
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sampler.py <grid_index_path>")
        sys.exit(1)

    index = GridIndex(sys.argv[1])
    config = PretrainConfig(
        n_cells_per_epoch=1000,
        n_neighbors=10,
        sampling_strategy='poisson_disk',
        min_cell_spacing_km=20.0,
    )

    sampler = EpochSampler(index, config)
    sample = sampler.new_epoch(seed=42)

    print(f"Sampled {len(sample)} valid targets")
    print(f"Example target {sample.target_indices[0]}:")
    neighbors, bearings, distances, terrain = sample.get_neighbors(sample.target_indices[0])
    print(f"  Neighbors: {neighbors}")
    print(f"  Distances: {[f'{d:.1f}km' for d in distances]}")
    print(f"  Bearings: {[f'{b:.0f}°' for b in bearings]}")
