"""
Adapter classes for integration with pretrain_build pipeline.

Provides drop-in replacements for pretrain_build/sequences.py and
pretrain_build/grid_index.py that source data from the unified cube.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from cube.pretrain.config import PretrainConfig

import numpy as np

try:
    from sklearn.neighbors import BallTree

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from cube.extractors.data_cube_extractor import DataCubeExtractor

logger = logging.getLogger(__name__)


class CubeSequenceExtractor:
    """
    Drop-in replacement for pretrain_build/sequences.SequenceExtractor.

    Implements the same interface but sources data from the unified cube
    instead of individual Zarr/NetCDF files.

    This allows existing PretrainDataset code to work with the cube
    without modification.
    """

    def __init__(
        self,
        cube_path: str,
        config: "PretrainConfig",
    ):
        """
        Initialize adapter.

        Args:
            cube_path: Path to cube.zarr store
            config: PretrainConfig from pretrain_build.config
        """
        self.cube = DataCubeExtractor(cube_path)
        self.config = config
        self._cache: Dict[Tuple, Tuple[np.ndarray, bool]] = {}
        self._cache_order = []
        self._cache_size = 10000

    def extract_sequence(
        self,
        lat: float,
        lon: float,
        end_date: np.datetime64,
        variable: str,
        terrain_vec: np.ndarray,
        row: Optional[int] = None,
        col: Optional[int] = None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Extract sequence from cube.

        Same interface as pretrain_build/sequences.SequenceExtractor.extract_sequence.

        Args:
            lat: Latitude of target cell
            lon: Longitude of target cell
            end_date: End date of sequence (inclusive)
            variable: Target variable name (e.g., 'tmax')
            terrain_vec: Pre-extracted terrain features (ignored - extracted from cube)
            row: Optional grid row index (for faster indexing)
            col: Optional grid col index (for faster indexing)

        Returns:
            (sequence, valid) tuple where:
                sequence: (seq_len, n_features) array
                valid: whether the sequence has complete data
        """
        # Use row/col if provided, otherwise find nearest cell
        if row is None or col is None:
            row, col = self.cube.grid.latlon_to_rowcol(lat, lon)

        # Check cache
        cache_key = (row, col, str(end_date), variable)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Extract from cube
        target_seq, exog_seq, terrain, valid = self.cube.extract_sequence(
            row=int(row),
            col=int(col),
            end_date=end_date,
            target_variable=variable,
            seq_len=self.config.seq_len,
        )

        if not valid:
            # Return format matching SequenceExtractor
            n_features = 1 + exog_seq.shape[1] if exog_seq.shape[0] > 0 else 1
            sequence = np.zeros((self.config.seq_len, n_features), dtype=np.float32)
            result = (sequence, False)
        else:
            # Combine target + exog to match SequenceExtractor output format
            # First column is target, rest are exog
            sequence = np.column_stack([target_seq, exog_seq])
            result = (sequence.astype(np.float32), True)

        # Cache with LRU eviction
        if len(self._cache) >= self._cache_size:
            old_key = self._cache_order.pop(0)
            del self._cache[old_key]
        self._cache[cache_key] = result
        self._cache_order.append(cache_key)

        return result

    def clear_cache(self):
        """Clear the sequence cache."""
        self._cache.clear()
        self._cache_order.clear()

    def close(self):
        """Close underlying cube extractor."""
        self.cube.close()

    def get_available_dates(
        self, source_name: str = None
    ) -> Tuple[np.datetime64, np.datetime64]:
        """Get available date range."""
        return self.cube.get_date_range()


class CubeGridIndex:
    """
    Drop-in replacement for pretrain_build/grid_index.GridIndex.

    Uses the cube's land mask and coordinates directly instead of
    maintaining a separate spatial index. BallTree uses euclidean metric
    on EPSG:5070 projected coordinates for exact distance queries.
    """

    def __init__(self, cube_path: str):
        """
        Initialize from cube.

        Args:
            cube_path: Path to cube.zarr store
        """
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for CubeGridIndex")

        self.cube = DataCubeExtractor(cube_path)

        # Get valid cells from land mask
        valid_cells = self.cube.get_valid_cells()
        self.n_cells = len(valid_cells)

        # Store row/col indices
        self.grid_indices = valid_cells  # (N, 2) [row, col]

        # Store projected coordinates for BallTree (easting, northing)
        self.proj_coords = np.stack(
            [
                self.cube.x_coords[valid_cells[:, 1]],
                self.cube.y_coords[valid_cells[:, 0]],
            ],
            axis=-1,
        )  # (N, 2) [easting, northing]

        # Store geographic coordinates for feature vectors
        self.geo_coords = np.stack(
            [
                self.cube.lat_2d[valid_cells[:, 0], valid_cells[:, 1]],
                self.cube.lon_2d[valid_cells[:, 0], valid_cells[:, 1]],
            ],
            axis=-1,
        )  # (N, 2) [lat, lon]

        # Cell IDs (simple sequential)
        self.cell_ids = np.arange(self.n_cells)

        # Extract terrain for all valid cells
        logger.info(f"Extracting terrain for {self.n_cells:,} valid cells...")
        self.terrain = np.array(
            [self.cube._get_terrain(int(r), int(c)) for r, c in valid_cells]
        )

        # Build spatial tree
        self._build_tree()

    def _build_tree(self):
        """Build BallTree for neighbor queries using euclidean metric on projected coords."""
        self.tree = BallTree(self.proj_coords, metric="euclidean")

    def __len__(self) -> int:
        return self.n_cells

    def query_neighbors(
        self,
        lat: float,
        lon: float,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return indices and distances (meters) of k nearest cells.

        Args:
            lat: Query latitude
            lon: Query longitude
            k: Number of neighbors to return

        Returns:
            (indices, distances_m) tuple
        """
        # Project query point to EPSG:5070
        easting, northing = self.cube.grid.latlon_to_xy(lat, lon)
        point = np.array([[easting, northing]])
        k = min(k, len(self.cell_ids))
        dist, idx = self.tree.query(point, k=k)
        return idx[0], dist[0]

    def query_radius(
        self,
        lat: float,
        lon: float,
        radius_m: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return indices and distances of cells within radius.

        Args:
            lat: Query latitude
            lon: Query longitude
            radius_m: Search radius in meters

        Returns:
            (indices, distances_m) tuple
        """
        easting, northing = self.cube.grid.latlon_to_xy(lat, lon)
        point = np.array([[easting, northing]])
        idx, dist = self.tree.query_radius(point, r=radius_m, return_distance=True)
        return idx[0], dist[0]

    def sample_cells(
        self,
        n: int,
        strategy: str = "uniform",
        min_spacing_m: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample n cells from the index.

        Args:
            n: Number of cells to sample
            strategy: 'uniform', 'poisson_disk', or 'stratified'
            min_spacing_m: Minimum spacing in meters for poisson_disk sampling
            rng: Random number generator

        Returns:
            Array of indices into self.proj_coords / self.cell_ids
        """
        if rng is None:
            rng = np.random.default_rng()

        n = min(n, len(self.cell_ids))

        if strategy == "uniform":
            return rng.choice(len(self.cell_ids), size=n, replace=False)

        elif strategy == "poisson_disk":
            return self._sample_poisson_disk(n, min_spacing_m or 20000.0, rng)

        elif strategy == "stratified":
            return self._sample_stratified(n, rng)

        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    def _sample_poisson_disk(
        self,
        n: int,
        min_spacing_m: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Greedy Poisson disk sampling with minimum spacing in meters."""
        selected = []
        available = set(range(len(self.cell_ids)))
        candidates = list(available)
        rng.shuffle(candidates)

        for idx in candidates:
            if idx not in available:
                continue
            if len(selected) >= n:
                break

            selected.append(idx)
            available.discard(idx)

            # Remove cells within min_spacing_m using projected coords
            point = self.proj_coords[idx : idx + 1]
            nbr_idx_arr, nbr_dist_arr = self.tree.query_radius(
                point, r=min_spacing_m, return_distance=True
            )
            for ni, nd in zip(nbr_idx_arr[0], nbr_dist_arr[0]):
                if nd < min_spacing_m and ni in available:
                    available.discard(ni)

        return np.array(selected)

    def _sample_stratified(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Stratified sampling by dividing domain into tiles using projected coords."""
        x_min, x_max = self.proj_coords[:, 0].min(), self.proj_coords[:, 0].max()
        y_min, y_max = self.proj_coords[:, 1].min(), self.proj_coords[:, 1].max()

        n_tiles = int(np.sqrt(n))
        x_step = (x_max - x_min) / n_tiles
        y_step = (y_max - y_min) / n_tiles

        selected = []
        samples_per_tile = max(1, n // (n_tiles * n_tiles))

        for i in range(n_tiles):
            for j in range(n_tiles):
                x_lo = x_min + j * x_step
                x_hi = x_min + (j + 1) * x_step
                y_lo = y_min + i * y_step
                y_hi = y_min + (i + 1) * y_step

                mask = (
                    (self.proj_coords[:, 0] >= x_lo)
                    & (self.proj_coords[:, 0] < x_hi)
                    & (self.proj_coords[:, 1] >= y_lo)
                    & (self.proj_coords[:, 1] < y_hi)
                )
                tile_indices = np.where(mask)[0]

                if len(tile_indices) > 0:
                    k = min(samples_per_tile, len(tile_indices))
                    sel = rng.choice(tile_indices, size=k, replace=False)
                    selected.extend(sel.tolist())

        # Fill remaining if needed
        if len(selected) < n:
            remaining = set(range(len(self.cell_ids))) - set(selected)
            needed = n - len(selected)
            if remaining:
                extra = rng.choice(
                    list(remaining), size=min(needed, len(remaining)), replace=False
                )
                selected.extend(extra.tolist())

        return np.array(selected[:n])

    def close(self):
        """Close underlying cube extractor."""
        self.cube.close()


def create_cube_adapter(
    cube_path: str, config: "PretrainConfig"
) -> Tuple[CubeGridIndex, CubeSequenceExtractor]:
    """
    Create cube-based replacements for GridIndex and SequenceExtractor.

    Convenience function for setting up cube-based pre-training.

    Args:
        cube_path: Path to cube.zarr store
        config: PretrainConfig from pretrain_build.config

    Returns:
        (grid_index, sequence_extractor) tuple
    """
    grid_index = CubeGridIndex(cube_path)
    seq_extractor = CubeSequenceExtractor(cube_path, config)
    return grid_index, seq_extractor


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cube_path = sys.argv[1]
    else:
        cube_path = "/data/ssd2/dads_cube/cube.zarr"

    if Path(cube_path).exists():
        print("Testing CubeGridIndex...")
        index = CubeGridIndex(cube_path)
        print(f"  {len(index):,} valid cells")
        print(
            f"  Projected coord range: x [{index.proj_coords[:, 0].min():.0f}, {index.proj_coords[:, 0].max():.0f}]"
        )
        print(f"  Terrain shape: {index.terrain.shape}")

        # Test sampling
        sample = index.sample_cells(100, strategy="poisson_disk", min_spacing_m=50000.0)
        print(f"  Sampled {len(sample)} cells with poisson_disk")

        index.close()
    else:
        print(f"Cube not found at {cube_path}")
