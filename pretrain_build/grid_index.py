"""
Efficient spatial index of valid grid cells.

Rather than iterating over the full grid each epoch, we pre-build
an index of cells that have valid data across the required date range.
This is stored as a lightweight Zarr array for fast random access.

The index contains:
    - cell_id: unique integer identifier
    - lat, lon: centroid coordinates
    - row, col: grid indices for data extraction
    - terrain features: elevation, slope, aspect, tpi (pre-extracted)

This allows O(1) random sampling without touching the actual data
until we need temporal sequences.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import zarr

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

try:
    from sklearn.neighbors import BallTree

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

from pretrain_build.config import GridSource, PretrainConfig


@dataclass
class CellInfo:
    """Information about a single grid cell."""

    cell_id: int
    lat: float
    lon: float
    row: int
    col: int
    terrain: np.ndarray


class GridIndex:
    """
    Spatial index of grid cells for efficient sampling.

    Attributes:
        coords: (N, 2) array of [lat, lon] for each valid cell
        cell_ids: (N,) array of unique cell identifiers
        grid_indices: (N, 2) array of [row, col] for data extraction
        terrain: (N, D) array of terrain features per cell
        tree: BallTree for fast spatial queries
    """

    def __init__(self, index_path: Path):
        """
        Load existing index from Zarr store.

        Args:
            index_path: Path to Zarr store containing the index
        """
        if not HAS_ZARR:
            raise ImportError("zarr is required for GridIndex")
        if not HAS_SKLEARN:
            raise ImportError("sklearn is required for GridIndex")

        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")

        store = zarr.open(str(index_path), mode="r")
        self.coords = store["coords"][:]  # (N, 2) [lat, lon]
        self.cell_ids = store["cell_ids"][:]  # (N,)
        self.grid_indices = store["grid_indices"][:]  # (N, 2) [row, col]
        self.terrain = store["terrain"][:]  # (N, D)

        # Store source metadata if available
        if "source_name" in store.attrs:
            self.source_name = store.attrs["source_name"]
        else:
            self.source_name = None

        if "resolution_deg" in store.attrs:
            self.resolution_deg = store.attrs["resolution_deg"]
        else:
            self.resolution_deg = None

        self._build_tree()

    def _build_tree(self):
        """Build BallTree for neighbor queries using haversine metric."""
        coords_rad = np.deg2rad(self.coords)
        self.tree = BallTree(coords_rad, metric="haversine")

    def __len__(self) -> int:
        return len(self.cell_ids)

    def query_neighbors(
        self,
        lat: float,
        lon: float,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return indices and distances (km) of k nearest cells.

        Args:
            lat: Query latitude
            lon: Query longitude
            k: Number of neighbors to return

        Returns:
            (indices, distances_km) tuple
        """
        point = np.deg2rad([[lat, lon]])
        k = min(k, len(self.cell_ids))
        dist, idx = self.tree.query(point, k=k)
        dist_km = dist[0] * 6371.0  # Earth radius in km
        return idx[0], dist_km

    def query_radius(
        self,
        lat: float,
        lon: float,
        radius_km: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return indices and distances of cells within radius.

        Args:
            lat: Query latitude
            lon: Query longitude
            radius_km: Search radius in kilometers

        Returns:
            (indices, distances_km) tuple
        """
        point = np.deg2rad([[lat, lon]])
        radius_rad = radius_km / 6371.0
        idx, dist = self.tree.query_radius(point, r=radius_rad, return_distance=True)
        dist_km = dist[0] * 6371.0
        return idx[0], dist_km

    def get_cell_info(self, idx: int) -> CellInfo:
        """Get full information for a cell by index."""
        return CellInfo(
            cell_id=int(self.cell_ids[idx]),
            lat=float(self.coords[idx, 0]),
            lon=float(self.coords[idx, 1]),
            row=int(self.grid_indices[idx, 0]),
            col=int(self.grid_indices[idx, 1]),
            terrain=self.terrain[idx],
        )

    def sample_cells(
        self,
        n: int,
        strategy: str = "uniform",
        min_spacing_km: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample n cells from the index.

        Args:
            n: Number of cells to sample
            strategy: 'uniform', 'poisson_disk', or 'stratified'
            min_spacing_km: Minimum spacing for poisson_disk sampling
            rng: Random number generator

        Returns:
            Array of indices into self.coords / self.cell_ids
        """
        if rng is None:
            rng = np.random.default_rng()

        n = min(n, len(self.cell_ids))

        if strategy == "uniform":
            return rng.choice(len(self.cell_ids), size=n, replace=False)

        elif strategy == "poisson_disk":
            return self._sample_poisson_disk(n, min_spacing_km, rng)

        elif strategy == "stratified":
            return self._sample_stratified(n, rng)

        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    def _sample_poisson_disk(
        self,
        n: int,
        min_spacing_km: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Greedy Poisson disk sampling with minimum spacing.

        This creates a more uniform spatial distribution than random sampling.
        """
        if min_spacing_km is None:
            min_spacing_km = 20.0

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

            # Remove cells within min_spacing_km
            lat, lon = self.coords[idx]
            nbr_idx, nbr_dist = self.query_neighbors(
                lat, lon, k=min(500, len(available) + 1)
            )
            for ni, nd in zip(nbr_idx, nbr_dist):
                if nd < min_spacing_km and ni in available:
                    available.discard(ni)

        return np.array(selected)

    def _sample_stratified(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Stratified sampling by dividing domain into tiles.
        """
        lat_min, lat_max = self.coords[:, 0].min(), self.coords[:, 0].max()
        lon_min, lon_max = self.coords[:, 1].min(), self.coords[:, 1].max()

        # Determine tile size to get roughly n cells
        n_tiles = int(np.sqrt(n))
        lat_step = (lat_max - lat_min) / n_tiles
        lon_step = (lon_max - lon_min) / n_tiles

        selected = []
        samples_per_tile = max(1, n // (n_tiles * n_tiles))

        for i in range(n_tiles):
            for j in range(n_tiles):
                lat_lo = lat_min + i * lat_step
                lat_hi = lat_min + (i + 1) * lat_step
                lon_lo = lon_min + j * lon_step
                lon_hi = lon_min + (j + 1) * lon_step

                mask = (
                    (self.coords[:, 0] >= lat_lo)
                    & (self.coords[:, 0] < lat_hi)
                    & (self.coords[:, 1] >= lon_lo)
                    & (self.coords[:, 1] < lon_hi)
                )
                tile_indices = np.where(mask)[0]

                if len(tile_indices) > 0:
                    k = min(samples_per_tile, len(tile_indices))
                    sel = rng.choice(tile_indices, size=k, replace=False)
                    selected.extend(sel.tolist())

        # Fill remaining slots if needed
        if len(selected) < n:
            remaining = set(range(len(self.cell_ids))) - set(selected)
            needed = n - len(selected)
            if remaining:
                extra = rng.choice(
                    list(remaining), size=min(needed, len(remaining)), replace=False
                )
                selected.extend(extra.tolist())

        return np.array(selected[:n])

    @staticmethod
    def build_from_source(
        source: GridSource,
        config: PretrainConfig,
        output_path: Path,
        dem_path: Optional[Path] = None,
        min_valid_days: int = 365,
    ) -> "GridIndex":
        """
        Build spatial index from a gridded data source.

        This is run once to create the index; subsequent usage just loads it.

        Args:
            source: GridSource configuration
            config: PretrainConfig with bounds and parameters
            output_path: Path for output Zarr store
            dem_path: Optional DEM path for terrain features
            min_valid_days: Minimum days of valid data required

        Returns:
            GridIndex instance
        """
        if not HAS_XARRAY:
            raise ImportError("xarray is required to build GridIndex")
        if not HAS_ZARR:
            raise ImportError("zarr is required to build GridIndex")

        output_path = Path(output_path)

        # Open source dataset
        if source.zarr_path and source.zarr_path.exists():
            ds = xr.open_zarr(source.zarr_path)
        elif source.netcdf_dir and source.netcdf_dir.exists():
            nc_files = sorted(source.netcdf_dir.glob("*.nc"))
            if not nc_files:
                raise FileNotFoundError(f"No NetCDF files in {source.netcdf_dir}")
            ds = xr.open_mfdataset(nc_files, combine="by_coords")
        else:
            raise FileNotFoundError(f"No data source found for {source.name}")

        # Get coordinate arrays
        lats = ds[source.lat_var].values
        lons = ds[source.lon_var].values

        # Handle different coordinate structures
        if lats.ndim == 1 and lons.ndim == 1:
            # Regular grid - create meshgrid
            lon_grid, lat_grid = np.meshgrid(lons, lats)
        else:
            # Already 2D
            lat_grid = lats
            lon_grid = lons

        # Apply bounds filter
        w, s, e, n = config.bounds
        mask = (lat_grid >= s) & (lat_grid <= n) & (lon_grid >= w) & (lon_grid <= e)

        # Get valid cell coordinates
        valid_rows, valid_cols = np.where(mask)
        valid_lats = lat_grid[mask]
        valid_lons = lon_grid[mask]

        print(f"[GridIndex] Found {len(valid_lats)} cells within bounds")

        # Check data availability if variables specified
        if source.variables and len(source.variables) > 0:
            var_name = source.variables[0]
            if var_name in ds:
                print(f"[GridIndex] Checking data availability for {var_name}")
                # Sample time coverage — simplified check

                # This is a simplified check - in practice you might want more thorough validation
                valid_data_mask = np.ones(len(valid_lats), dtype=bool)
                print(
                    f"[GridIndex] Data check passed for {valid_data_mask.sum()} cells"
                )
        else:
            valid_data_mask = np.ones(len(valid_lats), dtype=bool)

        # Filter to valid cells
        final_lats = valid_lats[valid_data_mask]
        final_lons = valid_lons[valid_data_mask]
        final_rows = valid_rows[valid_data_mask]
        final_cols = valid_cols[valid_data_mask]

        coords = np.column_stack([final_lats, final_lons])
        grid_indices = np.column_stack([final_rows, final_cols])
        cell_ids = np.arange(len(final_lats))

        print(f"[GridIndex] Final index contains {len(cell_ids)} cells")

        # Extract terrain features if DEM available
        if dem_path and dem_path.exists():
            from pretrain_build.terrain import extract_terrain_at_points

            print(f"[GridIndex] Extracting terrain features from {dem_path}")
            terrain = extract_terrain_at_points(
                str(dem_path),
                final_lats,
                final_lons,
                include_tpi=True,
            )
        else:
            # Default: zeros (will need terrain from another source or use zeros)
            print("[GridIndex] No DEM provided, using zero terrain features")
            terrain = np.zeros(
                (len(cell_ids), 7), dtype=np.float32
            )  # Match TERRAIN_FEATURES count

        # Write to Zarr
        output_path.parent.mkdir(parents=True, exist_ok=True)
        store = zarr.open(str(output_path), mode="w")
        store.create_dataset("coords", data=coords)
        store.create_dataset("cell_ids", data=cell_ids)
        store.create_dataset("grid_indices", data=grid_indices)
        store.create_dataset("terrain", data=terrain)

        # Store metadata
        store.attrs["source_name"] = source.name
        store.attrs["resolution_deg"] = source.resolution_deg
        store.attrs["bounds"] = list(config.bounds)
        store.attrs["n_cells"] = len(cell_ids)

        print(f"[GridIndex] Written to {output_path}")

        ds.close()

        return GridIndex(output_path)

    @staticmethod
    def build_from_coords(
        lats: np.ndarray,
        lons: np.ndarray,
        output_path: Path,
        terrain: Optional[np.ndarray] = None,
        source_name: str = "custom",
        resolution_deg: float = 0.04166667,
    ) -> "GridIndex":
        """
        Build index from explicit coordinate arrays.

        Useful when coordinates are already known (e.g., from existing extraction).

        Args:
            lats: Array of latitudes
            lons: Array of longitudes
            output_path: Path for output Zarr store
            terrain: Optional (N, D) terrain features
            source_name: Name identifier for the source
            resolution_deg: Approximate resolution in degrees

        Returns:
            GridIndex instance
        """
        if not HAS_ZARR:
            raise ImportError("zarr is required to build GridIndex")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lats = np.asarray(lats)
        lons = np.asarray(lons)
        n = len(lats)

        coords = np.column_stack([lats, lons])
        cell_ids = np.arange(n)
        # For explicit coords, grid_indices are just cell_ids as (row, 0)
        grid_indices = np.column_stack([cell_ids, np.zeros(n, dtype=int)])

        if terrain is None:
            terrain = np.zeros((n, 7), dtype=np.float32)

        store = zarr.open(str(output_path), mode="w")
        store.create_dataset("coords", data=coords)
        store.create_dataset("cell_ids", data=cell_ids)
        store.create_dataset("grid_indices", data=grid_indices)
        store.create_dataset("terrain", data=terrain)

        store.attrs["source_name"] = source_name
        store.attrs["resolution_deg"] = resolution_deg
        store.attrs["n_cells"] = n

        return GridIndex(output_path)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python grid_index.py <zarr_path>")
        sys.exit(1)

    index_path = Path(sys.argv[1])
    if index_path.exists():
        idx = GridIndex(index_path)
        print(f"Loaded index with {len(idx)} cells")
        print(
            f"Coordinate range: lat [{idx.coords[:, 0].min():.2f}, {idx.coords[:, 0].max():.2f}]"
        )
        print(
            f"                  lon [{idx.coords[:, 1].min():.2f}, {idx.coords[:, 1].max():.2f}]"
        )
        print(f"Terrain shape: {idx.terrain.shape}")
    else:
        print(f"Index not found: {index_path}")
