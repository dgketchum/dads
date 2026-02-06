"""
Master grid definition for DADS data cube.

Defines the unified 1km WGS84 coordinate system used by all cube layers.
Provides utilities for coordinate conversion, cell lookup, and spatial queries.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Union
from pathlib import Path

try:
    from affine import Affine
    HAS_AFFINE = True
except ImportError:
    HAS_AFFINE = False

try:
    from sklearn.neighbors import BallTree
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class MasterGrid:
    """
    Master grid definition for the data cube.

    Defines a regular lat/lon grid in WGS84 (EPSG:4326) at approximately
    1km resolution. All data layers are resampled to this common grid.

    Attributes:
        bounds: Spatial domain as (west, south, east, north) in degrees
        resolution_deg: Grid resolution in degrees
        crs: Coordinate reference system string

    The grid is defined with:
        - Origin at northwest corner (top-left)
        - Latitude decreasing southward (row index increases)
        - Longitude increasing eastward (column index increases)
        - Cell coordinates refer to cell centers
    """
    bounds: Tuple[float, float, float, float] = (-125.0, 24.0, -66.0, 50.0)
    resolution_deg: float = 0.008333333  # ~1km (1/120 degree)
    crs: str = 'EPSG:4326'

    # Computed arrays (lazy initialization)
    _lat: np.ndarray = field(default=None, repr=False, compare=False)
    _lon: np.ndarray = field(default=None, repr=False, compare=False)
    _tree: 'BallTree' = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        # Ensure bounds are in correct order
        w, s, e, n = self.bounds
        assert w < e, "West must be less than East"
        assert s < n, "South must be less than North"

    @property
    def west(self) -> float:
        return self.bounds[0]

    @property
    def south(self) -> float:
        return self.bounds[1]

    @property
    def east(self) -> float:
        return self.bounds[2]

    @property
    def north(self) -> float:
        return self.bounds[3]

    @property
    def n_lat(self) -> int:
        """Number of latitude cells (rows)."""
        return int(round((self.north - self.south) / self.resolution_deg))

    @property
    def n_lon(self) -> int:
        """Number of longitude cells (columns)."""
        return int(round((self.east - self.west) / self.resolution_deg))

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape as (n_lat, n_lon) = (rows, cols)."""
        return (self.n_lat, self.n_lon)

    @property
    def height(self) -> int:
        """Alias for n_lat (rasterio convention)."""
        return self.n_lat

    @property
    def width(self) -> int:
        """Alias for n_lon (rasterio convention)."""
        return self.n_lon

    @property
    def n_cells(self) -> int:
        """Total number of grid cells."""
        return self.n_lat * self.n_lon

    @property
    def lat(self) -> np.ndarray:
        """
        1D array of latitude values (cell centers).

        Ordered from north to south (decreasing).
        Shape: (n_lat,)
        """
        if self._lat is None:
            # Cell centers, starting from north
            half_res = self.resolution_deg / 2
            self._lat = np.linspace(
                self.north - half_res,
                self.south + half_res,
                self.n_lat
            )
        return self._lat

    @property
    def lon(self) -> np.ndarray:
        """
        1D array of longitude values (cell centers).

        Ordered from west to east (increasing).
        Shape: (n_lon,)
        """
        if self._lon is None:
            half_res = self.resolution_deg / 2
            self._lon = np.linspace(
                self.west + half_res,
                self.east - half_res,
                self.n_lon
            )
        return self._lon

    @property
    def transform(self) -> 'Affine':
        """
        Affine transform for georeferencing (rasterio convention).

        Maps pixel coordinates to geographic coordinates.
        Origin at top-left (northwest corner).
        """
        if not HAS_AFFINE:
            raise ImportError("affine package required for transform property")

        return Affine(
            self.resolution_deg,   # a: pixel width
            0.0,                   # b: row rotation (0 for north-up)
            self.west,             # c: x-coordinate of upper-left corner
            0.0,                   # d: column rotation (0 for north-up)
            -self.resolution_deg,  # e: pixel height (negative for north-up)
            self.north,            # f: y-coordinate of upper-left corner
        )

    def rowcol_to_latlon(
        self,
        row: Union[int, np.ndarray],
        col: Union[int, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert row/col indices to lat/lon coordinates (cell centers).

        Args:
            row: Row index or array of row indices
            col: Column index or array of column indices

        Returns:
            (lat, lon) tuple of coordinates
        """
        row = np.asarray(row)
        col = np.asarray(col)

        lat = self.lat[row]
        lon = self.lon[col]

        return lat, lon

    def latlon_to_rowcol(
        self,
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
    ) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray]]:
        """
        Convert lat/lon coordinates to row/col indices (nearest cell).

        Args:
            lat: Latitude or array of latitudes
            lon: Longitude or array of longitudes

        Returns:
            (row, col) tuple of indices
        """
        lat = np.asarray(lat)
        lon = np.asarray(lon)

        # Compute row (latitude decreases with row index)
        row = np.round((self.north - lat) / self.resolution_deg - 0.5).astype(int)
        row = np.clip(row, 0, self.n_lat - 1)

        # Compute column (longitude increases with col index)
        col = np.round((lon - self.west) / self.resolution_deg - 0.5).astype(int)
        col = np.clip(col, 0, self.n_lon - 1)

        return row, col

    def is_valid_cell(
        self,
        row: Union[int, np.ndarray],
        col: Union[int, np.ndarray],
    ) -> Union[bool, np.ndarray]:
        """
        Check if row/col indices are within grid bounds.

        Args:
            row: Row index or array
            col: Column index or array

        Returns:
            Boolean or boolean array
        """
        row = np.asarray(row)
        col = np.asarray(col)

        valid = (
            (row >= 0) & (row < self.n_lat) &
            (col >= 0) & (col < self.n_lon)
        )
        return valid

    def _build_tree(self, valid_mask: Optional[np.ndarray] = None):
        """
        Build BallTree for fast spatial queries.

        Args:
            valid_mask: Optional (n_lat, n_lon) boolean mask of valid cells
        """
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for spatial queries")

        # Create coordinate pairs for all cells
        lat_grid, lon_grid = np.meshgrid(self.lat, self.lon, indexing='ij')

        if valid_mask is not None:
            valid_rows, valid_cols = np.where(valid_mask)
            coords = np.column_stack([
                lat_grid[valid_rows, valid_cols],
                lon_grid[valid_rows, valid_cols],
            ])
            self._valid_indices = np.column_stack([valid_rows, valid_cols])
        else:
            coords = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
            self._valid_indices = None

        # Convert to radians for haversine
        coords_rad = np.deg2rad(coords)
        self._tree = BallTree(coords_rad, metric='haversine')

    def query_nearest(
        self,
        lat: float,
        lon: float,
        k: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest grid cells to a point.

        Args:
            lat: Query latitude
            lon: Query longitude
            k: Number of neighbors

        Returns:
            (indices, distances_km) where indices are into valid_indices if masked
        """
        if self._tree is None:
            self._build_tree()

        point = np.deg2rad([[lat, lon]])
        dist_rad, idx = self._tree.query(point, k=k)

        # Convert to km
        dist_km = dist_rad[0] * 6371.0

        return idx[0], dist_km

    def query_radius(
        self,
        lat: float,
        lon: float,
        radius_km: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find all grid cells within radius of a point.

        Args:
            lat: Query latitude
            lon: Query longitude
            radius_km: Search radius in kilometers

        Returns:
            (indices, distances_km)
        """
        if self._tree is None:
            self._build_tree()

        point = np.deg2rad([[lat, lon]])
        radius_rad = radius_km / 6371.0

        idx, dist_rad = self._tree.query_radius(
            point, r=radius_rad, return_distance=True
        )

        dist_km = dist_rad[0] * 6371.0
        return idx[0], dist_km

    def create_coords_dataset(self) -> dict:
        """
        Create coordinate arrays for zarr store initialization.

        Returns:
            Dict with 'lat', 'lon' arrays suitable for zarr
        """
        return {
            'lat': self.lat.astype(np.float64),
            'lon': self.lon.astype(np.float64),
        }

    def to_dict(self) -> dict:
        """Serialize grid parameters to dict."""
        return {
            'bounds': list(self.bounds),
            'resolution_deg': self.resolution_deg,
            'crs': self.crs,
            'shape': list(self.shape),
            'n_cells': self.n_cells,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'MasterGrid':
        """Create MasterGrid from dict."""
        return cls(
            bounds=tuple(d['bounds']),
            resolution_deg=d['resolution_deg'],
            crs=d.get('crs', 'EPSG:4326'),
        )

    def __repr__(self) -> str:
        return (
            f"MasterGrid(bounds={self.bounds}, "
            f"resolution={self.resolution_deg:.6f}°, "
            f"shape={self.shape}, "
            f"n_cells={self.n_cells:,})"
        )


def create_conus_grid() -> MasterGrid:
    """Create default CONUS grid at 1km resolution."""
    return MasterGrid(
        bounds=(-125.0, 24.0, -66.0, 50.0),
        resolution_deg=0.008333333,
        crs='EPSG:4326',
    )


def create_western_us_grid() -> MasterGrid:
    """Create Western US grid at 1km resolution."""
    return MasterGrid(
        bounds=(-125.0, 31.0, -102.0, 49.0),
        resolution_deg=0.008333333,
        crs='EPSG:4326',
    )


if __name__ == '__main__':
    # Demo
    grid = create_conus_grid()
    print(grid)
    print(f"\nCoordinate arrays:")
    print(f"  lat: [{grid.lat[0]:.4f}, ..., {grid.lat[-1]:.4f}] ({len(grid.lat)} values)")
    print(f"  lon: [{grid.lon[0]:.4f}, ..., {grid.lon[-1]:.4f}] ({len(grid.lon)} values)")

    # Test coordinate conversion
    test_lat, test_lon = 40.0, -105.0
    row, col = grid.latlon_to_rowcol(test_lat, test_lon)
    lat_back, lon_back = grid.rowcol_to_latlon(row, col)
    print(f"\nCoordinate conversion test:")
    print(f"  Input: ({test_lat}, {test_lon})")
    print(f"  Row/Col: ({row}, {col})")
    print(f"  Recovered: ({lat_back:.4f}, {lon_back:.4f})")
