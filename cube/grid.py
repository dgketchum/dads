"""
Master grid definition for DADS data cube.

Defines the unified 1km EPSG:5070 (Albers Equal Area Conic) coordinate system
used by all cube layers. Provides utilities for coordinate conversion, cell
lookup, and spatial queries.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np

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

try:
    from pyproj import Transformer

    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False


def _get_transformers():
    """Lazy-init pyproj Transformer pair (5070<->4326, always_xy=True)."""
    if not HAS_PYPROJ:
        raise ImportError("pyproj required for coordinate transformations")
    to_geo = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    to_proj = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    return to_geo, to_proj


def _geographic_bounds_to_albers(w, s, e, n, resolution=1000.0):
    """
    Convert geographic (lon/lat) bounds to EPSG:5070, snapped outward to resolution.

    Densely samples the boundary to find the enclosing rectangle in projected
    coordinates, then snaps outward to the nearest resolution step.

    Args:
        w, s, e, n: Geographic bounds (west, south, east, north) in degrees
        resolution: Grid resolution in meters for snapping

    Returns:
        (x_min, y_min, x_max, y_max) in EPSG:5070 meters, snapped to resolution
    """
    _, to_proj = _get_transformers()

    # Densely sample the geographic boundary
    n_pts = 200
    lons = np.concatenate(
        [
            np.linspace(w, e, n_pts),  # south edge
            np.full(n_pts, e),  # east edge
            np.linspace(e, w, n_pts),  # north edge
            np.full(n_pts, w),  # west edge
        ]
    )
    lats = np.concatenate(
        [
            np.full(n_pts, s),  # south edge
            np.linspace(s, n, n_pts),  # east edge
            np.full(n_pts, n),  # north edge
            np.linspace(n, s, n_pts),  # west edge
        ]
    )

    # always_xy: (lon, lat) -> (easting, northing)
    xs, ys = to_proj.transform(lons, lats)

    # Enclosing rectangle
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Snap outward to resolution
    x_min = np.floor(x_min / resolution) * resolution
    y_min = np.floor(y_min / resolution) * resolution
    x_max = np.ceil(x_max / resolution) * resolution
    y_max = np.ceil(y_max / resolution) * resolution

    return (x_min, y_min, x_max, y_max)


@dataclass
class MasterGrid:
    """
    Master grid definition for the data cube.

    Defines a regular grid in EPSG:5070 (Albers Equal Area Conic) at 1000m
    resolution. All data layers are resampled to this common grid.

    Attributes:
        bounds: Spatial domain as (x_min, y_min, x_max, y_max) in EPSG:5070 meters
        resolution: Grid resolution in meters (default 1000.0)
        crs: Coordinate reference system string

    The grid is defined with:
        - Origin at upper-left (x_min, y_max)
        - y (northing) decreasing southward (row index increases)
        - x (easting) increasing eastward (column index increases)
        - Cell coordinates refer to cell centers
    """

    bounds: Tuple[float, float, float, float] = (
        -2361000.0,
        258000.0,
        2264000.0,
        3177000.0,
    )
    resolution: float = 1000.0
    crs: str = "EPSG:5070"

    # Cached arrays (lazy initialization)
    _y: np.ndarray = field(default=None, repr=False, compare=False)
    _x: np.ndarray = field(default=None, repr=False, compare=False)
    _lat: np.ndarray = field(default=None, repr=False, compare=False)
    _lon: np.ndarray = field(default=None, repr=False, compare=False)
    _tree: "BallTree" = field(default=None, repr=False, compare=False)
    _valid_indices: np.ndarray = field(default=None, repr=False, compare=False)
    _to_geo: object = field(default=None, repr=False, compare=False)
    _to_proj: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        x_min, y_min, x_max, y_max = self.bounds
        assert x_min < x_max, "x_min must be less than x_max"
        assert y_min < y_max, "y_min must be less than y_max"

    def _ensure_transformers(self):
        """Lazy-init pyproj transformers."""
        if self._to_geo is None:
            self._to_geo, self._to_proj = _get_transformers()

    @property
    def n_y(self) -> int:
        """Number of rows (northing cells)."""
        x_min, y_min, x_max, y_max = self.bounds
        return int(round((y_max - y_min) / self.resolution))

    @property
    def n_x(self) -> int:
        """Number of columns (easting cells)."""
        x_min, y_min, x_max, y_max = self.bounds
        return int(round((x_max - x_min) / self.resolution))

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape as (n_y, n_x) = (rows, cols)."""
        return (self.n_y, self.n_x)

    @property
    def height(self) -> int:
        """Alias for n_y (rasterio convention)."""
        return self.n_y

    @property
    def width(self) -> int:
        """Alias for n_x (rasterio convention)."""
        return self.n_x

    @property
    def n_cells(self) -> int:
        """Total number of grid cells."""
        return self.n_y * self.n_x

    @property
    def y(self) -> np.ndarray:
        """
        1D array of northing values (cell centers).

        Ordered from north to south (decreasing).
        Shape: (n_y,)
        """
        if self._y is None:
            x_min, y_min, x_max, y_max = self.bounds
            half_res = self.resolution / 2
            self._y = np.linspace(
                y_max - half_res,
                y_min + half_res,
                self.n_y,
            )
        return self._y

    @property
    def x(self) -> np.ndarray:
        """
        1D array of easting values (cell centers).

        Ordered from west to east (increasing).
        Shape: (n_x,)
        """
        if self._x is None:
            x_min, y_min, x_max, y_max = self.bounds
            half_res = self.resolution / 2
            self._x = np.linspace(
                x_min + half_res,
                x_max - half_res,
                self.n_x,
            )
        return self._x

    @property
    def lat(self) -> np.ndarray:
        """
        2D array of latitude values (cell centers) via pyproj.

        Shape: (n_y, n_x)
        """
        if self._lat is None:
            self._compute_latlon()
        return self._lat

    @property
    def lon(self) -> np.ndarray:
        """
        2D array of longitude values (cell centers) via pyproj.

        Shape: (n_y, n_x)
        """
        if self._lon is None:
            self._compute_latlon()
        return self._lon

    def _compute_latlon(self):
        """Compute 2D lat/lon arrays from projected y/x via pyproj."""
        self._ensure_transformers()
        xx, yy = np.meshgrid(self.x, self.y)
        # always_xy: (easting, northing) -> (lon, lat)
        lon_2d, lat_2d = self._to_geo.transform(xx, yy)
        self._lat = lat_2d.astype(np.float64)
        self._lon = lon_2d.astype(np.float64)

    @property
    def transform(self) -> "Affine":
        """
        Affine transform for georeferencing (rasterio convention).

        Maps pixel coordinates to projected coordinates.
        Origin at upper-left (x_min, y_max).
        """
        if not HAS_AFFINE:
            raise ImportError("affine package required for transform property")

        x_min, y_min, x_max, y_max = self.bounds
        return Affine(
            self.resolution,  # a: pixel width in meters
            0.0,  # b: row rotation (0 for north-up)
            x_min,  # c: x-coordinate of upper-left corner
            0.0,  # d: column rotation (0 for north-up)
            -self.resolution,  # e: pixel height (negative for north-up)
            y_max,  # f: y-coordinate of upper-left corner
        )

    def rowcol_to_xy(
        self,
        row: Union[int, np.ndarray],
        col: Union[int, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert row/col indices to (easting, northing) coordinates (cell centers).

        Args:
            row: Row index or array of row indices
            col: Column index or array of column indices

        Returns:
            (easting, northing) tuple of coordinates
        """
        row = np.asarray(row)
        col = np.asarray(col)
        return self.x[col], self.y[row]

    def xy_to_rowcol(
        self,
        easting: Union[float, np.ndarray],
        northing: Union[float, np.ndarray],
    ) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray]]:
        """
        Convert (easting, northing) to row/col indices (nearest cell).

        Args:
            easting: Easting or array of eastings
            northing: Northing or array of northings

        Returns:
            (row, col) tuple of indices
        """
        easting = np.asarray(easting, dtype=np.float64)
        northing = np.asarray(northing, dtype=np.float64)

        x_min, y_min, x_max, y_max = self.bounds

        # Row: northing decreases with row index
        row = np.round((y_max - northing) / self.resolution - 0.5).astype(int)
        row = np.clip(row, 0, self.n_y - 1)

        # Col: easting increases with col index
        col = np.round((easting - x_min) / self.resolution - 0.5).astype(int)
        col = np.clip(col, 0, self.n_x - 1)

        return row, col

    def latlon_to_xy(
        self,
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert (lat, lon) to (easting, northing) via pyproj.

        Args:
            lat: Latitude(s)
            lon: Longitude(s)

        Returns:
            (easting, northing)
        """
        self._ensure_transformers()
        # always_xy: (lon, lat) -> (easting, northing)
        easting, northing = self._to_proj.transform(
            np.asarray(lon, dtype=np.float64),
            np.asarray(lat, dtype=np.float64),
        )
        return easting, northing

    def xy_to_latlon(
        self,
        easting: Union[float, np.ndarray],
        northing: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert (easting, northing) to (lat, lon) via pyproj.

        Args:
            easting: Easting(s)
            northing: Northing(s)

        Returns:
            (lat, lon)
        """
        self._ensure_transformers()
        # always_xy: (easting, northing) -> (lon, lat)
        lon, lat = self._to_geo.transform(
            np.asarray(easting, dtype=np.float64),
            np.asarray(northing, dtype=np.float64),
        )
        return lat, lon

    def latlon_to_rowcol(
        self,
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
    ) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray]]:
        """
        Convert (lat, lon) to row/col indices via projection then snapping.

        Args:
            lat: Latitude(s)
            lon: Longitude(s)

        Returns:
            (row, col) tuple of indices
        """
        easting, northing = self.latlon_to_xy(lat, lon)
        return self.xy_to_rowcol(easting, northing)

    def rowcol_to_latlon(
        self,
        row: Union[int, np.ndarray],
        col: Union[int, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert row/col indices to (lat, lon) via cell center then inverse projection.

        Args:
            row: Row index or array of row indices
            col: Column index or array of column indices

        Returns:
            (lat, lon) tuple of coordinates
        """
        easting, northing = self.rowcol_to_xy(row, col)
        return self.xy_to_latlon(easting, northing)

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

        valid = (row >= 0) & (row < self.n_y) & (col >= 0) & (col < self.n_x)
        return valid

    def _build_tree(self, valid_mask: Optional[np.ndarray] = None):
        """
        Build BallTree for fast spatial queries using euclidean metric
        on projected (easting, northing) coordinates.

        Args:
            valid_mask: Optional (n_y, n_x) boolean mask of valid cells
        """
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for spatial queries")

        x_grid, y_grid = np.meshgrid(self.x, self.y)

        if valid_mask is not None:
            valid_rows, valid_cols = np.where(valid_mask)
            coords = np.column_stack(
                [
                    x_grid[valid_rows, valid_cols],
                    y_grid[valid_rows, valid_cols],
                ]
            )
            self._valid_indices = np.column_stack([valid_rows, valid_cols])
        else:
            coords = np.column_stack([x_grid.ravel(), y_grid.ravel()])
            self._valid_indices = None

        self._tree = BallTree(coords, metric="euclidean")

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
            (indices, distances_m) where distances are in meters
        """
        if self._tree is None:
            self._build_tree()

        easting, northing = self.latlon_to_xy(lat, lon)
        point = np.array([[easting, northing]])
        dist, idx = self._tree.query(point, k=k)

        return idx[0], dist[0]

    def query_radius(
        self,
        lat: float,
        lon: float,
        radius_m: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find all grid cells within radius of a point.

        Args:
            lat: Query latitude
            lon: Query longitude
            radius_m: Search radius in meters

        Returns:
            (indices, distances_m)
        """
        if self._tree is None:
            self._build_tree()

        easting, northing = self.latlon_to_xy(lat, lon)
        point = np.array([[easting, northing]])

        idx, dist = self._tree.query_radius(point, r=radius_m, return_distance=True)

        return idx[0], dist[0]

    def create_coords_dataset(self) -> dict:
        """
        Create coordinate arrays for zarr store initialization.

        Returns:
            Dict with 'y' (1D), 'x' (1D), 'lat' (2D), 'lon' (2D)
        """
        return {
            "y": self.y.astype(np.float64),
            "x": self.x.astype(np.float64),
            "lat": self.lat.astype(np.float64),
            "lon": self.lon.astype(np.float64),
        }

    def to_dict(self) -> dict:
        """Serialize grid parameters to dict."""
        return {
            "bounds": list(self.bounds),
            "resolution": self.resolution,
            "crs": self.crs,
            "shape": list(self.shape),
            "n_cells": self.n_cells,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MasterGrid":
        """Create MasterGrid from dict."""
        return cls(
            bounds=tuple(d["bounds"]),
            resolution=d["resolution"],
            crs=d.get("crs", "EPSG:5070"),
        )

    def __repr__(self) -> str:
        return (
            f"MasterGrid(bounds={self.bounds}, "
            f"resolution={self.resolution:.0f}m, "
            f"shape={self.shape}, "
            f"n_cells={self.n_cells:,})"
        )


def create_conus_grid() -> MasterGrid:
    """Create default CONUS grid at 1km resolution in EPSG:5070."""
    bounds = _geographic_bounds_to_albers(-125.0, 24.0, -66.0, 50.0)
    return MasterGrid(bounds=bounds, resolution=1000.0, crs="EPSG:5070")


def create_test_region_grid() -> MasterGrid:
    """Create Pacific NW test region (WA/OR/ID/MT) at 1km in EPSG:5070."""
    bounds = _geographic_bounds_to_albers(-125.0, 42.0, -104.0, 49.0)
    return MasterGrid(bounds=bounds, resolution=1000.0, crs="EPSG:5070")


if __name__ == "__main__":
    grid = create_conus_grid()
    print(grid)
    print("\nCoordinate arrays:")
    print(f"  y: [{grid.y[0]:.0f}, ..., {grid.y[-1]:.0f}] ({len(grid.y)} values)")
    print(f"  x: [{grid.x[0]:.0f}, ..., {grid.x[-1]:.0f}] ({len(grid.x)} values)")

    # Test coordinate conversion
    test_lat, test_lon = 40.0, -105.0
    row, col = grid.latlon_to_rowcol(test_lat, test_lon)
    lat_back, lon_back = grid.rowcol_to_latlon(row, col)
    print("\nCoordinate conversion test:")
    print(f"  Input: ({test_lat}, {test_lon})")
    print(f"  Row/Col: ({row}, {col})")
    print(f"  Recovered: ({lat_back:.4f}, {lon_back:.4f})")
