"""
Extract terrain features at grid cell centroids.

Uses rasterio to sample from DEM and compute derived products:
    - elevation
    - slope
    - aspect
    - TPI (topographic position index) at multiple scales

These become the pseudo-embeddings for grid cells (replacing station
autoencoder embeddings in the observation system).
"""

import numpy as np
from typing import Tuple, Optional

try:
    import rasterio
    from rasterio.windows import Window  # noqa: F401

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from scipy.ndimage import generic_filter

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Default TPI radii matching TERRAIN_FEATURES in columns_desc
DEFAULT_TPI_RADII = (
    5,
    11,
    25,
    50,
)  # Corresponds to ~500m, ~1km, ~2.5km, ~10km at 100m res


def extract_terrain_at_points(
    dem_path: str,
    lats: np.ndarray,
    lons: np.ndarray,
    tpi_radii: Tuple[int, ...] = DEFAULT_TPI_RADII,
    include_tpi: bool = True,
) -> np.ndarray:
    """
    Extract terrain features at given lat/lon points.

    Args:
        dem_path: Path to DEM raster file (GeoTIFF)
        lats: Array of latitudes
        lons: Array of longitudes
        tpi_radii: Radii in pixels for TPI computation
        include_tpi: Whether to compute TPI features (slower)

    Returns:
        (N, D) array where D = 3 + len(tpi_radii) if include_tpi else 3
        Columns: [elevation, slope, aspect, *tpi_scales]
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for terrain extraction")

    lats = np.asarray(lats)
    lons = np.asarray(lons)
    n_points = len(lats)

    with rasterio.open(dem_path) as src:
        transform = src.transform
        dem_data = src.read(1)
        nodata = src.nodata
        cell_size = abs(transform[0])  # Assume square pixels

        # Get pixel coordinates for each point
        rows, cols = rasterio.transform.rowcol(transform, lons, lats)
        rows, cols = np.array(rows), np.array(cols)

        # Clamp to valid range
        rows = np.clip(rows, 0, dem_data.shape[0] - 1)
        cols = np.clip(cols, 0, dem_data.shape[1] - 1)

        # Read elevation
        elevation = dem_data[rows, cols].astype(np.float32)
        if nodata is not None:
            elevation[elevation == nodata] = np.nan

        # Compute slope and aspect from local 3x3 window
        slope = np.zeros(n_points, dtype=np.float32)
        aspect = np.zeros(n_points, dtype=np.float32)

        for i, (r, c) in enumerate(zip(rows, cols)):
            if (
                r < 1
                or r >= dem_data.shape[0] - 1
                or c < 1
                or c >= dem_data.shape[1] - 1
            ):
                continue
            patch = dem_data[r - 1 : r + 2, c - 1 : c + 2].astype(np.float64)
            if nodata is not None:
                patch[patch == nodata] = np.nan
            if np.isnan(patch).any():
                continue
            s, a = _compute_slope_aspect(patch, cell_size)
            slope[i] = s
            aspect[i] = a

        result_cols = [elevation, slope, aspect]

        # Compute TPI at multiple scales
        if include_tpi and HAS_SCIPY:
            for radius in tpi_radii:
                tpi = _compute_tpi_at_points(dem_data, rows, cols, radius, nodata)
                result_cols.append(tpi)

    return np.column_stack(result_cols).astype(np.float32)


def _compute_slope_aspect(patch: np.ndarray, cell_size: float) -> Tuple[float, float]:
    """
    Compute slope (degrees) and aspect (degrees) from 3x3 patch.

    Uses Horn's method for gradient estimation.

    Args:
        patch: 3x3 array of elevation values
        cell_size: Size of grid cell in map units

    Returns:
        (slope_degrees, aspect_degrees)
    """
    # Horn's method for gradient
    dzdx = (
        (patch[0, 2] + 2 * patch[1, 2] + patch[2, 2])
        - (patch[0, 0] + 2 * patch[1, 0] + patch[2, 0])
    ) / (8 * cell_size)

    dzdy = (
        (patch[2, 0] + 2 * patch[2, 1] + patch[2, 2])
        - (patch[0, 0] + 2 * patch[0, 1] + patch[0, 2])
    ) / (8 * cell_size)

    # Slope in degrees
    slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

    # Aspect in degrees (0-360, with 0/360 = north)
    aspect = np.degrees(np.arctan2(-dzdx, dzdy))
    if aspect < 0:
        aspect += 360

    return float(slope), float(aspect)


def _compute_tpi_at_points(
    dem: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    radius: int,
    nodata: Optional[float] = None,
) -> np.ndarray:
    """
    Compute Topographic Position Index at given points.

    TPI = elevation - mean(neighborhood elevation)

    Args:
        dem: 2D DEM array
        rows: Row indices of points
        cols: Column indices of points
        radius: Radius in pixels for neighborhood
        nodata: NoData value to exclude

    Returns:
        TPI values at each point
    """
    n_points = len(rows)
    tpi = np.zeros(n_points, dtype=np.float32)

    for i, (r, c) in enumerate(zip(rows, cols)):
        r_min = max(0, r - radius)
        r_max = min(dem.shape[0], r + radius + 1)
        c_min = max(0, c - radius)
        c_max = min(dem.shape[1], c + radius + 1)

        neighborhood = dem[r_min:r_max, c_min:c_max].astype(np.float64)
        center_val = dem[r, c]

        if nodata is not None:
            mask = neighborhood != nodata
            if mask.sum() == 0 or center_val == nodata:
                tpi[i] = np.nan
                continue
            mean_val = neighborhood[mask].mean()
        else:
            mean_val = neighborhood.mean()

        tpi[i] = center_val - mean_val

    return tpi


def compute_terrain_grid(
    dem_path: str,
    output_path: str,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    tpi_radii: Tuple[int, ...] = DEFAULT_TPI_RADII,
) -> None:
    """
    Pre-compute terrain features for an entire grid and save to Zarr.

    This is useful for creating a terrain cache that can be quickly
    indexed during training.

    Args:
        dem_path: Path to DEM raster
        output_path: Path for output Zarr store
        bounds: Optional (west, south, east, north) to subset
        tpi_radii: Radii for TPI computation
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for terrain computation")
    if not HAS_SCIPY:
        raise ImportError("scipy is required for TPI computation")

    import zarr

    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        nodata = src.nodata
        transform = src.transform
        cell_size = abs(transform[0])

        # Compute lat/lon for each cell
        rows, cols = np.meshgrid(
            np.arange(dem_data.shape[0]), np.arange(dem_data.shape[1]), indexing="ij"
        )
        lons, lats = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
        lons = np.array(lons).reshape(dem_data.shape)
        lats = np.array(lats).reshape(dem_data.shape)

        # Apply bounds if specified
        if bounds:
            w, s, e, n = bounds
            mask = (lats >= s) & (lats <= n) & (lons >= w) & (lons <= e)
        else:
            mask = np.ones_like(dem_data, dtype=bool)

        # Compute slope and aspect
        dem_float = dem_data.astype(np.float64)
        if nodata is not None:
            dem_float[dem_float == nodata] = np.nan

        slope = np.zeros_like(dem_float)
        aspect = np.zeros_like(dem_float)

        for i in range(1, dem_data.shape[0] - 1):
            for j in range(1, dem_data.shape[1] - 1):
                patch = dem_float[i - 1 : i + 2, j - 1 : j + 2]
                if np.isnan(patch).any():
                    continue
                s, a = _compute_slope_aspect(patch, cell_size)
                slope[i, j] = s
                aspect[i, j] = a

        # Compute TPI at each scale
        tpi_arrays = []
        for radius in tpi_radii:

            def tpi_func(values):
                center = values[len(values) // 2]
                valid = values[~np.isnan(values)]
                if len(valid) == 0:
                    return np.nan
                return center - np.nanmean(valid)

            tpi = generic_filter(
                dem_float, tpi_func, size=2 * radius + 1, mode="constant", cval=np.nan
            )
            tpi_arrays.append(tpi)

        # Save to Zarr
        store = zarr.open(output_path, mode="w")
        store.create_dataset("lat", data=lats, chunks=(512, 512))
        store.create_dataset("lon", data=lons, chunks=(512, 512))
        store.create_dataset("elevation", data=dem_float, chunks=(512, 512))
        store.create_dataset("slope", data=slope, chunks=(512, 512))
        store.create_dataset("aspect", data=aspect, chunks=(512, 512))
        for i, (radius, tpi) in enumerate(zip(tpi_radii, tpi_arrays)):
            store.create_dataset(f"tpi_{radius}", data=tpi, chunks=(512, 512))
        store.create_dataset("valid_mask", data=mask, chunks=(512, 512))


class TerrainCache:
    """
    Cache for quick terrain feature lookup at arbitrary lat/lon.

    Uses pre-computed terrain grid stored in Zarr for fast indexed access.
    """

    def __init__(self, zarr_path: str):
        """
        Load terrain cache from Zarr store.

        Args:
            zarr_path: Path to Zarr store created by compute_terrain_grid
        """
        import zarr

        self.store = zarr.open(zarr_path, mode="r")
        self.lat = self.store["lat"][:]
        self.lon = self.store["lon"][:]
        self.elevation = self.store["elevation"][:]
        self.slope = self.store["slope"][:]
        self.aspect = self.store["aspect"][:]

        # Load TPI arrays
        self.tpi_arrays = []
        self.tpi_names = []
        for key in sorted(self.store.keys()):
            if key.startswith("tpi_"):
                self.tpi_arrays.append(self.store[key][:])
                self.tpi_names.append(key)

        # Build KDTree for fast nearest-neighbor lookup
        from scipy.spatial import cKDTree

        points = np.column_stack([self.lat.ravel(), self.lon.ravel()])
        valid = ~np.isnan(self.elevation.ravel())
        self.valid_indices = np.where(valid)[0]
        self.tree = cKDTree(points[valid])
        self.shape = self.lat.shape

    def get_features(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """
        Get terrain features at given coordinates.

        Args:
            lats: Array of latitudes
            lons: Array of longitudes

        Returns:
            (N, D) array of terrain features
        """
        lats = np.asarray(lats)
        lons = np.asarray(lons)
        points = np.column_stack([lats, lons])

        # Find nearest valid points
        _, indices = self.tree.query(points)
        flat_indices = self.valid_indices[indices]

        # Convert to 2D indices
        rows = flat_indices // self.shape[1]
        cols = flat_indices % self.shape[1]

        # Extract features
        features = [
            self.elevation[rows, cols],
            self.slope[rows, cols],
            self.aspect[rows, cols],
        ]
        for tpi in self.tpi_arrays:
            features.append(tpi[rows, cols])

        return np.column_stack(features).astype(np.float32)


if __name__ == "__main__":
    # Example: Extract terrain at sample points
    import sys

    if len(sys.argv) < 2:
        print("Usage: python terrain.py <dem_path> [output_zarr]")
        sys.exit(1)

    dem = sys.argv[1]

    # Sample points (example)
    sample_lats = np.array([40.0, 41.0, 42.0])
    sample_lons = np.array([-120.0, -119.0, -118.0])

    features = extract_terrain_at_points(dem, sample_lats, sample_lons)
    print(f"Extracted terrain features shape: {features.shape}")
    print(f"Features:\n{features}")
