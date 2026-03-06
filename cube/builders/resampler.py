"""
Resampling utilities for aligning multi-resolution sources to master grid.

Handles reprojection and resampling from various source formats (GeoTIFF,
NetCDF, zarr) to the unified 1km EPSG:5070 grid.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import (
        Resampling,
        reproject,
    )

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import rioxarray as rxr  # noqa: F401  (side-effect: registers .rio accessor)

    HAS_RIOXARRAY = True
except ImportError:
    HAS_RIOXARRAY = False

from cube.grid import MasterGrid

logger = logging.getLogger(__name__)


# Resampling method mapping
RESAMPLING_METHODS = {
    # Aggregation for fine->coarse (DEM, Landsat)
    "average": Resampling.average if HAS_RASTERIO else None,
    "mean": Resampling.average if HAS_RASTERIO else None,
    # Interpolation for coarse->fine or similar resolution
    "bilinear": Resampling.bilinear if HAS_RASTERIO else None,
    "cubic": Resampling.cubic if HAS_RASTERIO else None,
    # Nearest neighbor (categorical data, masks)
    "nearest": Resampling.nearest if HAS_RASTERIO else None,
    # Mode for categorical (land cover, masks)
    "mode": Resampling.mode if HAS_RASTERIO else None,
}


class GridResampler:
    """
    Resamples raster data to the master grid.

    Handles various source formats and resolutions, projecting everything
    to the unified 1km EPSG:5070 grid.

    Typical source resolutions:
        - DEM: 30-90m -> 1km (aggregation with mean)
        - CDR: ~5km -> 1km (bilinear interpolation)
        - NLDAS: ~12km -> 1km (bilinear)
        - PRISM: 4km -> 1km (bilinear)
        - GridMET: 4km -> 1km (bilinear)
        - Landsat: 30m -> 1km (aggregation)
    """

    def __init__(self, grid: MasterGrid):
        """
        Initialize resampler with target grid.

        Args:
            grid: MasterGrid defining target coordinate system
        """
        if not HAS_RASTERIO:
            raise ImportError("rasterio is required for GridResampler")

        self.grid = grid
        self.dst_crs = CRS.from_string(grid.crs)
        self.dst_transform = grid.transform
        self.dst_shape = grid.shape

    def resample_raster(
        self,
        src_path: Union[str, Path],
        method: str = "bilinear",
        band: int = 1,
    ) -> np.ndarray:
        """
        Resample a raster file to the master grid.

        Args:
            src_path: Path to source raster (GeoTIFF)
            method: Resampling method ('bilinear', 'average', 'nearest', 'mode')
            band: Band number to read (1-indexed)

        Returns:
            2D array resampled to master grid shape (n_y, n_x)
        """
        resampling = RESAMPLING_METHODS.get(method, Resampling.bilinear)

        with rasterio.open(src_path) as src:
            # Create output array
            dst_array = np.empty(self.dst_shape, dtype=src.dtypes[band - 1])

            reproject(
                source=rasterio.band(src, band),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=self.dst_transform,
                dst_crs=self.dst_crs,
                resampling=resampling,
            )

        return dst_array

    def resample_array(
        self,
        data: np.ndarray,
        src_transform,
        src_crs: str,
        method: str = "bilinear",
        nodata: Optional[float] = None,
    ) -> np.ndarray:
        """
        Resample an in-memory array to the master grid.

        Args:
            data: 2D source array
            src_transform: Affine transform of source data
            src_crs: Source CRS string (e.g., 'EPSG:4326')
            method: Resampling method
            nodata: NoData value to mask

        Returns:
            2D array resampled to master grid
        """
        resampling = RESAMPLING_METHODS.get(method, Resampling.bilinear)

        dst_array = np.empty(self.dst_shape, dtype=data.dtype)

        if nodata is not None:
            # Mask nodata before reprojection
            data = np.where(data == nodata, np.nan, data)

        reproject(
            source=data,
            destination=dst_array,
            src_transform=src_transform,
            src_crs=CRS.from_string(src_crs),
            dst_transform=self.dst_transform,
            dst_crs=self.dst_crs,
            resampling=resampling,
        )

        return dst_array

    def resample_xarray(
        self,
        da: "xr.DataArray",
        method: str = "bilinear",
    ) -> "xr.DataArray":
        """
        Resample an xarray DataArray to the master grid.

        Uses rioxarray for coordinate-aware resampling.

        Args:
            da: Source DataArray with spatial coordinates
            method: Resampling method

        Returns:
            DataArray resampled to master grid coordinates
        """
        if not HAS_RIOXARRAY:
            raise ImportError("rioxarray is required for xarray resampling")

        resampling = RESAMPLING_METHODS.get(method, Resampling.bilinear)

        # Ensure CRS is set
        if da.rio.crs is None:
            # Assume WGS84 if not specified
            da = da.rio.write_crs("EPSG:4326")

        # Reproject to target grid
        da_resampled = da.rio.reproject(
            self.dst_crs,
            shape=self.dst_shape,
            transform=self.dst_transform,
            resampling=resampling,
        )

        return da_resampled

    def resample_temporal_xarray(
        self,
        da: "xr.DataArray",
        method: str = "bilinear",
    ) -> np.ndarray:
        """
        Resample a temporal xarray DataArray, processing each timestep.

        For large time series, processes in chunks to manage memory.

        Args:
            da: DataArray with dims (time, lat, lon) or similar
            method: Resampling method

        Returns:
            3D array (time, lat, lon) resampled to master grid
        """
        if not HAS_XARRAY:
            raise ImportError("xarray is required")

        # Identify time dimension
        time_dim = None
        for dim in da.dims:
            if dim in ("time", "Time", "DATE"):
                time_dim = dim
                break

        if time_dim is None:
            # No time dimension - treat as spatial
            return self.resample_xarray(da, method).values

        n_times = da.sizes[time_dim]

        # Initialize output array
        output = np.empty((n_times, *self.dst_shape), dtype=np.float32)

        # Process each timestep
        for t in range(n_times):
            da_t = da.isel({time_dim: t})
            output[t] = self.resample_xarray(da_t, method).values

            if (t + 1) % 100 == 0:
                logger.info(f"Resampled {t + 1}/{n_times} timesteps")

        return output


def resample_to_grid(
    da: "xr.DataArray",
    grid: MasterGrid,
    method: str = "bilinear",
) -> "xr.DataArray":
    """
    Convenience function to resample a DataArray to the master grid.

    Args:
        da: Source DataArray
        grid: Target MasterGrid
        method: Resampling method

    Returns:
        Resampled DataArray
    """
    resampler = GridResampler(grid)
    return resampler.resample_xarray(da, method)


def compute_source_to_target_mapping(
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    grid: MasterGrid,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mapping from source grid cells to target grid cells.

    For each source cell, finds the nearest target cell.
    Useful for point-by-point data transfer without interpolation.

    Args:
        src_lat: 1D or 2D array of source latitudes
        src_lon: 1D or 2D array of source longitudes
        grid: Target MasterGrid

    Returns:
        (target_rows, target_cols) indices for each source cell
    """
    src_lat = np.atleast_1d(src_lat).ravel()
    src_lon = np.atleast_1d(src_lon).ravel()

    target_rows, target_cols = grid.latlon_to_rowcol(src_lat, src_lon)

    return target_rows, target_cols


def get_recommended_method(
    src_resolution_m: float,
    layer_type: str = "continuous",
    target_resolution_m: float = 1000.0,
) -> str:
    """
    Get recommended resampling method based on source resolution.

    Args:
        src_resolution_m: Source resolution in meters
        layer_type: 'continuous', 'categorical', or 'terrain'
        target_resolution_m: Target resolution in meters (default 1000)

    Returns:
        Recommended resampling method name
    """
    if layer_type == "categorical":
        return "mode"

    if src_resolution_m < target_resolution_m:
        # Aggregating (fine -> coarse)
        return "average"
    else:
        # Interpolating (coarse -> fine or similar)
        return "bilinear"


if __name__ == "__main__":
    # Demo
    from cube.grid import create_conus_grid

    grid = create_conus_grid()
    resampler = GridResampler(grid)

    print(f"Target grid shape: {resampler.dst_shape}")
    print(f"Target CRS: {resampler.dst_crs}")
    print(f"Target transform: {resampler.dst_transform}")
