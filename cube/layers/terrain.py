"""
Terrain layer builder for the data cube.

Builds static terrain features from DEM:
    - elevation
    - slope
    - aspect
    - TPI at multiple scales (500m, 2500m, 10000m, 22500m)
    - land_mask

These features match TERRAIN_FEATURES from prep/columns_desc.py.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import numpy as np

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from scipy.ndimage import uniform_filter, generic_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

from cube.layers.base import BaseLayer
from cube.grid import MasterGrid
from cube.config import CubeConfig, CHUNKS, TERRAIN_FEATURES

logger = logging.getLogger(__name__)


class TerrainLayer(BaseLayer):
    """
    Builds static terrain features from DEM.

    Features computed:
        - elevation: Direct from DEM (meters)
        - slope: Computed via Horn's method (degrees)
        - aspect: Computed via Horn's method (degrees from north)
        - tpi_500: Topographic Position Index at 500m radius
        - tpi_2500: TPI at 2500m radius
        - tpi_10000: TPI at 10km radius
        - tpi_22500: TPI at 22.5km radius
        - land_mask: Binary mask (1=valid land, 0=water/nodata)

    TPI radii in pixels (at 1km resolution):
        500m -> not computed at 1km (would be < 1 pixel)
        2500m -> ~3 pixels radius
        10000m -> ~10 pixels radius
        22500m -> ~23 pixels radius

    Note: tpi_500 at 1km grid is approximated by tpi at ~1 pixel radius.
    """

    # TPI radii in km -> pixels at 1km resolution
    TPI_RADII_KM = {
        'tpi_500': 0.5,     # ~0.5 pixel, use 1
        'tpi_2500': 2.5,    # ~3 pixels
        'tpi_10000': 10.0,  # ~10 pixels
        'tpi_22500': 22.5,  # ~23 pixels
    }

    @property
    def name(self) -> str:
        return 'static'

    @property
    def variables(self) -> List[str]:
        return ['elevation', 'slope', 'aspect',
                'tpi_500', 'tpi_2500', 'tpi_10000', 'tpi_22500',
                'land_mask']

    @property
    def dimensions(self) -> Tuple[str, ...]:
        return ('lat', 'lon')

    @property
    def chunks(self) -> Dict[str, int]:
        return CHUNKS['static']

    def build(
        self,
        source_paths: Optional[Dict[str, Path]] = None,
        overwrite: bool = False,
        compute_tpi: bool = True,
    ) -> None:
        """
        Build terrain layer from DEM.

        Args:
            source_paths: Dict with 'dem' key pointing to DEM GeoTIFF
            overwrite: Whether to overwrite existing data
            compute_tpi: Whether to compute TPI features (slower)
        """
        if not HAS_RASTERIO:
            raise ImportError("rasterio required for terrain building")

        source_paths = source_paths or self.config.source_paths
        dem_path = source_paths.get('dem')

        if dem_path is None:
            raise ValueError("DEM path not provided in source_paths['dem']")

        dem_path = Path(dem_path)
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM not found: {dem_path}")

        logger.info(f"Building terrain layer from {dem_path}")

        # Open store and create group
        store = self._open_store('a')
        self._write_coords(store)
        group = self._ensure_group(store)

        # Read and resample DEM
        logger.info("Resampling DEM to master grid...")
        elevation, cell_size = self._resample_dem(dem_path)

        # Create land mask from valid (non-nan) elevation
        land_mask = (~np.isnan(elevation)).astype(np.uint8)

        # Write elevation and land_mask
        logger.info("Writing elevation...")
        self._write_array(group, 'elevation', elevation.astype(np.float32), overwrite)

        logger.info("Writing land_mask...")
        group.create_dataset(
            'land_mask',
            data=land_mask,
            chunks=tuple(self.chunks.values()),
            dtype='uint8',
            overwrite=overwrite,
        )

        # Compute slope and aspect
        logger.info("Computing slope and aspect...")
        slope, aspect = self._compute_slope_aspect(elevation, cell_size)
        self._write_array(group, 'slope', slope.astype(np.float32), overwrite)
        self._write_array(group, 'aspect', aspect.astype(np.float32), overwrite)

        # Compute TPI at multiple scales
        if compute_tpi and HAS_SCIPY:
            for tpi_name, radius_km in self.TPI_RADII_KM.items():
                # Convert km to pixels (at ~1km resolution)
                radius_pixels = max(1, int(round(radius_km)))
                logger.info(f"Computing {tpi_name} (radius={radius_pixels} pixels)...")
                tpi = self._compute_tpi(elevation, radius_pixels)
                self._write_array(group, tpi_name, tpi.astype(np.float32), overwrite)
        else:
            # Write zeros if TPI not computed
            logger.warning("Skipping TPI computation (scipy not available or disabled)")
            for tpi_name in self.TPI_RADII_KM.keys():
                zeros = np.zeros(self.grid.shape, dtype=np.float32)
                self._write_array(group, tpi_name, zeros, overwrite)

        # Store metadata
        store.attrs['terrain_dem_source'] = str(dem_path)
        store.attrs['terrain_cell_size_m'] = float(cell_size)

        logger.info("Terrain layer complete")

    def _resample_dem(self, dem_path: Path) -> Tuple[np.ndarray, float]:
        """
        Resample DEM to master grid.

        Args:
            dem_path: Path to DEM GeoTIFF

        Returns:
            (elevation_array, cell_size_meters)
        """
        from cube.builders.resampler import GridResampler

        resampler = GridResampler(self.grid)

        with rasterio.open(dem_path) as src:
            # Get approximate cell size in meters for slope/aspect calculation
            # (at grid center latitude)
            center_lat = (self.grid.north + self.grid.south) / 2
            # degrees to meters (approximate)
            cell_size_m = self.grid.resolution_deg * 111320 * np.cos(np.radians(center_lat))

            # Read source nodata
            nodata = src.nodata

        # Resample using average (aggregation from high-res DEM)
        elevation = resampler.resample_raster(dem_path, method='average')

        # Handle nodata
        if nodata is not None:
            elevation = np.where(elevation == nodata, np.nan, elevation)

        return elevation, cell_size_m

    def _compute_slope_aspect(
        self,
        elevation: np.ndarray,
        cell_size: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute slope and aspect using Horn's method (3x3 kernel).

        Args:
            elevation: 2D elevation array
            cell_size: Cell size in meters

        Returns:
            (slope_degrees, aspect_degrees)
        """
        # Pad edges for boundary handling
        padded = np.pad(elevation, 1, mode='edge')

        # Extract 3x3 neighborhood values
        # z1 z2 z3
        # z4 z5 z6
        # z7 z8 z9
        z1 = padded[:-2, :-2]
        z2 = padded[:-2, 1:-1]
        z3 = padded[:-2, 2:]
        z4 = padded[1:-1, :-2]
        # z5 = padded[1:-1, 1:-1]  # center (not used)
        z6 = padded[1:-1, 2:]
        z7 = padded[2:, :-2]
        z8 = padded[2:, 1:-1]
        z9 = padded[2:, 2:]

        # Horn's method for gradient
        dzdx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8 * cell_size)
        dzdy = ((z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)) / (8 * cell_size)

        # Slope in degrees
        slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

        # Aspect in degrees (0-360, 0=north, clockwise)
        aspect = np.degrees(np.arctan2(-dzdx, dzdy))
        aspect = np.where(aspect < 0, aspect + 360, aspect)

        # Handle flat areas (undefined aspect)
        flat_mask = (dzdx == 0) & (dzdy == 0)
        aspect = np.where(flat_mask, -1, aspect)  # -1 indicates flat

        # Propagate NaN from elevation
        nan_mask = np.isnan(elevation)
        slope = np.where(nan_mask, np.nan, slope)
        aspect = np.where(nan_mask, np.nan, aspect)

        return slope, aspect

    def _compute_tpi(
        self,
        elevation: np.ndarray,
        radius_pixels: int,
    ) -> np.ndarray:
        """
        Compute Topographic Position Index.

        TPI = elevation - mean(neighborhood elevation)

        Positive values indicate ridges/peaks
        Negative values indicate valleys/depressions
        Near-zero indicates slopes or flat areas

        Args:
            elevation: 2D elevation array
            radius_pixels: Neighborhood radius in pixels

        Returns:
            TPI array
        """
        if not HAS_SCIPY:
            raise ImportError("scipy required for TPI computation")

        # Diameter = 2*radius + 1
        window_size = 2 * radius_pixels + 1

        # Compute mean of neighborhood using uniform filter
        # This is much faster than generic_filter for simple mean
        with np.errstate(invalid='ignore'):
            # Create mask for valid values
            valid_mask = ~np.isnan(elevation)

            # Replace NaN with 0 for filtering, then adjust
            elev_filled = np.where(valid_mask, elevation, 0)

            # Sum of values in window
            window_sum = uniform_filter(
                elev_filled.astype(np.float64),
                size=window_size,
                mode='constant',
                cval=0.0
            )

            # Count of valid values in window
            valid_count = uniform_filter(
                valid_mask.astype(np.float64),
                size=window_size,
                mode='constant',
                cval=0.0
            )

            # Mean = sum / count (avoid division by zero)
            neighborhood_mean = np.where(
                valid_count > 0,
                window_sum / valid_count,
                np.nan
            )

        # TPI = elevation - neighborhood mean
        tpi = elevation - neighborhood_mean

        return tpi.astype(np.float32)

    def validate(self) -> Dict[str, bool]:
        """Validate terrain layer with additional physical checks."""
        checks = super().validate()

        if not all(checks.values()):
            return checks

        try:
            store = zarr.open(str(self.store_path), mode='r')
            group = store[self.name]

            # Physical range checks
            elev = group['elevation'][:]
            valid_elev = elev[~np.isnan(elev)]
            checks['elevation_range'] = (
                valid_elev.min() > -500 and valid_elev.max() < 9000
            )

            slope = group['slope'][:]
            valid_slope = slope[~np.isnan(slope)]
            checks['slope_range'] = (
                valid_slope.min() >= 0 and valid_slope.max() <= 90
            )

            aspect = group['aspect'][:]
            valid_aspect = aspect[(~np.isnan(aspect)) & (aspect >= 0)]
            checks['aspect_range'] = (
                valid_aspect.min() >= 0 and valid_aspect.max() <= 360
            )

            # Land mask coverage
            land_mask = group['land_mask'][:]
            land_fraction = land_mask.mean()
            checks['land_coverage'] = 0.3 < land_fraction < 0.9  # Reasonable for CONUS

        except Exception as e:
            logger.error(f"Validation error: {e}")
            checks['validation_error'] = False

        return checks


if __name__ == '__main__':
    # Example usage
    from cube.config import default_conus_config

    config = default_conus_config(
        dem_path='/data/dem/conus_dem.tif'
    )

    layer = TerrainLayer(config)
    print(layer)
    print(f"Expected shape: {layer._get_expected_shape()}")
