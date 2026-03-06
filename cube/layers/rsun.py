"""
Clear-sky solar irradiance layer builder for the data cube.

Builds DOY-indexed clear-sky global horizontal irradiance (rsun) from
pre-computed r.sun output GeoTIFFs.

Input format: Directory with 365 GeoTIFFs, one per DOY, named like:
    - rsun_doy_001.tif, rsun_doy_002.tif, ... rsun_doy_365.tif
    - OR organized by tile subdirectories with irradiance_day_{doy}_{tile}.tif

Output: cube.zarr/doy_indexed/rsun with shape (365, n_y, n_x)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from cube.builders.resampler import GridResampler

import numpy as np

try:
    import rasterio
    from rasterio.merge import merge as rasterio_merge

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import zarr

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

from cube.config import CHUNKS, CubeConfig
from cube.layers.base import BaseLayer

logger = logging.getLogger(__name__)


class RSUNLayer(BaseLayer):
    """
    Builds DOY-indexed clear-sky irradiance layer from r.sun output.

    Expected input formats:
        1. Single directory with files: rsun_doy_{doy:03d}.tif or rsun_{doy}.tif
        2. Tile-organized: {rsun_dir}/{tile}/irradiance_day_{doy}_{tile}.tif

    Output: (365, y, x) array of daily total clear-sky irradiance (Wh/m²/day)
    """

    @property
    def name(self) -> str:
        return "doy_indexed"

    @property
    def variables(self) -> List[str]:
        return ["rsun"]

    @property
    def dimensions(self) -> Tuple[str, ...]:
        return ("doy", "y", "x")

    @property
    def chunks(self) -> Dict[str, int]:
        return CHUNKS["doy_indexed"]

    def build(
        self,
        source_paths: Optional[Dict[str, Path]] = None,
        overwrite: bool = False,
        merge_tiles: bool = True,
    ) -> None:
        """
        Build rsun layer from GeoTIFF files.

        Args:
            source_paths: Dict with 'rsun_dir' key pointing to rsun directory
            overwrite: Whether to overwrite existing data
            merge_tiles: Whether to merge tile-organized files (slower but handles tiled input)
        """
        if not HAS_RASTERIO:
            raise ImportError("rasterio required for rsun layer building")

        source_paths = source_paths or self.config.source_paths
        rsun_dir = source_paths.get("rsun_dir")

        if rsun_dir is None:
            raise ValueError("rsun_dir path not provided in source_paths['rsun_dir']")

        rsun_dir = Path(rsun_dir)
        if not rsun_dir.exists():
            raise FileNotFoundError(f"rsun directory not found: {rsun_dir}")

        logger.info(f"Building rsun layer from {rsun_dir}")

        # Detect input organization
        input_format = self._detect_input_format(rsun_dir)
        logger.info(f"Detected input format: {input_format}")

        # Open store and create group
        store = self._open_store("a")
        self._write_coords(store)
        self._write_doy_coord(store)
        group = self._ensure_group(store)

        # Create output array
        shape = (365, self.grid.n_y, self.grid.n_x)
        chunks = (self.chunks["doy"], self.chunks["y"], self.chunks["x"])

        if "rsun" in group and not overwrite:
            logger.info(
                "rsun array already exists, skipping (use overwrite=True to rebuild)"
            )
            return

        rsun_array = group.create_dataset(
            "rsun",
            shape=shape,
            chunks=chunks,
            dtype="float32",
            compressors=self.compression,
            fill_value=np.nan,
            overwrite=overwrite,
        )

        # Process each DOY
        from cube.builders.resampler import GridResampler

        resampler = GridResampler(self.grid)

        for doy in range(1, 366):
            if doy % 30 == 0 or doy == 1:
                logger.info(f"Processing DOY {doy}/365...")

            try:
                if input_format == "flat":
                    data = self._load_flat_doy(rsun_dir, doy, resampler)
                elif input_format == "tiled":
                    data = self._load_tiled_doy(rsun_dir, doy, resampler, merge_tiles)
                else:
                    raise ValueError(f"Unknown input format: {input_format}")

                # Store in zarr (0-indexed)
                rsun_array[doy - 1, :, :] = data.astype(np.float32)

            except FileNotFoundError as e:
                logger.warning(f"DOY {doy} not found: {e}")
                # Leave as NaN fill value
            except Exception as e:
                logger.error(f"Error processing DOY {doy}: {e}")
                raise

        # Store metadata
        store.attrs["rsun_source"] = str(rsun_dir)
        store.attrs["rsun_units"] = "Wh/m2/day"
        store.attrs["rsun_description"] = "Clear-sky global horizontal irradiance"

        logger.info("rsun layer complete")

    def _detect_input_format(self, rsun_dir: Path) -> str:
        """
        Detect input file organization.

        Returns:
            'flat': Single directory with rsun_doy_*.tif files
            'tiled': Subdirectories per tile with irradiance_day_*.tif
        """
        # Check for flat organization
        flat_files = list(rsun_dir.glob("rsun_doy_*.tif")) + list(
            rsun_dir.glob("rsun_*.tif")
        )
        if flat_files:
            return "flat"

        # Check for tile subdirectories
        subdirs = [d for d in rsun_dir.iterdir() if d.is_dir()]
        if subdirs:
            # Look for irradiance files in subdirs
            for subdir in subdirs[:3]:  # Check first few
                tile_files = list(subdir.glob("irradiance_day_*.tif"))
                if tile_files:
                    return "tiled"

        raise ValueError(
            f"Could not detect rsun input format in {rsun_dir}. "
            "Expected either rsun_doy_*.tif files or tile subdirectories with irradiance_day_*.tif"
        )

    def _load_flat_doy(
        self,
        rsun_dir: Path,
        doy: int,
        resampler: "GridResampler",
    ) -> np.ndarray:
        """Load a single DOY from flat file organization."""
        # Try different naming conventions
        patterns = [
            f"rsun_doy_{doy:03d}.tif",
            f"rsun_{doy}.tif",
            f"rsun_{doy:03d}.tif",
            f"irradiance_day_{doy}.tif",
        ]

        for pattern in patterns:
            filepath = rsun_dir / pattern
            if filepath.exists():
                return resampler.resample_raster(filepath, method="bilinear")

        raise FileNotFoundError(f"No rsun file found for DOY {doy} in {rsun_dir}")

    def _load_tiled_doy(
        self,
        rsun_dir: Path,
        doy: int,
        resampler: "GridResampler",
        merge_tiles: bool = True,
    ) -> np.ndarray:
        """Load a single DOY from tile-organized structure."""
        # Collect all tile files for this DOY
        tile_files = []
        for tile_dir in rsun_dir.iterdir():
            if not tile_dir.is_dir():
                continue

            tile_id = tile_dir.name
            # Match irradiance_day_{doy}_{tile}.tif pattern
            pattern = f"irradiance_day_{doy}_{tile_id}.tif"
            filepath = tile_dir / pattern

            if filepath.exists():
                tile_files.append(filepath)

        if not tile_files:
            raise FileNotFoundError(f"No tile files found for DOY {doy}")

        if len(tile_files) == 1:
            # Single tile - just resample
            return resampler.resample_raster(tile_files[0], method="bilinear")

        if merge_tiles:
            # Merge tiles then resample
            return self._merge_and_resample(tile_files, resampler)
        else:
            # Resample each tile and combine (faster but may have edge effects)
            return self._resample_and_combine(tile_files, resampler)

    def _merge_and_resample(
        self,
        tile_files: List[Path],
        resampler: "GridResampler",
    ) -> np.ndarray:
        """Merge tiles using rasterio.merge then resample to grid."""

        # Open all tiles
        src_files = [rasterio.open(f) for f in tile_files]

        try:
            # Merge tiles
            mosaic, out_transform = rasterio_merge(src_files)

            # Get CRS from first file
            src_crs = src_files[0].crs.to_string()

            # Resample merged mosaic
            data = resampler.resample_array(
                mosaic[0],  # First band
                src_transform=out_transform,
                src_crs=src_crs,
                method="bilinear",
            )

        finally:
            for src in src_files:
                src.close()

        return data

    def _resample_and_combine(
        self,
        tile_files: List[Path],
        resampler: "GridResampler",
    ) -> np.ndarray:
        """Resample each tile independently and combine using nanmean."""
        # Initialize accumulator
        combined = np.full((self.grid.n_y, self.grid.n_x), np.nan, dtype=np.float32)
        count = np.zeros((self.grid.n_y, self.grid.n_x), dtype=np.int32)

        for filepath in tile_files:
            try:
                tile_data = resampler.resample_raster(filepath, method="bilinear")

                # Accumulate valid values
                valid = ~np.isnan(tile_data)
                combined = np.where(valid & (count == 0), tile_data, combined)
                combined = np.where(
                    valid & (count > 0),
                    (combined * count + tile_data) / (count + 1),
                    combined,
                )
                count[valid] += 1

            except Exception as e:
                logger.warning(f"Error resampling {filepath}: {e}")

        return combined

    def validate(self) -> Dict[str, bool]:
        """Validate rsun layer with physical checks."""
        checks = super().validate()

        if not all(checks.values()):
            return checks

        try:
            store = zarr.open(str(self.store_path), mode="r")
            group = store[self.name]

            rsun = group["rsun"][:]

            # Physical range checks (Wh/m²/day)
            # Typical range: 0 to ~10000 Wh/m²/day for clear sky
            valid = rsun[~np.isnan(rsun)]
            checks["rsun_min"] = valid.min() >= 0
            checks["rsun_max"] = valid.max() < 15000  # Upper bound

            # Check seasonal variation (summer DOYs should have higher values at mid-latitudes)
            summer_mean = np.nanmean(rsun[150:220, :, :])  # ~June-August
            winter_mean = np.nanmean(rsun[330:365, :, :])  # ~December
            checks["seasonal_pattern"] = summer_mean > winter_mean

            # Coverage check
            nan_fraction = np.isnan(rsun).mean()
            checks["coverage"] = nan_fraction < 0.5  # At least 50% coverage

        except Exception as e:
            logger.error(f"Validation error: {e}")
            checks["validation_error"] = False

        return checks

    def get_rsun_for_doy(self, doy: int) -> np.ndarray:
        """
        Load rsun values for a specific DOY.

        Args:
            doy: Day of year (1-365)

        Returns:
            2D array (y, x) of rsun values
        """
        if not self.store_path.exists():
            raise FileNotFoundError(f"Cube not found: {self.store_path}")

        store = zarr.open(str(self.store_path), mode="r")
        return store[self.name]["rsun"][doy - 1, :, :]


def build_rsun_from_grass_export(
    grass_export_dir: Path,
    config: CubeConfig,
    overwrite: bool = False,
) -> None:
    """
    Convenience function to build rsun layer from GRASS r.sun export.

    Expected structure:
        grass_export_dir/
            {tile_id}/
                irradiance_day_1_{tile_id}.tif
                irradiance_day_2_{tile_id}.tif
                ...
                irradiance_day_365_{tile_id}.tif

    Args:
        grass_export_dir: Directory with GRASS r.sun exports organized by tile
        config: CubeConfig instance
        overwrite: Whether to overwrite existing data
    """
    # Update config with rsun_dir
    config.source_paths["rsun_dir"] = grass_export_dir

    layer = RSUNLayer(config)
    layer.build(overwrite=overwrite, merge_tiles=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        rsun_dir = sys.argv[1]
        cube_path = sys.argv[2]
    else:
        rsun_dir = "/nas/dads/dem/rsun_irradiance"
        cube_path = "/data/ssd2/dads_cube/cube.zarr"

    if Path(rsun_dir).exists():
        from cube.config import default_conus_config

        config = default_conus_config(
            cube_path=cube_path,
            rsun_dir=rsun_dir,
        )

        layer = RSUNLayer(config)
        print(layer)
        print(f"Expected shape: {layer._get_expected_shape()}")

        # Build if cube path exists
        if Path(cube_path).parent.exists():
            print("\nBuilding rsun layer...")
            layer.build(overwrite=False)
            print("\nValidating...")
            checks = layer.validate()
            for check, passed in checks.items():
                status = "✓" if passed else "✗"
                print(f"  {status} {check}")
    else:
        print(f"rsun directory not found: {rsun_dir}")
