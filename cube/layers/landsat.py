"""
Landsat composite layer builder for the data cube.

Builds seasonal Landsat composites (surface reflectance + thermal) from
pre-composited GeoTIFFs that are already pixel-aligned to the 1km EPSG:5070
MasterGrid.

Input format: GeoTIFFs named {year}_p{period}.tif with 7 bands:
    - B2–B7: surface reflectance (int32, scaled ×10000, zero=nodata)
    - B10: brightness temperature (int32, scaled ×100, zero=nodata)

Output: cube.zarr/composites/{landsat_b2..landsat_b10} with shape
    (n_composites, n_y, n_x) where n_composites = n_years × 5 periods.

Gap-fill: per-pixel seasonal mean across all years, applied per period.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import zarr

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

from cube.config import CHUNKS, LANDSAT_FEATURES, N_PERIODS_PER_YEAR, CubeConfig
from cube.layers.base import BaseLayer

logger = logging.getLogger(__name__)

# Source band names in the GeoTIFF (band order)
BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B10"]

# Descaling divisors: reflectance ×10000, thermal ×100
BAND_SCALE = [10000.0] * 6 + [100.0]

# Number of reflectance bands (for negative-clamping logic)
N_REFL_BANDS = 6


class LandsatLayer(BaseLayer):
    """
    Builds seasonal Landsat composite layer from pre-aligned GeoTIFFs.

    Input GeoTIFFs are already on the 1km EPSG:5070 grid — no resampling needed.
    Zero values are treated as nodata (mapped to NaN). Negative reflectance
    values are clamped to 0. After initial ingest, per-pixel seasonal mean
    gap-fill ensures continuous coverage.

    Output: (composite_time, y, x) arrays for each Landsat band.
    """

    @property
    def name(self) -> str:
        return "composites"

    @property
    def variables(self) -> List[str]:
        return list(LANDSAT_FEATURES)

    @property
    def dimensions(self) -> Tuple[str, ...]:
        return ("composite_time", "y", "x")

    @property
    def chunks(self) -> Dict[str, int]:
        return CHUNKS["composites"]

    def build(
        self,
        source_paths: Optional[Dict[str, Path]] = None,
        overwrite: bool = False,
        start_year: int = 1987,
        end_year: int = 2025,
        gap_fill: bool = True,
    ) -> None:
        """
        Build Landsat composite layer from GeoTIFF files.

        Args:
            source_paths: Dict with 'landsat_dir' key pointing to composites directory
            overwrite: Whether to overwrite existing data
            start_year: First year of composites (inclusive)
            end_year: Last year of composites (inclusive)
            gap_fill: Whether to apply per-pixel seasonal mean gap-fill
        """
        if not HAS_RASTERIO:
            raise ImportError("rasterio required for Landsat layer building")

        source_paths = source_paths or self.config.source_paths
        landsat_dir = source_paths.get("landsat_dir")

        if landsat_dir is None:
            raise ValueError("landsat_dir not provided in source_paths")

        landsat_dir = Path(landsat_dir)
        if not landsat_dir.exists():
            raise FileNotFoundError(f"Landsat directory not found: {landsat_dir}")

        n_years = end_year - start_year + 1
        n_composites = n_years * N_PERIODS_PER_YEAR

        logger.info(f"Building Landsat layer from {landsat_dir}")
        logger.info(
            f"Years: {start_year}–{end_year} ({n_years} years, {n_composites} composites)"
        )

        # Open store and write coordinates
        store = self._open_store("a")
        self._write_coords(store)
        self._write_composite_time_coord(store, start_year, end_year)
        group = self._ensure_group(store)

        # Create output arrays
        shape = (n_composites, self.grid.n_y, self.grid.n_x)
        chunks = (self.chunks["composite_time"], self.chunks["y"], self.chunks["x"])

        arrays = {}
        for var in self.variables:
            if var in group and not overwrite:
                logger.info(f"{var} already exists, skipping (use overwrite=True)")
                arrays[var] = group[var]
            else:
                arrays[var] = group.create_dataset(
                    var,
                    shape=shape,
                    chunks=chunks,
                    dtype="float32",
                    compressors=self.compression,
                    fill_value=np.nan,
                    overwrite=overwrite,
                )

        # Map feature names to band indices
        band_to_feat = {i: feat for i, feat in enumerate(LANDSAT_FEATURES)}

        # Ingest composites
        files_found = 0
        for year in range(start_year, end_year + 1):
            for period in range(N_PERIODS_PER_YEAR):
                t_idx = (year - start_year) * N_PERIODS_PER_YEAR + period
                tif = landsat_dir / f"{year}_p{period}.tif"

                if not tif.exists():
                    logger.debug(f"Missing: {tif.name}")
                    continue

                files_found += 1
                if files_found % 50 == 0 or files_found == 1:
                    logger.info(f"Processing {tif.name} (composite {files_found})...")

                try:
                    with rasterio.open(tif) as src:
                        # Read all bands at once: shape (n_bands, height, width)
                        raw = src.read()

                    for band_idx, feat in band_to_feat.items():
                        band_data = raw[band_idx].astype(np.float32)

                        # Zero → NaN (nodata)
                        band_data[band_data == 0] = np.nan

                        # Descale
                        band_data /= BAND_SCALE[band_idx]

                        # Clamp negative reflectance to 0 (thermal can be negative in scaled form)
                        if band_idx < N_REFL_BANDS:
                            band_data = np.where(
                                (band_data < 0) & ~np.isnan(band_data), 0.0, band_data
                            )

                        arrays[feat][t_idx, :, :] = band_data

                except Exception as e:
                    logger.error(f"Error processing {tif.name}: {e}")
                    raise

        logger.info(f"Ingested {files_found} composites out of {n_composites} possible")

        # Gap-fill
        if gap_fill:
            logger.info("Running per-pixel seasonal mean gap-fill...")
            self._climatological_fill(group, start_year, end_year)

        # Metadata
        store.attrs["landsat_source"] = str(landsat_dir)
        store.attrs["landsat_start_year"] = start_year
        store.attrs["landsat_end_year"] = end_year
        store.attrs["landsat_gap_filled"] = gap_fill
        store.attrs["landsat_bands"] = BAND_NAMES
        store.attrs["landsat_scale_factors"] = BAND_SCALE

        logger.info("Landsat layer complete")

    def _climatological_fill(
        self,
        group: "zarr.Group",
        start_year: int,
        end_year: int,
    ) -> None:
        """
        Fill NaN pixels with per-pixel seasonal mean.

        For each variable and each period (P0–P4), gather all years for that
        period, compute the mean across years (ignoring NaN), then fill any
        remaining NaN pixels with that mean.

        Memory: processes one variable × one period at a time
        (~n_years × n_y × n_x × 4 bytes per stack).
        """
        n_years = end_year - start_year + 1

        for var in self.variables:
            if var not in group:
                continue

            arr = group[var]

            for period in range(N_PERIODS_PER_YEAR):
                # Indices for this period across all years
                indices = [
                    year_offset * N_PERIODS_PER_YEAR + period
                    for year_offset in range(n_years)
                ]

                # Load stack: (n_years, y, x)
                stack = np.stack([arr[idx, :, :] for idx in indices], axis=0)

                # Per-pixel mean ignoring NaN
                climatology = np.nanmean(stack, axis=0)

                # Fill NaN pixels in each year with the climatology
                filled = 0
                for i, t_idx in enumerate(indices):
                    slab = stack[i]
                    fill_mask = np.isnan(slab) & ~np.isnan(climatology)
                    n_fill = fill_mask.sum()
                    if n_fill > 0:
                        slab[fill_mask] = climatology[fill_mask]
                        arr[t_idx, :, :] = slab
                        filled += n_fill

                if filled > 0:
                    logger.info(
                        f"  {var} P{period}: filled {filled:,} pixels across {n_years} years"
                    )

    def validate(self) -> Dict[str, bool]:
        """Validate Landsat layer with physical range checks."""
        checks = super().validate()

        if not all(checks.values()):
            return checks

        try:
            store = zarr.open(str(self.store_path), mode="r")
            group = store[self.name]

            for var in self.variables:
                if var not in group:
                    continue

                # Sample a temporal slice for validation
                sample = group[var][0:10, ::10, ::10]
                valid = sample[~np.isnan(sample)]

                if valid.size == 0:
                    checks[f"{var}_has_data"] = False
                    continue

                if var.endswith("b10"):
                    # Thermal: expect ~200–350 K range after ÷100
                    checks[f"{var}_range"] = (
                        float(valid.min()) > 100 and float(valid.max()) < 400
                    )
                else:
                    # Reflectance: expect [0, 1.0] after ÷10000
                    checks[f"{var}_range"] = (
                        float(valid.min()) >= 0 and float(valid.max()) <= 1.5
                    )

            # NaN fraction after gap-fill
            sample_all = group[self.variables[0]][:]
            nan_frac = float(np.isnan(sample_all).mean())
            checks["nan_fraction_low"] = nan_frac < 0.10

        except Exception as e:
            logger.error(f"Validation error: {e}")
            checks["validation_error"] = False

        return checks


def build_landsat_layer(
    landsat_dir: Path,
    config: CubeConfig,
    start_year: int = 1987,
    end_year: int = 2025,
    overwrite: bool = False,
    gap_fill: bool = True,
) -> None:
    """
    Convenience function to build Landsat composite layer.

    Args:
        landsat_dir: Directory with {year}_p{period}.tif composites
        config: CubeConfig instance
        start_year: First year (inclusive)
        end_year: Last year (inclusive)
        overwrite: Whether to overwrite existing data
        gap_fill: Whether to apply seasonal mean gap-fill
    """
    config.source_paths["landsat_dir"] = landsat_dir

    layer = LandsatLayer(config)
    layer.build(
        start_year=start_year,
        end_year=end_year,
        overwrite=overwrite,
        gap_fill=gap_fill,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        landsat_dir = sys.argv[1]
        cube_path = sys.argv[2]
    else:
        landsat_dir = "/nas/dads/landsat_composites"
        cube_path = "/data/ssd2/dads_cube/cube.zarr"

    if Path(landsat_dir).exists():
        from cube.config import default_conus_config

        config = default_conus_config(
            cube_path=cube_path,
            landsat_dir=landsat_dir,
        )

        layer = LandsatLayer(config)
        print(layer)
        print(f"Expected shape: {layer._get_expected_shape()}")

        if Path(cube_path).parent.exists():
            print("\nBuilding Landsat layer...")
            layer.build(overwrite=False)
            print("\nValidating...")
            checks = layer.validate()
            for check, passed in checks.items():
                status = "PASS" if passed else "FAIL"
                print(f"  {status} {check}")
    else:
        print(f"Landsat directory not found: {landsat_dir}")
