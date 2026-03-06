"""
RTMA temperature climatology layer builder for the data cube.

Builds DOY-indexed tmax climatology from multi-year daily RTMA GeoTIFFs.
Used as the pretraining background surface: B_rtma(x, doy).

Input format: Directory with RTMA_YYYYMMDD.tif files (10-band int32, EPSG:4326).
              Band 2 (TMP) contains temperature in centidegrees C.

Output: cube.zarr/doy_indexed/tmax_clim_rtma with shape (365, n_y, n_x)
        Units: degrees Celsius
"""

from __future__ import annotations

import datetime
import logging
from collections import defaultdict
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

from cube.config import CHUNKS
from cube.layers.base import BaseLayer

logger = logging.getLogger(__name__)

# RTMA Band 2 is TMP (temperature) in centidegrees C
DEFAULT_TMP_BAND = 2

# Reasonable TMP range in degrees C (anything outside is treated as nodata)
VALID_TEMP_MIN = -80.0
VALID_TEMP_MAX = 70.0


def _date_to_doy365(month: int, day: int) -> int:
    """Map (month, day) to DOY in [1, 365], with Feb 29 mapped to 60 (Mar 1).

    Uses a non-leap year (2001) as reference so every (month, day) maps
    consistently to the same DOY regardless of source year.
    """
    if month == 2 and day == 29:
        return 60
    return datetime.date(2001, month, day).timetuple().tm_yday


def _discover_rtma_files(rtma_dir: Path) -> List[Tuple[int, Path]]:
    """Discover RTMA_YYYYMMDD.tif files and return (doy365, path) pairs."""
    results = []
    for p in sorted(rtma_dir.glob("RTMA_*.tif")):
        try:
            datestr = p.stem.split("_")[1]
            dt = datetime.datetime.strptime(datestr, "%Y%m%d")
            doy = _date_to_doy365(dt.month, dt.day)
            results.append((doy, p))
        except (ValueError, IndexError):
            continue
    return results


class RTMAClimLayer(BaseLayer):
    """
    Builds DOY-indexed RTMA temperature climatology for pretraining.

    Computes the per-pixel mean of RTMA Band 2 (TMP) across all available
    years for each DOY (365 days, Feb 29 folded into DOY 60). The native
    ~2.5 km EPSG:4326 data is resampled to the 1 km EPSG:5070 master grid.

    Optional Gaussian smoothing along the DOY dimension removes year-to-year
    noise and creates a cleaner climatological signal.

    Output: (365, y, x) array in degrees Celsius.
    """

    @property
    def name(self) -> str:
        return "doy_indexed"

    @property
    def variables(self) -> List[str]:
        return ["tmax_clim_rtma"]

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
        smooth_sigma: float = 5.0,
        tmp_band: int = DEFAULT_TMP_BAND,
        resume: bool = False,
    ) -> None:
        """
        Build tmax_clim_rtma layer from RTMA GeoTIFFs.

        Args:
            source_paths: Dict with 'rtma_dir' key pointing to RTMA GeoTIFF directory
            overwrite: Whether to overwrite existing data
            smooth_sigma: Gaussian smoothing sigma (DOY units). 0 disables smoothing.
            tmp_band: Band index for TMP (1-indexed, default 2)
            resume: If True and the array exists, skip DOYs that already have data
        """
        if not HAS_RASTERIO:
            raise ImportError("rasterio required for RTMA clim layer building")

        source_paths = source_paths or self.config.source_paths
        rtma_dir = source_paths.get("rtma_dir")

        if rtma_dir is None:
            raise ValueError("rtma_dir path not provided in source_paths['rtma_dir']")

        rtma_dir = Path(rtma_dir)
        if not rtma_dir.exists():
            raise FileNotFoundError(f"RTMA directory not found: {rtma_dir}")

        logger.info("Building tmax_clim_rtma layer from %s", rtma_dir)

        # Discover and group files by DOY
        doy_files_list = _discover_rtma_files(rtma_dir)
        if not doy_files_list:
            raise FileNotFoundError(f"No RTMA_*.tif files found in {rtma_dir}")

        doy_groups: Dict[int, List[Path]] = defaultdict(list)
        for doy, path in doy_files_list:
            doy_groups[doy].append(path)

        n_files = len(doy_files_list)
        n_doys = len(doy_groups)
        logger.info("  %d files across %d DOYs", n_files, n_doys)

        # Open store and create group
        store = self._open_store("a")
        self._write_coords(store)
        self._write_doy_coord(store)
        group = self._ensure_group(store)

        existing = "tmax_clim_rtma" in group

        if existing and not overwrite and not resume:
            logger.info(
                "tmax_clim_rtma already exists, skipping (use overwrite=True or resume=True)"
            )
            return

        shape = (365, self.grid.n_y, self.grid.n_x)
        chunks = (self.chunks["doy"], self.chunks["y"], self.chunks["x"])

        if existing and resume:
            clim_array = group["tmax_clim_rtma"]
            logger.info("Resuming build into existing array %s", clim_array.shape)
        else:
            clim_array = group.create_dataset(
                "tmax_clim_rtma",
                shape=shape,
                chunks=chunks,
                dtype="float32",
                compressors=self.compression,
                fill_value=np.nan,
                overwrite=overwrite,
            )

        from cube.builders.resampler import GridResampler

        resampler = GridResampler(self.grid)

        skipped = 0
        for doy in range(1, 366):
            files = doy_groups.get(doy, [])
            if not files:
                if doy <= 10 or doy % 30 == 0:
                    logger.warning("  DOY %d: no files, leaving as NaN", doy)
                continue

            # Resume: skip DOYs that already have data
            if resume and existing:
                slab = clim_array[doy - 1, :, :]
                if np.any(~np.isnan(slab)):
                    skipped += 1
                    continue

            mean_native, src_transform, src_crs = _compute_doy_mean(files, tmp_band)

            resampled = resampler.resample_array(
                mean_native,
                src_transform=src_transform,
                src_crs=src_crs,
                method="bilinear",
            )

            clim_array[doy - 1, :, :] = resampled.astype(np.float32)

            if doy == 1 or doy % 30 == 0:
                logger.info("  DOY %d/%d done (%d files)", doy, 365, len(files))

        if skipped:
            logger.info("  Resumed: skipped %d DOYs with existing data", skipped)

        # Temporal smoothing
        if smooth_sigma > 0:
            logger.info("Applying DOY smoothing (sigma=%.1f)", smooth_sigma)
            _smooth_doy_inplace(clim_array, smooth_sigma)

        # Metadata
        years = set()
        for _, path in doy_files_list:
            try:
                years.add(int(path.stem.split("_")[1][:4]))
            except (ValueError, IndexError):
                pass

        store.attrs["tmax_clim_rtma_source"] = str(rtma_dir)
        store.attrs["tmax_clim_rtma_units"] = "degC"
        store.attrs["tmax_clim_rtma_tmp_band"] = tmp_band
        store.attrs["tmax_clim_rtma_n_files"] = n_files
        store.attrs["tmax_clim_rtma_smooth_sigma"] = smooth_sigma
        if years:
            store.attrs["tmax_clim_rtma_year_range"] = [min(years), max(years)]

        logger.info("tmax_clim_rtma layer complete")

    def validate(self) -> Dict[str, bool]:
        """Validate tmax_clim_rtma layer with physical range checks."""
        checks = super().validate()

        if not all(checks.values()):
            return checks

        try:
            store = zarr.open(str(self.store_path), mode="r")
            group = store[self.name]

            clim = group["tmax_clim_rtma"][:]
            valid = clim[~np.isnan(clim)]

            if len(valid) == 0:
                checks["has_data"] = False
                return checks
            checks["has_data"] = True

            # Physical range: CONUS temps should be in [-40, 55] degC
            checks["temp_min"] = float(valid.min()) >= -60.0
            checks["temp_max"] = float(valid.max()) <= 60.0

            # Seasonal pattern: summer DOYs should be warmer than winter
            summer = clim[150:220, :, :]
            winter = clim[330:365, :, :]
            has_summer = np.any(~np.isnan(summer))
            has_winter = np.any(~np.isnan(winter))
            if has_summer and has_winter:
                checks["seasonal_pattern"] = float(np.nanmean(summer)) > float(
                    np.nanmean(winter)
                )
            else:
                # Can't check seasonal pattern without data in both ranges
                checks["seasonal_pattern"] = True

            # Coverage: at least 50% non-NaN
            nan_frac = np.isnan(clim).mean()
            checks["coverage"] = nan_frac < 0.5

        except Exception as e:
            logger.error("Validation error: %s", e)
            checks["validation_error"] = False

        return checks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_doy_mean(
    files: List[Path],
    tmp_band: int,
) -> Tuple[np.ndarray, object, str]:
    """Read TMP band from all files for one DOY and return per-pixel mean.

    Returns (mean_array, src_transform, src_crs) where mean_array is in
    degrees C (float32) at the native RTMA resolution.
    """
    acc_sum = None
    acc_count = None
    src_transform = None
    src_crs = None

    for f in files:
        with rasterio.open(f) as src:
            raw = src.read(tmp_band).astype(np.float64)

            if acc_sum is None:
                acc_sum = np.zeros_like(raw)
                acc_count = np.zeros(raw.shape, dtype=np.int32)
                src_transform = src.transform
                src_crs = str(src.crs)

            # Convert centidegrees to degrees
            temp = raw / 100.0

            # Mask unreasonable values
            valid = (temp >= VALID_TEMP_MIN) & (temp <= VALID_TEMP_MAX)
            acc_sum[valid] += temp[valid]
            acc_count[valid] += 1

    mean = np.where(acc_count > 0, acc_sum / acc_count, np.nan).astype(np.float32)
    return mean, src_transform, src_crs


def _smooth_doy_inplace(arr: zarr.Array, sigma: float) -> None:
    """Apply circular Gaussian smoothing along DOY axis of a zarr array.

    Processes in spatial chunks to keep memory manageable. Uses NaN-aware
    normalized convolution so that missing pixels are handled correctly.
    """
    from scipy.ndimage import gaussian_filter1d

    pad = int(3 * sigma) + 1
    chunk_y = arr.chunks[1]
    chunk_x = arr.chunks[2]
    n_y = arr.shape[1]
    n_x = arr.shape[2]

    for y0 in range(0, n_y, chunk_y):
        y1 = min(y0 + chunk_y, n_y)
        for x0 in range(0, n_x, chunk_x):
            x1 = min(x0 + chunk_x, n_x)

            block = arr[:, y0:y1, x0:x1].astype(np.float64)
            mask = (~np.isnan(block)).astype(np.float64)
            block_filled = np.where(np.isnan(block), 0.0, block)

            # Circular padding for DOY wrap-around
            padded_data = np.concatenate(
                [block_filled[-pad:], block_filled, block_filled[:pad]], axis=0
            )
            padded_mask = np.concatenate([mask[-pad:], mask, mask[:pad]], axis=0)

            smooth_data = gaussian_filter1d(padded_data, sigma, axis=0)
            smooth_mask = gaussian_filter1d(padded_mask, sigma, axis=0)

            center = smooth_data[pad : pad + 365]
            center_mask = smooth_mask[pad : pad + 365]

            result = np.where(
                center_mask > 1e-6,
                center / center_mask,
                np.nan,
            ).astype(np.float32)

            arr[:, y0:y1, x0:x1] = result

        if y0 > 0 and y0 % (chunk_y * 10) == 0:
            logger.info("  smoothing: row %d / %d", y0, n_y)
