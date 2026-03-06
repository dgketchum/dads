"""
CDR Native-resolution layer builder for the data cube.

Stores NOAA CDR AVHRR/VIIRS surface reflectance and brightness temperature
at the native 0.05-degree grid resolution in ``cube.zarr/cdr_native/``.

Variables (12):
    - sr1, sr2, sr3: Surface reflectance (channels 1-3), float32
    - bt1, bt2, bt3: Brightness temperature (channels 1-3), float32
    - sr1_miss .. bt3_miss: Missingness flags (uint8, 1=missing 0=valid)

Dimensions: (time, cdr_lat, cdr_lon) — not (y, x).

The PNW subset covers lat 42.0-49.0, lon -125.0 to -104.0 on the global
0.05-degree grid, yielding ~140 lat x 420 lon pixels.

Cross-calibration: VIIRS (2014+) is z-score normalized to AVHRR using
overlap-era statistics (AVHRR 2010-2013, VIIRS 2014-2017).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import zarr

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

from cube.config import (
    CDR_FEATURES,
    CDR_MISS_FEATURES,
    CDR_NATIVE_FEATURES,
    CHUNKS,
    COMPRESSION,
    CubeConfig,
)
from cube.layers.base import BaseLayer

logger = logging.getLogger(__name__)

# Variable name mappings per sensor
AVHRR_VARS = ["SREFL_CH1", "SREFL_CH2", "SREFL_CH3", "BT_CH3", "BT_CH4", "BT_CH5"]
VIIRS_VARS = [
    "BRDF_corrected_I1_SurfRefl_CMG",
    "BRDF_corrected_I2_SurfRefl_CMG",
    "BRDF_corrected_I3_SurfRefl_CMG",
    "BT_CH12",
    "BT_CH15",
    "BT_CH16",
]
HARMONIZED_VARS = ["sr1", "sr2", "sr3", "bt1", "bt2", "bt3"]

VIIRS_START_YEAR = 2014

# Global CDR grid parameters
CDR_RESOLUTION = 0.05  # degrees
CDR_GLOBAL_LAT = np.arange(89.975, -90.0, -CDR_RESOLUTION)  # 3600 values, N to S
CDR_GLOBAL_LON = np.arange(-179.975, 180.0, CDR_RESOLUTION)  # 7200 values, W to E

# Scale factors from CDR documentation
SR_SCALE = 0.0001
BT_SCALE = 0.1

# Date pattern for CDR filenames: extract YYYYMMDD
_DATE_RE = re.compile(r"(\d{8})")


def _parse_date_from_filename(filename: str) -> Optional[pd.Timestamp]:
    """Extract date from CDR filename (YYYYMMDD pattern)."""
    m = _DATE_RE.search(filename)
    if m:
        try:
            return pd.Timestamp(m.group(1))
        except ValueError:
            return None
    return None


def _compute_pnw_subset_indices(
    lat_min: float = 42.0,
    lat_max: float = 49.0,
    lon_min: float = -125.0,
    lon_max: float = -104.0,
) -> Tuple[int, int, int, int]:
    """Compute row/col slice indices for PNW subset of global CDR grid.

    Returns (row_start, row_end, col_start, col_end) for numpy slicing.
    """
    # CDR lat runs N to S, so max lat -> smaller index
    row_start = int(np.searchsorted(-CDR_GLOBAL_LAT, -lat_max))
    row_end = int(np.searchsorted(-CDR_GLOBAL_LAT, -lat_min))
    col_start = int(np.searchsorted(CDR_GLOBAL_LON, lon_min))
    col_end = int(np.searchsorted(CDR_GLOBAL_LON, lon_max))
    return row_start, row_end, col_start, col_end


class CDRNativeLayer(BaseLayer):
    """
    Builds CDR surface reflectance and brightness temperature at native resolution.

    Output group: ``cube.zarr/cdr_native/``
    Dimensions: (time, cdr_lat, cdr_lon)
    """

    def __init__(
        self,
        config: CubeConfig,
        grid=None,
        lat_min: float = 42.0,
        lat_max: float = 49.0,
        lon_min: float = -125.0,
        lon_max: float = -104.0,
    ):
        super().__init__(config, grid)
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max

        # Compute subset indices
        rs, re, cs, ce = _compute_pnw_subset_indices(lat_min, lat_max, lon_min, lon_max)
        self._row_start = rs
        self._row_end = re
        self._col_start = cs
        self._col_end = ce
        self._cdr_lat = CDR_GLOBAL_LAT[rs:re]
        self._cdr_lon = CDR_GLOBAL_LON[cs:ce]

    @property
    def name(self) -> str:
        return "cdr_native"

    @property
    def variables(self) -> List[str]:
        return list(CDR_NATIVE_FEATURES)

    @property
    def dimensions(self) -> Tuple[str, ...]:
        return ("time", "cdr_lat", "cdr_lon")

    @property
    def chunks(self) -> Dict[str, int]:
        return CHUNKS["cdr_native"]

    @property
    def n_cdr_lat(self) -> int:
        return len(self._cdr_lat)

    @property
    def n_cdr_lon(self) -> int:
        return len(self._cdr_lon)

    def _get_expected_shape(self) -> Tuple[int, ...]:
        """Override to return CDR-specific shape."""
        start = pd.Timestamp(self.config.start_date)
        end = pd.Timestamp(self.config.end_date)
        n_time = (end - start).days + 1
        return (n_time, self.n_cdr_lat, self.n_cdr_lon)

    def _write_cdr_coords(self, store: "zarr.Group") -> None:
        """Write 1D cdr_lat and cdr_lon coordinate arrays."""
        if "cdr_lat" not in store:
            arr = store.create_dataset(
                "cdr_lat",
                shape=self._cdr_lat.astype(np.float64).shape,
                data=self._cdr_lat.astype(np.float64),
                chunks=(len(self._cdr_lat),),
                dtype="float64",
            )
            arr.attrs["standard_name"] = "latitude"
            arr.attrs["units"] = "degrees_north"
            arr.attrs["axis"] = "Y"

        if "cdr_lon" not in store:
            arr = store.create_dataset(
                "cdr_lon",
                shape=self._cdr_lon.astype(np.float64).shape,
                data=self._cdr_lon.astype(np.float64),
                chunks=(len(self._cdr_lon),),
                dtype="float64",
            )
            arr.attrs["standard_name"] = "longitude"
            arr.attrs["units"] = "degrees_east"
            arr.attrs["axis"] = "X"

    def build(
        self,
        source_paths: Optional[Dict[str, Path]] = None,
        overwrite: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        normalize_viirs: bool = True,
        avhrr_years: Tuple[int, int] = (2010, 2013),
        viirs_years: Tuple[int, int] = (2014, 2017),
    ) -> None:
        """
        Build CDR native layer from NetCDF files.

        Parameters
        ----------
        source_paths : dict, optional
            Dict with 'cdr_dir' key. Falls back to config.source_paths.
        overwrite : bool
            Whether to overwrite existing arrays.
        start_date, end_date : str, optional
            Override config date range.
        normalize_viirs : bool
            Apply AVHRR/VIIRS cross-calibration.
        avhrr_years, viirs_years : tuple
            Year ranges for computing normalization statistics.
        """
        if not HAS_XARRAY:
            raise ImportError("xarray required for CDR layer building")

        source_paths = source_paths or self.config.source_paths
        cdr_dir = source_paths.get("cdr_dir")
        if cdr_dir is None:
            raise ValueError("cdr_dir not in source_paths")

        cdr_dir = Path(cdr_dir)
        if not cdr_dir.exists():
            raise FileNotFoundError(f"CDR directory not found: {cdr_dir}")

        start = pd.Timestamp(start_date or self.config.start_date)
        end = pd.Timestamp(end_date or self.config.end_date)
        dates = pd.date_range(start, end, freq="D")
        n_times = len(dates)

        logger.info("Building CDR native layer from %s", cdr_dir)
        logger.info("Date range: %s to %s (%d days)", start.date(), end.date(), n_times)
        logger.info("CDR subset: %d lat x %d lon", self.n_cdr_lat, self.n_cdr_lon)

        # Scan and sort NetCDF files by date
        nc_files = sorted(cdr_dir.glob("*.nc"))
        file_dates = {}
        for f in nc_files:
            dt = _parse_date_from_filename(f.name)
            if dt is not None and start <= dt <= end:
                file_dates[f] = dt

        logger.info("Found %d NetCDF files in date range", len(file_dates))

        # Open store and write coordinates
        store = self._open_store("a")
        self._write_cdr_coords(store)
        self._write_time_coord(store, dates.values.astype("datetime64[D]"))
        group = store.require_group(self.name)

        # Create arrays
        shape = (n_times, self.n_cdr_lat, self.n_cdr_lon)
        time_chunk = min(self.chunks.get("time", 365), n_times)
        lat_chunk = min(self.chunks.get("cdr_lat", self.n_cdr_lat), self.n_cdr_lat)
        lon_chunk = min(self.chunks.get("cdr_lon", self.n_cdr_lon), self.n_cdr_lon)
        float_chunks = (time_chunk, lat_chunk, lon_chunk)
        compressor_f = COMPRESSION.get("float")
        compressor_i = COMPRESSION.get("int")

        arrays = {}
        for var in CDR_FEATURES:
            if var in group and not overwrite:
                arrays[var] = group[var]
            else:
                arrays[var] = group.create_dataset(
                    var,
                    shape=shape,
                    chunks=float_chunks,
                    dtype="float32",
                    compressors=compressor_f,
                    fill_value=np.nan,
                    overwrite=overwrite,
                )

        for var in CDR_MISS_FEATURES:
            if var in group and not overwrite:
                arrays[var] = group[var]
            else:
                arrays[var] = group.create_dataset(
                    var,
                    shape=shape,
                    chunks=float_chunks,
                    dtype="uint8",
                    compressors=compressor_i,
                    fill_value=1,
                    overwrite=overwrite,
                )

        # Date-to-index mapping
        date_to_idx = {d.date(): i for i, d in enumerate(dates)}

        # Process files sorted by date
        sorted_files = sorted(file_dates.items(), key=lambda x: x[1])
        n_processed = 0
        for filepath, file_dt in sorted_files:
            t_idx = date_to_idx.get(file_dt.date())
            if t_idx is None:
                continue

            try:
                self._process_one_file(filepath, file_dt.year, t_idx, arrays)
                n_processed += 1
                if n_processed % 500 == 0:
                    logger.info(
                        "  processed %d / %d files", n_processed, len(sorted_files)
                    )
            except Exception as e:
                logger.warning("Error processing %s: %s", filepath.name, e)

        logger.info("Processed %d files total", n_processed)

        # Cross-calibration
        if normalize_viirs:
            norm_stats = self._cross_calibrate(group, dates, avhrr_years, viirs_years)
            group.attrs["normalization_stats"] = {
                k: {
                    "avhrr_mean": v[0],
                    "avhrr_std": v[1],
                    "viirs_mean": v[2],
                    "viirs_std": v[3],
                }
                for k, v in norm_stats.items()
            }

        # Provenance
        group.attrs["source_dir"] = str(cdr_dir)
        group.attrs["start_date"] = str(start.date())
        group.attrs["end_date"] = str(end.date())
        group.attrs["lat_range"] = [self.lat_min, self.lat_max]
        group.attrs["lon_range"] = [self.lon_min, self.lon_max]
        group.attrs["n_files_processed"] = n_processed
        group.attrs["viirs_normalized"] = normalize_viirs

        logger.info("CDR native layer complete")

    def _process_one_file(
        self,
        filepath: Path,
        year: int,
        t_idx: int,
        arrays: Dict[str, zarr.Array],
    ) -> None:
        """Process a single CDR NetCDF file and write to zarr arrays."""
        is_viirs = year >= VIIRS_START_YEAR
        input_vars = VIIRS_VARS if is_viirs else AVHRR_VARS

        ds = xr.open_dataset(filepath, engine="netcdf4", decode_cf=False)

        try:
            # Verify lat/lon dimension names exist
            if "latitude" not in ds.dims and "lat" not in ds.dims:
                raise ValueError(
                    f"Unknown lat/lon dims in {filepath.name}: {list(ds.dims)}"
                )

            for in_var, out_var in zip(input_vars, HARMONIZED_VARS):
                if in_var not in ds:
                    continue

                # Extract PNW subset
                data = ds[in_var].values
                if data.ndim == 3:
                    data = data[0]  # take first time step if present
                subset = data[
                    self._row_start : self._row_end, self._col_start : self._col_end
                ]

                # Apply scale factors
                subset = subset.astype(np.float32)
                if out_var.startswith("sr"):
                    subset *= SR_SCALE
                else:
                    subset *= BT_SCALE

                # Replace fill values with NaN
                subset[subset < -900] = np.nan

                # Clamp negative SR to 0
                if out_var.startswith("sr"):
                    subset = np.where(subset < 0, 0.0, subset)

                # Write data
                arrays[out_var][t_idx, :, :] = subset

                # Write missingness flag
                miss_var = out_var + "_miss"
                miss = np.where(np.isnan(subset), np.uint8(1), np.uint8(0))
                arrays[miss_var][t_idx, :, :] = miss
        finally:
            ds.close()

    def _cross_calibrate(
        self,
        group: "zarr.Group",
        dates: pd.DatetimeIndex,
        avhrr_years: Tuple[int, int],
        viirs_years: Tuple[int, int],
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """Apply VIIRS-to-AVHRR z-score normalization in-place.

        Returns dict mapping var name to (avhrr_mean, avhrr_std, viirs_mean, viirs_std).
        """
        logger.info("Cross-calibrating VIIRS to AVHRR scale...")

        # Build time masks
        years = dates.year
        avhrr_mask = (years >= avhrr_years[0]) & (years <= avhrr_years[1])
        viirs_mask = (years >= viirs_years[0]) & (years <= viirs_years[1])
        viirs_all_mask = years >= VIIRS_START_YEAR

        avhrr_idx = np.where(avhrr_mask)[0]
        viirs_stat_idx = np.where(viirs_mask)[0]
        viirs_all_idx = np.where(viirs_all_mask)[0]

        if len(avhrr_idx) == 0 or len(viirs_stat_idx) == 0:
            logger.warning("Insufficient data for cross-calibration")
            return {}

        stats = {}
        # Sample spatially for statistics (every 4th pixel to reduce memory)
        lat_slice = slice(None, None, 4)
        lon_slice = slice(None, None, 4)

        for var in CDR_FEATURES:
            arr = group[var]

            # Compute AVHRR stats from sample
            avhrr_samples = []
            for idx in avhrr_idx[::7]:  # weekly samples
                slab = arr[idx, lat_slice, lon_slice]
                valid = slab[~np.isnan(slab)]
                if len(valid) > 0:
                    avhrr_samples.append(valid)

            if not avhrr_samples:
                continue

            avhrr_all = np.concatenate(avhrr_samples)
            a_mean = float(np.mean(avhrr_all))
            a_std = float(np.std(avhrr_all))

            if a_std == 0:
                continue

            # Compute VIIRS stats from sample
            viirs_samples = []
            for idx in viirs_stat_idx[::7]:
                slab = arr[idx, lat_slice, lon_slice]
                valid = slab[~np.isnan(slab)]
                if len(valid) > 0:
                    viirs_samples.append(valid)

            if not viirs_samples:
                continue

            viirs_all = np.concatenate(viirs_samples)
            v_mean = float(np.mean(viirs_all))
            v_std = float(np.std(viirs_all))

            if v_std == 0:
                continue

            stats[var] = (a_mean, a_std, v_mean, v_std)
            logger.info(
                "  %s: AVHRR(%.4f, %.4f) VIIRS(%.4f, %.4f)",
                var,
                a_mean,
                a_std,
                v_mean,
                v_std,
            )

            # Apply normalization to all VIIRS time steps
            for idx in viirs_all_idx:
                slab = arr[idx, :, :]
                normalized = (slab - v_mean) * (a_std / v_std) + a_mean
                arr[idx, :, :] = normalized

        return stats

    def validate(self) -> Dict[str, bool]:
        """Validate CDR native layer."""
        checks = {}

        if not self.store_path.exists():
            checks["store_exists"] = False
            return checks

        try:
            store = zarr.open(str(self.store_path), mode="r")
            if self.name not in store:
                checks["group_exists"] = False
                return checks
            checks["group_exists"] = True

            group = store[self.name]

            for var in self.variables:
                checks[f"{var}_exists"] = var in group

            # Check coordinates
            checks["cdr_lat_exists"] = "cdr_lat" in store
            checks["cdr_lon_exists"] = "cdr_lon" in store

            # Sample check on shapes
            if CDR_FEATURES[0] in group:
                arr = group[CDR_FEATURES[0]]
                checks["ndim"] = arr.ndim == 3
                checks["spatial_shape"] = arr.shape[1:] == (
                    self.n_cdr_lat,
                    self.n_cdr_lon,
                )

        except Exception as e:
            logger.error("Validation error: %s", e)
            checks["error"] = False

        return checks


def build_cdr_native_layer(
    cdr_dir: str,
    config: CubeConfig,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    overwrite: bool = False,
    **kwargs,
) -> None:
    """Convenience function to build CDR native layer."""
    config.source_paths["cdr_dir"] = Path(cdr_dir)
    layer = CDRNativeLayer(config)
    layer.build(
        start_date=start_date,
        end_date=end_date,
        overwrite=overwrite,
        **kwargs,
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    cdr_dir = sys.argv[1] if len(sys.argv) > 1 else "/nas/dads/rs/cdr/nc"
    cube_path = sys.argv[2] if len(sys.argv) > 2 else "/data/ssd2/dads_cube/cube.zarr"

    if Path(cdr_dir).exists():
        from cube.config import default_conus_config

        cfg = default_conus_config(cube_path=cube_path, cdr_dir=cdr_dir)
        layer = CDRNativeLayer(cfg)
        print(layer)
        print(f"CDR subset: {layer.n_cdr_lat} lat x {layer.n_cdr_lon} lon")
        print(f"Expected shape: {layer._get_expected_shape()}")
    else:
        print(f"CDR directory not found: {cdr_dir}")
