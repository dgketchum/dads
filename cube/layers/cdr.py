"""
CDR (Climate Data Record) layer builder for the data cube.

Builds daily time series of NOAA CDR AVHRR/VIIRS surface reflectance and
brightness temperature from NetCDF files.

Variables:
    - sr1, sr2, sr3: Surface reflectance (channels 1-3)
    - bt1, bt2, bt3: Brightness temperature (channels 1-3)

Input format: Monthly NetCDF files with naming convention:
    AVHRR-Land_*_{YYYYMM}*.nc (pre-2014)
    VIIRS_*_{YYYYMM}*.nc (2014+)

The layer harmonizes AVHRR and VIIRS variable names and applies basic
cross-calibration normalization.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

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

from cube.layers.base import BaseLayer
from cube.grid import MasterGrid
from cube.config import CubeConfig, CHUNKS, CDR_FEATURES

logger = logging.getLogger(__name__)

# Variable name mappings
AVHRR_VARS = ['SREFL_CH1', 'SREFL_CH2', 'SREFL_CH3', 'BT_CH3', 'BT_CH4', 'BT_CH5']
VIIRS_VARS = [
    'BRDF_corrected_I1_SurfRefl_CMG',
    'BRDF_corrected_I2_SurfRefl_CMG',
    'BRDF_corrected_I3_SurfRefl_CMG',
    'BT_CH12', 'BT_CH15', 'BT_CH16'
]
HARMONIZED_VARS = ['sr1', 'sr2', 'sr3', 'bt1', 'bt2', 'bt3']

# Transition year from AVHRR to VIIRS
VIIRS_START_YEAR = 2014


class CDRLayer(BaseLayer):
    """
    Builds daily CDR surface reflectance and brightness temperature layer.

    Handles both AVHRR (1981-2013) and VIIRS (2014-present) sensors,
    harmonizing variable names and applying cross-calibration.

    Output: (time, lat, lon) arrays for sr1, sr2, sr3, bt1, bt2, bt3
    """

    @property
    def name(self) -> str:
        return 'daily'

    @property
    def variables(self) -> List[str]:
        return list(HARMONIZED_VARS)  # sr1, sr2, sr3, bt1, bt2, bt3

    @property
    def dimensions(self) -> Tuple[str, ...]:
        return ('time', 'lat', 'lon')

    @property
    def chunks(self) -> Dict[str, int]:
        return CHUNKS['daily']

    def build(
        self,
        source_paths: Optional[Dict[str, Path]] = None,
        overwrite: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        normalize_viirs: bool = True,
    ) -> None:
        """
        Build CDR layer from NetCDF files.

        Args:
            source_paths: Dict with 'cdr_dir' key pointing to NetCDF directory
            overwrite: Whether to overwrite existing data
            start_date: Start date (default: config.start_date)
            end_date: End date (default: config.end_date)
            normalize_viirs: Whether to normalize VIIRS to AVHRR scale
        """
        if not HAS_XARRAY:
            raise ImportError("xarray required for CDR layer building")

        source_paths = source_paths or self.config.source_paths
        cdr_dir = source_paths.get('cdr_dir')

        if cdr_dir is None:
            raise ValueError("cdr_dir path not provided in source_paths['cdr_dir']")

        cdr_dir = Path(cdr_dir)
        if not cdr_dir.exists():
            raise FileNotFoundError(f"CDR directory not found: {cdr_dir}")

        # Date range
        start = pd.Timestamp(start_date or self.config.start_date)
        end = pd.Timestamp(end_date or self.config.end_date)

        logger.info(f"Building CDR layer from {cdr_dir}")
        logger.info(f"Date range: {start.date()} to {end.date()}")

        # Build date index
        dates = pd.date_range(start, end, freq='D')
        n_times = len(dates)

        logger.info(f"Total time steps: {n_times:,}")

        # Open store and create group
        store = self._open_store('a')
        self._write_coords(store)
        self._write_time_coord(store, dates.values.astype('datetime64[D]'))
        group = self._ensure_group(store)

        # Create output arrays for each variable
        shape = (n_times, self.grid.n_lat, self.grid.n_lon)
        chunks = (self.chunks['time'], self.chunks['lat'], self.chunks['lon'])

        arrays = {}
        for var in self.variables:
            if var in group and not overwrite:
                logger.info(f"{var} already exists, skipping")
                arrays[var] = group[var]
            else:
                arrays[var] = group.create_dataset(
                    var,
                    shape=shape,
                    chunks=chunks,
                    dtype='float32',
                    compressor=self.compression,
                    fill_value=np.nan,
                    overwrite=overwrite,
                )

        # Compute AVHRR statistics for VIIRS normalization
        avhrr_stats = None
        if normalize_viirs:
            avhrr_stats = self._compute_avhrr_stats(cdr_dir, start, min(end, pd.Timestamp('2013-12-31')))

        # Process by month for memory efficiency
        from cube.builders.resampler import GridResampler
        resampler = GridResampler(self.grid)

        for year in range(start.year, end.year + 1):
            for month in range(1, 13):
                month_start = pd.Timestamp(year, month, 1)
                if month_start < start or month_start > end:
                    continue

                month_end = month_start + pd.offsets.MonthEnd()
                if month_end > end:
                    month_end = end

                logger.info(f"Processing {year}-{month:02d}...")

                try:
                    # Load month's data
                    month_data = self._load_month(
                        cdr_dir, year, month, resampler,
                        normalize_viirs=normalize_viirs,
                        avhrr_stats=avhrr_stats,
                    )

                    if month_data is None:
                        logger.warning(f"No data for {year}-{month:02d}")
                        continue

                    # Find time indices for this month
                    month_dates = month_data['time']
                    for var in self.variables:
                        if var not in month_data:
                            continue

                        var_data = month_data[var]

                        # Write each day
                        for i, date in enumerate(month_dates):
                            t_idx = (date - dates[0]).days
                            if 0 <= t_idx < n_times:
                                arrays[var][t_idx, :, :] = var_data[i].astype(np.float32)

                except Exception as e:
                    logger.error(f"Error processing {year}-{month:02d}: {e}")
                    continue

        # Store metadata
        store.attrs['cdr_source'] = str(cdr_dir)
        store.attrs['cdr_start_date'] = str(start.date())
        store.attrs['cdr_end_date'] = str(end.date())
        store.attrs['cdr_viirs_normalized'] = normalize_viirs

        logger.info("CDR layer complete")

    def _load_month(
        self,
        cdr_dir: Path,
        year: int,
        month: int,
        resampler: 'GridResampler',
        normalize_viirs: bool = True,
        avhrr_stats: Optional[Dict] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Load and process a month of CDR data.

        Args:
            cdr_dir: CDR NetCDF directory
            year: Year
            month: Month
            resampler: GridResampler instance
            normalize_viirs: Whether to normalize VIIRS data
            avhrr_stats: AVHRR mean/std for normalization

        Returns:
            Dict with 'time' and variable arrays, or None if no data
        """
        date_string = f'{year}{month:02d}'

        # Find NetCDF files for this month
        nc_files = list(cdr_dir.glob(f'*_{date_string}*.nc'))
        if not nc_files:
            nc_files = list(cdr_dir.glob(f'*{date_string}*.nc'))

        if not nc_files:
            return None

        # Determine sensor type
        is_viirs = year >= VIIRS_START_YEAR
        input_vars = VIIRS_VARS if is_viirs else AVHRR_VARS

        # Load and concatenate files
        datasets = []
        for f in sorted(nc_files):
            try:
                ds = xr.open_dataset(f, engine='netcdf4', decode_cf=False)
                # Subset to approximate CONUS bounds
                w, s, e, n = self.grid.bounds
                ds = ds.sel(latitude=slice(n + 1, s - 1), longitude=slice(w - 1, e + 1))
                datasets.append(ds)
            except Exception as e:
                logger.warning(f"Error reading {f}: {e}")
                continue

        if not datasets:
            return None

        # Concatenate along time
        ds = xr.concat(datasets, dim='time')

        # Convert time coordinate
        time_values = pd.to_datetime(
            ds['time'].values,
            unit='D',
            origin=pd.Timestamp('1981-01-01')
        )
        ds = ds.assign_coords(time=time_values)

        # Extract and resample each variable
        result = {'time': time_values.values}

        for in_var, out_var in zip(input_vars, HARMONIZED_VARS):
            if in_var not in ds:
                logger.warning(f"Variable {in_var} not found in dataset")
                continue

            var_data = ds[in_var].values

            # Handle fill values / negative values for reflectance
            if out_var.startswith('sr'):
                var_data = np.where(var_data < 0, np.nan, var_data)

            # Resample each timestep
            resampled = np.empty(
                (len(time_values), self.grid.n_lat, self.grid.n_lon),
                dtype=np.float32
            )

            for t in range(len(time_values)):
                # Get source transform and CRS
                lat = ds['latitude'].values
                lon = ds['longitude'].values

                # Create affine transform from coordinates
                import rasterio
                from rasterio.crs import CRS

                # Assume regular grid
                lat_res = abs(lat[1] - lat[0]) if len(lat) > 1 else 0.05
                lon_res = abs(lon[1] - lon[0]) if len(lon) > 1 else 0.05

                src_transform = rasterio.Affine(
                    lon_res, 0, lon.min() - lon_res / 2,
                    0, -lat_res, lat.max() + lat_res / 2
                )

                resampled[t] = resampler.resample_array(
                    var_data[t],
                    src_transform=src_transform,
                    src_crs='EPSG:4326',
                    method='bilinear',
                )

            # Normalize VIIRS to AVHRR scale
            if is_viirs and normalize_viirs and avhrr_stats is not None:
                if out_var in avhrr_stats:
                    a_mean, a_std = avhrr_stats[out_var]
                    v_mean = np.nanmean(resampled)
                    v_std = np.nanstd(resampled)
                    if v_std > 0:
                        resampled = (resampled - v_mean) * (a_std / v_std) + a_mean

            result[out_var] = resampled

        # Close datasets
        ds.close()
        for d in datasets:
            d.close()

        return result

    def _compute_avhrr_stats(
        self,
        cdr_dir: Path,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute AVHRR mean and std for VIIRS normalization.

        Samples a subset of AVHRR data to compute statistics efficiently.
        """
        logger.info("Computing AVHRR statistics for VIIRS normalization...")

        stats = {}
        sample_years = list(range(max(start.year, 2000), min(end.year + 1, 2014)))

        if not sample_years:
            logger.warning("No AVHRR years available for normalization")
            return stats

        # Sample a few months per year
        sample_months = [1, 4, 7, 10]

        values = {var: [] for var in HARMONIZED_VARS}

        for year in sample_years[-3:]:  # Last 3 years of AVHRR
            for month in sample_months:
                date_string = f'{year}{month:02d}'
                nc_files = list(cdr_dir.glob(f'*_{date_string}*.nc'))

                if not nc_files:
                    continue

                try:
                    ds = xr.open_dataset(nc_files[0], engine='netcdf4', decode_cf=False)

                    for in_var, out_var in zip(AVHRR_VARS, HARMONIZED_VARS):
                        if in_var in ds:
                            data = ds[in_var].values.flatten()
                            valid = data[~np.isnan(data) & (data > 0)]
                            if len(valid) > 0:
                                # Sample to reduce memory
                                sample = valid[::100] if len(valid) > 10000 else valid
                                values[out_var].extend(sample.tolist())

                    ds.close()

                except Exception as e:
                    logger.warning(f"Error computing stats for {year}-{month}: {e}")

        for var in HARMONIZED_VARS:
            if values[var]:
                arr = np.array(values[var])
                stats[var] = (np.mean(arr), np.std(arr))
                logger.info(f"  {var}: mean={stats[var][0]:.2f}, std={stats[var][1]:.2f}")

        return stats

    def validate(self) -> Dict[str, bool]:
        """Validate CDR layer with physical checks."""
        checks = super().validate()

        if not all(checks.values()):
            return checks

        try:
            store = zarr.open(str(self.store_path), mode='r')
            group = store[self.name]

            # Check each variable
            for var in self.variables:
                if var not in group:
                    checks[f'{var}_exists'] = False
                    continue

                data = group[var]

                # Sample a slice for validation
                sample = data[0:365, ::10, ::10]
                valid = sample[~np.isnan(sample)]

                if var.startswith('sr'):
                    # Surface reflectance: 0-10000 (scaled) or 0-1
                    checks[f'{var}_range'] = valid.min() >= 0 and valid.max() <= 15000
                else:
                    # Brightness temperature: typically 200-350 K
                    checks[f'{var}_range'] = valid.min() > 150 and valid.max() < 400

            # Check temporal continuity (sample)
            time_valid = np.sum(~np.isnan(group['sr1'][:, 100, 100])) > 0
            checks['temporal_data'] = time_valid

        except Exception as e:
            logger.error(f"Validation error: {e}")
            checks['validation_error'] = False

        return checks


def build_cdr_layer(
    cdr_dir: Path,
    config: CubeConfig,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Convenience function to build CDR layer.

    Args:
        cdr_dir: Directory with CDR NetCDF files
        config: CubeConfig instance
        start_date: Start date (default: config.start_date)
        end_date: End date (default: config.end_date)
        overwrite: Whether to overwrite existing data
    """
    config.source_paths['cdr_dir'] = cdr_dir
    layer = CDRLayer(config)
    layer.build(
        start_date=start_date,
        end_date=end_date,
        overwrite=overwrite,
    )


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 2:
        cdr_dir = sys.argv[1]
        cube_path = sys.argv[2]
    else:
        cdr_dir = '/home/dgketchum/data/IrrigationGIS/dads/rs/cdr/nc'
        cube_path = '/data/ssd2/dads_cube/cube.zarr'

    if Path(cdr_dir).exists():
        from cube.config import default_conus_config

        config = default_conus_config(
            cube_path=cube_path,
            cdr_dir=cdr_dir,
        )

        layer = CDRLayer(config)
        print(layer)
        print(f"Expected shape: {layer._get_expected_shape()}")
        print(f"Variables: {layer.variables}")
    else:
        print(f"CDR directory not found: {cdr_dir}")
