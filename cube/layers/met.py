"""
Meteorological layer builder for the data cube.

Builds daily time series of meteorological variables from gridded sources
(NLDAS, PRISM, GridMET).

Variables:
    - tmax: Maximum temperature (K or °C)
    - tmin: Minimum temperature (K or °C)
    - prcp: Precipitation (mm/day)
    - rsds: Surface downwelling shortwave radiation (W/m²)
    - vpd: Vapor pressure deficit (kPa)
    - wind: Wind speed (m/s)
    - ea: Actual vapor pressure (kPa)

Source priority:
    1. NLDAS (hourly -> daily aggregation, 12km)
    2. GridMET (daily, 4km, CONUS only)
    3. PRISM (daily, 4km, CONUS only)

Each source provides different variable subsets, and the layer merges them.
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
from cube.config import CubeConfig, CHUNKS, MET_FEATURES

logger = logging.getLogger(__name__)

# Variable mappings from sources to canonical names
NLDAS_VARIABLES = {
    'TMP': 'tmax',       # Temperature (2m) - aggregate to max
    'SPFH': 'ea',        # Specific humidity -> actual vapor pressure
    'PRES': 'pres',      # Pressure (for ea calculation)
    'UGRD': 'u_wind',    # U-component wind
    'VGRD': 'v_wind',    # V-component wind
    'DLWRF': 'dlwrf',    # Downwelling longwave
    'DSWRF': 'rsds',     # Downwelling shortwave
    'APCP': 'prcp',      # Precipitation
}

GRIDMET_VARIABLES = {
    'tmmx': 'tmax',      # Max temperature
    'tmmn': 'tmin',      # Min temperature
    'vs': 'wind',        # Wind speed
    'srad': 'rsds',      # Shortwave radiation
    'vpd': 'vpd',        # Vapor pressure deficit
    'pr': 'prcp',        # Precipitation
    'rmax': 'rh_max',    # Max relative humidity
    'rmin': 'rh_min',    # Min relative humidity
    'sph': 'ea',         # Specific humidity
}

PRISM_VARIABLES = {
    'tmax': 'tmax',
    'tmin': 'tmin',
    'ppt': 'prcp',
}


class MetLayer(BaseLayer):
    """
    Builds daily meteorological variable layer from gridded sources.

    Supports multiple input sources (NLDAS, GridMET, PRISM) with automatic
    variable mapping and unit conversion.

    Output: (time, lat, lon) arrays for tmax, tmin, prcp, rsds, vpd, wind, ea
    """

    @property
    def name(self) -> str:
        return 'daily'

    @property
    def variables(self) -> List[str]:
        return list(MET_FEATURES)  # tmax, tmin, prcp, rsds, vpd, wind, ea

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
        sources: Optional[List[str]] = None,
    ) -> None:
        """
        Build meteorological layer from gridded sources.

        Args:
            source_paths: Dict with keys like 'nldas_dir', 'gridmet_dir', 'prism_dir'
            overwrite: Whether to overwrite existing data
            start_date: Start date (default: config.start_date)
            end_date: End date (default: config.end_date)
            sources: List of sources to use (default: ['gridmet', 'prism', 'nldas'])
        """
        if not HAS_XARRAY:
            raise ImportError("xarray required for met layer building")

        source_paths = source_paths or self.config.source_paths
        sources = sources or ['gridmet', 'prism', 'nldas']

        # Date range
        start = pd.Timestamp(start_date or self.config.start_date)
        end = pd.Timestamp(end_date or self.config.end_date)

        logger.info(f"Building meteorological layer")
        logger.info(f"Date range: {start.date()} to {end.date()}")
        logger.info(f"Sources: {sources}")

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

        from cube.builders.resampler import GridResampler
        resampler = GridResampler(self.grid)

        # Process each source
        for source_name in sources:
            source_key = f'{source_name}_dir'
            if source_key not in source_paths:
                logger.warning(f"No path for {source_name}, skipping")
                continue

            source_dir = Path(source_paths[source_key])
            if not source_dir.exists():
                logger.warning(f"{source_name} directory not found: {source_dir}")
                continue

            logger.info(f"Processing {source_name} from {source_dir}...")

            if source_name == 'gridmet':
                self._process_gridmet(source_dir, arrays, dates, resampler)
            elif source_name == 'prism':
                self._process_prism(source_dir, arrays, dates, resampler)
            elif source_name == 'nldas':
                self._process_nldas(source_dir, arrays, dates, resampler)

        # Store metadata
        store.attrs['met_sources'] = str(sources)
        store.attrs['met_start_date'] = str(start.date())
        store.attrs['met_end_date'] = str(end.date())

        logger.info("Meteorological layer complete")

    def _process_gridmet(
        self,
        source_dir: Path,
        arrays: Dict[str, zarr.Array],
        dates: pd.DatetimeIndex,
        resampler: 'GridResampler',
    ) -> None:
        """Process GridMET NetCDF files."""
        # GridMET has one file per variable per year: {var}_{year}.nc
        for year in range(dates[0].year, dates[-1].year + 1):
            year_start = max(dates[0], pd.Timestamp(year, 1, 1))
            year_end = min(dates[-1], pd.Timestamp(year, 12, 31))

            if year_start > year_end:
                continue

            logger.info(f"  GridMET {year}...")

            for gm_var, out_var in GRIDMET_VARIABLES.items():
                if out_var not in arrays:
                    continue

                nc_file = source_dir / f'{gm_var}_{year}.nc'
                if not nc_file.exists():
                    continue

                try:
                    self._process_gridmet_file(
                        nc_file, gm_var, out_var, arrays,
                        dates, year_start, year_end, resampler
                    )
                except Exception as e:
                    logger.warning(f"Error processing {nc_file}: {e}")

    def _process_gridmet_file(
        self,
        nc_file: Path,
        gm_var: str,
        out_var: str,
        arrays: Dict[str, zarr.Array],
        dates: pd.DatetimeIndex,
        year_start: pd.Timestamp,
        year_end: pd.Timestamp,
        resampler: 'GridResampler',
    ) -> None:
        """Process a single GridMET file."""
        ds = xr.open_dataset(nc_file)

        # Rename time dimension if needed
        time_var = 'day' if 'day' in ds.dims else 'time'

        # Get variable data
        data = ds[gm_var]

        # Select time range
        data = data.sel({time_var: slice(str(year_start.date()), str(year_end.date()))})

        if len(data[time_var]) == 0:
            ds.close()
            return

        # Get coordinates
        lat_var = 'lat' if 'lat' in ds.coords else 'y'
        lon_var = 'lon' if 'lon' in ds.coords else 'x'

        lat = ds[lat_var].values
        lon = ds[lon_var].values

        # Create source transform
        import rasterio
        lat_res = abs(lat[1] - lat[0]) if len(lat) > 1 else 0.04
        lon_res = abs(lon[1] - lon[0]) if len(lon) > 1 else 0.04
        src_transform = rasterio.Affine(
            lon_res, 0, lon.min() - lon_res / 2,
            0, -lat_res, lat.max() + lat_res / 2
        )

        # Convert units if needed
        values = data.values
        if out_var in ('tmax', 'tmin') and values.max() > 200:
            # Convert K to C
            values = values - 273.15

        # Resample each timestep
        times = pd.to_datetime(data[time_var].values)
        for i, t in enumerate(times):
            t_idx = (t - dates[0]).days
            if 0 <= t_idx < len(dates):
                resampled = resampler.resample_array(
                    values[i],
                    src_transform=src_transform,
                    src_crs='EPSG:4326',
                    method='bilinear',
                )
                # Only write if we don't have data yet (from higher priority source)
                existing = arrays[out_var][t_idx, :, :]
                mask = np.isnan(existing)
                if np.any(mask):
                    updated = np.where(mask, resampled.astype(np.float32), existing)
                    arrays[out_var][t_idx, :, :] = updated

        ds.close()

    def _process_prism(
        self,
        source_dir: Path,
        arrays: Dict[str, zarr.Array],
        dates: pd.DatetimeIndex,
        resampler: 'GridResampler',
    ) -> None:
        """Process PRISM NetCDF files."""
        # PRISM files can be organized as: {var}_{YYYYMM}.nc or similar
        for year in range(dates[0].year, dates[-1].year + 1):
            for month in range(1, 13):
                month_start = pd.Timestamp(year, month, 1)
                if month_start < dates[0] or month_start > dates[-1]:
                    continue

                logger.info(f"  PRISM {year}-{month:02d}...")

                for prism_var, out_var in PRISM_VARIABLES.items():
                    if out_var not in arrays:
                        continue

                    # Try different file naming conventions
                    patterns = [
                        f'{prism_var}_{year}{month:02d}.nc',
                        f'PRISM_{prism_var}_{year}{month:02d}.nc',
                        f'{prism_var}_{year}_{month:02d}.nc',
                    ]

                    for pattern in patterns:
                        nc_file = source_dir / pattern
                        if nc_file.exists():
                            try:
                                self._process_prism_file(
                                    nc_file, prism_var, out_var, arrays,
                                    dates, resampler
                                )
                            except Exception as e:
                                logger.warning(f"Error processing {nc_file}: {e}")
                            break

    def _process_prism_file(
        self,
        nc_file: Path,
        prism_var: str,
        out_var: str,
        arrays: Dict[str, zarr.Array],
        dates: pd.DatetimeIndex,
        resampler: 'GridResampler',
    ) -> None:
        """Process a single PRISM file."""
        ds = xr.open_dataset(nc_file)

        # Get variable (may have different name)
        var_name = prism_var
        if var_name not in ds:
            for v in ds.data_vars:
                if prism_var in v.lower():
                    var_name = v
                    break

        if var_name not in ds:
            ds.close()
            return

        data = ds[var_name]

        # Get coordinates
        lat = ds['lat'].values if 'lat' in ds.coords else ds['y'].values
        lon = ds['lon'].values if 'lon' in ds.coords else ds['x'].values

        # Create source transform
        import rasterio
        lat_res = abs(lat[1] - lat[0]) if len(lat) > 1 else 0.04
        lon_res = abs(lon[1] - lon[0]) if len(lon) > 1 else 0.04
        src_transform = rasterio.Affine(
            lon_res, 0, lon.min() - lon_res / 2,
            0, -lat_res, lat.max() + lat_res / 2
        )

        # Get time dimension
        time_var = 'time' if 'time' in ds.dims else 'day'
        times = pd.to_datetime(ds[time_var].values)

        values = data.values

        for i, t in enumerate(times):
            t_idx = (t - dates[0]).days
            if 0 <= t_idx < len(dates):
                resampled = resampler.resample_array(
                    values[i],
                    src_transform=src_transform,
                    src_crs='EPSG:4326',
                    method='bilinear',
                )
                existing = arrays[out_var][t_idx, :, :]
                mask = np.isnan(existing)
                if np.any(mask):
                    updated = np.where(mask, resampled.astype(np.float32), existing)
                    arrays[out_var][t_idx, :, :] = updated

        ds.close()

    def _process_nldas(
        self,
        source_dir: Path,
        arrays: Dict[str, zarr.Array],
        dates: pd.DatetimeIndex,
        resampler: 'GridResampler',
    ) -> None:
        """Process NLDAS NetCDF files (hourly -> daily aggregation)."""
        # NLDAS files: NLDAS_FORA0125_H.A{YYYYMMDD}.{HHMM}.*.nc

        for year in range(dates[0].year, dates[-1].year + 1):
            for month in range(1, 13):
                month_start = pd.Timestamp(year, month, 1)
                if month_start < dates[0] or month_start > dates[-1]:
                    continue

                logger.info(f"  NLDAS {year}-{month:02d}...")

                try:
                    self._process_nldas_month(
                        source_dir, year, month, arrays, dates, resampler
                    )
                except Exception as e:
                    logger.warning(f"Error processing NLDAS {year}-{month:02d}: {e}")

    def _process_nldas_month(
        self,
        source_dir: Path,
        year: int,
        month: int,
        arrays: Dict[str, zarr.Array],
        dates: pd.DatetimeIndex,
        resampler: 'GridResampler',
    ) -> None:
        """Process a month of NLDAS hourly data."""
        date_string = f'{year}{month:02d}'

        # Find hourly files for this month
        nc_files = sorted(source_dir.glob(f'NLDAS_FORA0125_H.A{date_string}*.nc'))

        if not nc_files:
            # Try alternative pattern
            nc_files = sorted(source_dir.glob(f'*{date_string}*.nc'))

        if not nc_files:
            return

        # Open as single dataset
        ds = xr.open_mfdataset(nc_files, combine='by_coords')

        # Aggregate hourly to daily
        daily = self._aggregate_nldas_daily(ds)

        # Get coordinates
        lat = ds['lat'].values
        lon = ds['lon'].values

        # Create source transform
        import rasterio
        lat_res = abs(lat[1] - lat[0]) if len(lat) > 1 else 0.125
        lon_res = abs(lon[1] - lon[0]) if len(lon) > 1 else 0.125
        src_transform = rasterio.Affine(
            lon_res, 0, lon.min() - lon_res / 2,
            0, -lat_res, lat.max() + lat_res / 2
        )

        # Process each output variable
        for out_var, daily_data in daily.items():
            if out_var not in arrays:
                continue

            times = daily_data['time']
            values = daily_data['values']

            for i, t in enumerate(times):
                t_idx = (t - dates[0]).days
                if 0 <= t_idx < len(dates):
                    resampled = resampler.resample_array(
                        values[i],
                        src_transform=src_transform,
                        src_crs='EPSG:4326',
                        method='bilinear',
                    )
                    existing = arrays[out_var][t_idx, :, :]
                    mask = np.isnan(existing)
                    if np.any(mask):
                        updated = np.where(mask, resampled.astype(np.float32), existing)
                        arrays[out_var][t_idx, :, :] = updated

        ds.close()

    def _aggregate_nldas_daily(self, ds: xr.Dataset) -> Dict[str, Dict]:
        """
        Aggregate NLDAS hourly data to daily values.

        Returns dict of {var_name: {'time': times, 'values': values}}
        """
        result = {}

        # Group by date
        time_values = pd.to_datetime(ds['time'].values)
        dates = time_values.date
        unique_dates = np.unique(dates)

        for out_var in self.variables:
            if out_var == 'tmax' and 'TMP' in ds:
                # Daily maximum temperature
                values = []
                for d in unique_dates:
                    mask = dates == d
                    day_data = ds['TMP'].isel(time=mask).max(dim='time').values
                    values.append(day_data - 273.15)  # K to C
                result['tmax'] = {
                    'time': pd.to_datetime(unique_dates),
                    'values': np.array(values)
                }

            elif out_var == 'tmin' and 'TMP' in ds:
                # Daily minimum temperature
                values = []
                for d in unique_dates:
                    mask = dates == d
                    day_data = ds['TMP'].isel(time=mask).min(dim='time').values
                    values.append(day_data - 273.15)  # K to C
                result['tmin'] = {
                    'time': pd.to_datetime(unique_dates),
                    'values': np.array(values)
                }

            elif out_var == 'rsds' and 'DSWRF' in ds:
                # Daily mean shortwave radiation
                values = []
                for d in unique_dates:
                    mask = dates == d
                    day_data = ds['DSWRF'].isel(time=mask).mean(dim='time').values
                    values.append(day_data)
                result['rsds'] = {
                    'time': pd.to_datetime(unique_dates),
                    'values': np.array(values)
                }

            elif out_var == 'prcp' and 'APCP' in ds:
                # Daily total precipitation
                values = []
                for d in unique_dates:
                    mask = dates == d
                    day_data = ds['APCP'].isel(time=mask).sum(dim='time').values
                    values.append(day_data)
                result['prcp'] = {
                    'time': pd.to_datetime(unique_dates),
                    'values': np.array(values)
                }

            elif out_var == 'wind' and 'UGRD' in ds and 'VGRD' in ds:
                # Daily mean wind speed from u/v components
                values = []
                for d in unique_dates:
                    mask = dates == d
                    u = ds['UGRD'].isel(time=mask).mean(dim='time').values
                    v = ds['VGRD'].isel(time=mask).mean(dim='time').values
                    wind_speed = np.sqrt(u**2 + v**2)
                    values.append(wind_speed)
                result['wind'] = {
                    'time': pd.to_datetime(unique_dates),
                    'values': np.array(values)
                }

        return result

    def validate(self) -> Dict[str, bool]:
        """Validate met layer with physical checks."""
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
                sample = data[0:365, ::10, ::10]
                valid = sample[~np.isnan(sample)]

                if len(valid) == 0:
                    checks[f'{var}_has_data'] = False
                    continue

                # Physical range checks
                if var in ('tmax', 'tmin'):
                    # Temperature: -60 to 60 C
                    checks[f'{var}_range'] = valid.min() > -70 and valid.max() < 70
                elif var == 'prcp':
                    # Precipitation: >= 0
                    checks[f'{var}_range'] = valid.min() >= 0
                elif var == 'rsds':
                    # Radiation: 0 to 1400 W/m²
                    checks[f'{var}_range'] = valid.min() >= 0 and valid.max() < 1500
                elif var == 'vpd':
                    # VPD: 0 to 10 kPa
                    checks[f'{var}_range'] = valid.min() >= 0 and valid.max() < 15
                elif var == 'wind':
                    # Wind: 0 to 50 m/s
                    checks[f'{var}_range'] = valid.min() >= 0 and valid.max() < 60
                elif var == 'ea':
                    # Actual vapor pressure: 0 to 7 kPa
                    checks[f'{var}_range'] = valid.min() >= 0 and valid.max() < 10

        except Exception as e:
            logger.error(f"Validation error: {e}")
            checks['validation_error'] = False

        return checks


if __name__ == '__main__':
    import sys

    cube_path = '/data/ssd2/dads_cube/cube.zarr'

    from cube.config import default_conus_config

    config = default_conus_config(
        cube_path=cube_path,
        nldas_dir='/data/ssd1/nldas2/netcdf/',
        gridmet_dir='/data/gridmet/',
        prism_dir='/data/prism/netcdf/',
    )

    layer = MetLayer(config)
    print(layer)
    print(f"Expected shape: {layer._get_expected_shape()}")
    print(f"Variables: {layer.variables}")
