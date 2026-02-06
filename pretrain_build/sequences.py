"""
Extracts temporal sequences from gridded data sources.

Key insight: rather than pre-extracting to per-cell parquets (expensive),
we use xarray's lazy loading with Zarr to extract sequences on-the-fly.

For hourly data (ERA5, NLDAS), we aggregate to daily before extracting.

The sequence format matches DadsDataset:
    - Target column (pseudo-observation)
    - Exogenous columns (rsun, doy_sin, doy_cos, terrain features)
"""
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from functools import lru_cache
from datetime import datetime

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

from pretrain_build.config import PretrainConfig, GridSource


class SequenceExtractor:
    """
    Extracts temporal sequences from gridded Zarr/NetCDF sources.

    Maintains open dataset handles for efficient repeated access.
    Caches recently accessed data for performance.

    The extractor produces sequences matching the DadsDataset format:
        - Column 0: target variable (pseudo-observation)
        - Columns 1+: exogenous features (rsun, doy_sin, doy_cos, etc.)
    """

    def __init__(
        self,
        sources: Dict[str, GridSource],
        config: PretrainConfig,
    ):
        """
        Initialize extractor with data sources.

        Args:
            sources: Dict mapping source name to GridSource config
            config: PretrainConfig with temporal parameters
        """
        if not HAS_XARRAY:
            raise ImportError("xarray is required for SequenceExtractor")

        self.sources = sources
        self.config = config
        self._datasets: Dict[str, xr.Dataset] = {}
        self._open_sources()

    def _open_sources(self):
        """Open all data sources as lazy xarray Datasets."""
        for name, src in self.sources.items():
            try:
                if src.zarr_path and src.zarr_path.exists():
                    self._datasets[name] = xr.open_zarr(src.zarr_path)
                    print(f"[SequenceExtractor] Opened Zarr: {src.zarr_path}")
                elif src.netcdf_dir and src.netcdf_dir.exists():
                    nc_files = sorted(src.netcdf_dir.glob('*.nc'))
                    if nc_files:
                        self._datasets[name] = xr.open_mfdataset(
                            nc_files,
                            combine='by_coords',
                            chunks={'time': 365},
                        )
                        print(f"[SequenceExtractor] Opened {len(nc_files)} NetCDF files from {src.netcdf_dir}")
            except Exception as e:
                print(f"[SequenceExtractor] Failed to open {name}: {e}")

    def close(self):
        """Close all open datasets."""
        for name, ds in self._datasets.items():
            try:
                ds.close()
            except:
                pass
        self._datasets.clear()

    def __del__(self):
        self.close()

    def extract_sequence(
        self,
        lat: float,
        lon: float,
        end_date: np.datetime64,
        variable: str,
        terrain_vec: np.ndarray,
        row: Optional[int] = None,
        col: Optional[int] = None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Extract a seq_len-day sequence ending at end_date.

        Args:
            lat: Latitude of target cell
            lon: Longitude of target cell
            end_date: End date of sequence (inclusive)
            variable: Target variable name (e.g., 'tmax', 'rsds')
            terrain_vec: Pre-extracted terrain features for this cell
            row: Optional grid row index (for faster indexing)
            col: Optional grid col index (for faster indexing)

        Returns:
            (sequence, valid) tuple where:
                sequence: (seq_len, n_features) array
                valid: whether the sequence has complete data

        Features layout:
            [0]: target variable (pseudo-observation)
            [1]: rsun (computed from lat/doy)
            [2]: doy_sin
            [3]: doy_cos
            [4:]: terrain features (static, repeated)
        """
        seq_len = self.config.seq_len
        n_terrain = len(terrain_vec)
        n_features = 1 + 3 + n_terrain  # target + rsun + doy_sin/cos + terrain

        # Compute date range
        end_dt = end_date.astype('datetime64[D]')
        start_dt = end_dt - np.timedelta64(seq_len - 1, 'D')

        # Find source for this variable
        source_name = self._variable_to_source(variable)
        if source_name is None or source_name not in self._datasets:
            return np.zeros((seq_len, n_features), dtype=np.float32), False

        ds = self._datasets[source_name]
        src = self.sources[source_name]

        # Map variable name to source's variable name
        src_var = self._map_variable_name(variable, source_name)
        if src_var not in ds:
            return np.zeros((seq_len, n_features), dtype=np.float32), False

        try:
            # Select data at nearest grid cell
            if row is not None and col is not None:
                # Use grid indices for faster access
                cell_data = ds.isel({src.lat_var: row, src.lon_var: col})
            else:
                # Use lat/lon selection (slower but more flexible)
                cell_data = ds.sel(
                    {src.lat_var: lat, src.lon_var: lon},
                    method='nearest'
                )

            # Select time range
            time_data = cell_data.sel(
                {src.time_var: slice(str(start_dt), str(end_dt))}
            )

            # Load target variable
            var_data = time_data[src_var].values

            # Handle case where we got different length than expected
            if len(var_data) != seq_len:
                return np.zeros((seq_len, n_features), dtype=np.float32), False

            # Check for missing values
            if np.any(np.isnan(var_data)):
                return np.zeros((seq_len, n_features), dtype=np.float32), False

        except Exception as e:
            return np.zeros((seq_len, n_features), dtype=np.float32), False

        # Compute exogenous features
        # DOY from dates
        dates = np.arange(start_dt, end_dt + np.timedelta64(1, 'D'), dtype='datetime64[D]')
        doys = _compute_doy(dates)

        # rsun (clear-sky irradiance)
        rsun = _compute_rsun(lat, doys)

        # DOY encoding
        doy_sin = np.sin(2 * np.pi * doys / 365.25)
        doy_cos = np.cos(2 * np.pi * doys / 365.25)

        # Terrain (static, repeat across time)
        terrain_rep = np.tile(terrain_vec, (seq_len, 1))

        # Assemble sequence
        sequence = np.column_stack([
            var_data.astype(np.float32),
            rsun.astype(np.float32),
            doy_sin.astype(np.float32),
            doy_cos.astype(np.float32),
            terrain_rep.astype(np.float32),
        ])

        return sequence, True

    def _variable_to_source(self, variable: str) -> Optional[str]:
        """Map a variable name to its source dataset."""
        return self.config.variable_map.get(variable)

    def _map_variable_name(self, variable: str, source_name: str) -> str:
        """
        Map canonical variable name to source-specific name.

        Different sources use different variable names for the same quantity.
        """
        # Variable name mappings for common sources
        mappings = {
            'prism': {
                'tmax': 'tmax',
                'tmin': 'tmin',
                'ppt': 'ppt',
            },
            'gridmet': {
                'tmax': 'tmmx',
                'tmin': 'tmmn',
                'rsds': 'srad',
                'srad': 'srad',
                'vpd': 'vpd',
                'wind': 'vs',
                'ea': 'etr',  # Reference ET as proxy
            },
            'era5': {
                'tmax': 't2m',
                'tmin': 't2m',
                'rsds': 'ssrd',
            },
        }

        if source_name in mappings:
            return mappings[source_name].get(variable, variable)
        return variable

    def get_available_dates(self, source_name: str) -> Tuple[np.datetime64, np.datetime64]:
        """
        Get the available date range for a source.

        Args:
            source_name: Name of the data source

        Returns:
            (start_date, end_date) tuple
        """
        if source_name not in self._datasets:
            raise ValueError(f"Source {source_name} not loaded")

        ds = self._datasets[source_name]
        src = self.sources[source_name]

        times = ds[src.time_var].values
        return np.datetime64(times[0], 'D'), np.datetime64(times[-1], 'D')


def _compute_doy(dates: np.ndarray) -> np.ndarray:
    """Compute day of year from datetime64 array."""
    # Convert to Python dates and extract DOY
    doys = []
    for d in dates:
        dt = d.astype('datetime64[D]').astype(datetime)
        doys.append(dt.timetuple().tm_yday)
    return np.array(doys)


def _compute_rsun(lat: float, doys: np.ndarray) -> np.ndarray:
    """
    Compute clear-sky solar irradiance (MJ/m²/day) for given lat and DOYs.

    Uses simplified FAO-56 method for extraterrestrial radiation.

    Args:
        lat: Latitude in degrees
        doys: Array of day-of-year values (1-365)

    Returns:
        Array of clear-sky irradiance values
    """
    lat_rad = np.radians(lat)
    doys = np.asarray(doys)

    # Solar declination (radians)
    decl = 0.409 * np.sin(2 * np.pi / 365 * doys - 1.39)

    # Sunset hour angle (handle edge cases at high latitudes)
    cos_ws = -np.tan(lat_rad) * np.tan(decl)
    cos_ws = np.clip(cos_ws, -1.0, 1.0)
    ws = np.arccos(cos_ws)

    # Inverse relative distance Earth-Sun
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * doys)

    # Extraterrestrial radiation (MJ/m²/day)
    Gsc = 0.0820  # Solar constant (MJ/m²/min)
    Ra = (24 * 60 / np.pi) * Gsc * dr * (
        ws * np.sin(lat_rad) * np.sin(decl) +
        np.cos(lat_rad) * np.cos(decl) * np.sin(ws)
    )

    # Clear-sky radiation (Rs0 ≈ 0.75 * Ra for sea level)
    # This is a simplification; could add elevation adjustment
    Rs0 = 0.75 * Ra

    return np.maximum(Rs0, 0.0)


class CachedSequenceExtractor(SequenceExtractor):
    """
    Sequence extractor with LRU caching for repeated cell access.

    Useful when the same cells are accessed multiple times during
    an epoch (e.g., as both targets and neighbors).
    """

    def __init__(
        self,
        sources: Dict[str, GridSource],
        config: PretrainConfig,
        cache_size: int = 10000,
    ):
        super().__init__(sources, config)
        self.cache_size = cache_size
        self._cache: Dict[Tuple, Tuple[np.ndarray, bool]] = {}
        self._cache_order: List[Tuple] = []

    def extract_sequence(
        self,
        lat: float,
        lon: float,
        end_date: np.datetime64,
        variable: str,
        terrain_vec: np.ndarray,
        row: Optional[int] = None,
        col: Optional[int] = None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Extract sequence with caching.

        Cache key is (row, col, end_date, variable) for grid-indexed access,
        or (lat, lon, end_date, variable) for coordinate access.
        """
        if row is not None and col is not None:
            cache_key = (row, col, str(end_date), variable)
        else:
            cache_key = (round(lat, 6), round(lon, 6), str(end_date), variable)

        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Extract and cache
        result = super().extract_sequence(
            lat, lon, end_date, variable, terrain_vec, row, col
        )

        # Add to cache with LRU eviction
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry
            old_key = self._cache_order.pop(0)
            del self._cache[old_key]

        self._cache[cache_key] = result
        self._cache_order.append(cache_key)

        return result

    def clear_cache(self):
        """Clear the sequence cache."""
        self._cache.clear()
        self._cache_order.clear()


class ParquetSequenceExtractor:
    """
    Alternative extractor that reads from pre-extracted parquet files.

    This is useful when gridded data has already been extracted to
    per-cell parquet files (similar to the observation pipeline).
    """

    def __init__(
        self,
        parquet_dir: Path,
        config: PretrainConfig,
        variable: str,
    ):
        """
        Initialize extractor with parquet directory.

        Args:
            parquet_dir: Directory containing cell parquet files
            config: PretrainConfig
            variable: Target variable name
        """
        import pandas as pd

        self.parquet_dir = Path(parquet_dir)
        self.config = config
        self.variable = variable
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def extract_sequence(
        self,
        cell_id: str,
        end_date: np.datetime64,
        terrain_vec: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """
        Extract sequence from parquet file.

        Args:
            cell_id: Cell identifier (parquet filename without extension)
            end_date: End date of sequence
            terrain_vec: Terrain features for this cell

        Returns:
            (sequence, valid) tuple
        """
        import pandas as pd

        seq_len = self.config.seq_len
        n_terrain = len(terrain_vec)
        n_features = 1 + 3 + n_terrain

        # Load from cache or file
        if cell_id not in self._cache:
            parquet_path = self.parquet_dir / f"{cell_id}.parquet"
            if not parquet_path.exists():
                return np.zeros((seq_len, n_features), dtype=np.float32), False

            try:
                df = pd.read_parquet(parquet_path)
                values = df[self.variable].values
                dates = df.index.to_numpy()
                self._cache[cell_id] = (values, dates)
            except Exception:
                return np.zeros((seq_len, n_features), dtype=np.float32), False

        values, dates = self._cache[cell_id]

        # Find end date index
        end_dt = end_date.astype('datetime64[D]')
        end_idx = np.searchsorted(dates, end_dt)

        if end_idx < seq_len - 1 or end_idx >= len(dates):
            return np.zeros((seq_len, n_features), dtype=np.float32), False

        # Check consecutive days
        start_idx = end_idx - seq_len + 1
        window_dates = dates[start_idx:end_idx + 1]
        if not np.all(np.diff(window_dates.astype('datetime64[D]').astype(int)) == 1):
            return np.zeros((seq_len, n_features), dtype=np.float32), False

        var_data = values[start_idx:end_idx + 1]
        if np.any(np.isnan(var_data)):
            return np.zeros((seq_len, n_features), dtype=np.float32), False

        # Compute exogenous features
        doys = _compute_doy(window_dates)
        # Get lat from cached data if available, otherwise estimate from cell_id
        lat = 40.0  # Default - should be passed in or extracted from parquet metadata
        rsun = _compute_rsun(lat, doys)
        doy_sin = np.sin(2 * np.pi * doys / 365.25)
        doy_cos = np.cos(2 * np.pi * doys / 365.25)
        terrain_rep = np.tile(terrain_vec, (seq_len, 1))

        sequence = np.column_stack([
            var_data,
            rsun,
            doy_sin,
            doy_cos,
            terrain_rep,
        ]).astype(np.float32)

        return sequence, True


if __name__ == '__main__':
    # Example usage
    print("SequenceExtractor module - no standalone execution")
    print("Use with GridIndex and EpochSampler for sequence extraction")
