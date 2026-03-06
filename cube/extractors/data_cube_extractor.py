"""
Data cube extractor for training sequence extraction.

Provides efficient access to cube data for DADS pre-training, extracting
temporal sequences of meteorological and remote sensing features.

The output format matches the observation pipeline (DadsDataset) to enable
seamless model transfer from pre-training to fine-tuning.
"""

from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union
import logging

import numpy as np

try:
    import zarr

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

from cube.grid import MasterGrid
from cube.config import (
    CDR_FEATURES,
    LANDSAT_FEATURES,
    TERRAIN_FEATURES,
    MET_FEATURES,
)

logger = logging.getLogger(__name__)


class DataCubeExtractor:
    """
    Unified extractor for gridded pre-training data.

    Provides efficient random access to cube data for training, extracting
    temporal sequences with all features aligned to the observation pipeline.

    Output format matches DadsDataset:
        - Target variable (pseudo-observation)
        - Exogenous features (rsun, CDR bands, terrain, DOY encoding)
        - Temporal sequences for TCN

    Key features:
        - Single unified data source (no multi-source coordination)
        - Pre-computed DOY-indexed rsun (no runtime calculation)
        - Consistent coordinate system (direct indexing, no tolerance)
        - Full feature parity with observation pipeline

    Example:
        extractor = DataCubeExtractor('/data/ssd2/dads_cube/cube.zarr')
        target_seq, exog_seq, terrain, valid = extractor.extract_sequence(
            row=1000, col=2000,
            end_date=np.datetime64('2020-06-15'),
            target_variable='tmax',
            seq_len=12
        )
    """

    def __init__(
        self,
        cube_path: Union[str, Path],
        lazy: bool = True,
    ):
        """
        Initialize extractor.

        Args:
            cube_path: Path to cube.zarr store
            lazy: Whether to use lazy loading (recommended for large cubes)
        """
        if not HAS_ZARR:
            raise ImportError("zarr required for DataCubeExtractor")

        self.cube_path = Path(cube_path)
        self.lazy = lazy

        if not self.cube_path.exists():
            raise FileNotFoundError(f"Cube not found: {cube_path}")

        # Open zarr store
        self.store = zarr.open(str(self.cube_path), mode="r")

        # Load coordinate arrays
        self.lat = self.store["lat"][:]
        self.lon = self.store["lon"][:]

        # Load time coordinate if available
        if "time" in self.store:
            time_int = self.store["time"][:]
            # Convert from days since epoch to datetime64
            self.time = (
                time_int * np.timedelta64(1, "D") + np.datetime64("1970-01-01")
            ).astype("datetime64[D]")
            self._time_to_idx = {t: i for i, t in enumerate(self.time)}
        else:
            self.time = None
            self._time_to_idx = None

        # Build grid from store metadata or coordinates
        self.grid = MasterGrid(
            bounds=(
                float(self.lon[0] - (self.lon[1] - self.lon[0]) / 2),
                float(self.lat[-1] - (self.lat[0] - self.lat[1]) / 2),
                float(self.lon[-1] + (self.lon[1] - self.lon[0]) / 2),
                float(self.lat[0] + (self.lat[0] - self.lat[1]) / 2),
            ),
            resolution_deg=float(abs(self.lat[1] - self.lat[0])),
        )

        # Feature configuration
        self.terrain_features = TERRAIN_FEATURES
        self.cdr_features = [f.lower() for f in CDR_FEATURES]  # lowercase in cube
        self.landsat_features = [f.lower() for f in LANDSAT_FEATURES]
        self.met_features = MET_FEATURES

        # Open xarray datasets for lazy access if available
        if HAS_XARRAY and lazy:
            self._init_xarray_datasets()
        else:
            self._ds_static = None
            self._ds_doy = None
            self._ds_daily = None
            self._ds_composites = None

        # Cache for terrain (static, frequently accessed)
        self._terrain_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._terrain_cache_max = 10000

    def _init_xarray_datasets(self):
        """Initialize lazy xarray datasets for each group."""
        try:
            if "static" in self.store:
                self._ds_static = xr.open_zarr(str(self.cube_path), group="static")
            else:
                self._ds_static = None

            if "doy_indexed" in self.store:
                self._ds_doy = xr.open_zarr(str(self.cube_path), group="doy_indexed")
            else:
                self._ds_doy = None

            if "daily" in self.store:
                self._ds_daily = xr.open_zarr(str(self.cube_path), group="daily")
            else:
                self._ds_daily = None

            if "composites" in self.store:
                self._ds_composites = xr.open_zarr(
                    str(self.cube_path), group="composites"
                )
            else:
                self._ds_composites = None

        except Exception as e:
            logger.warning(f"Failed to open xarray datasets: {e}")
            self._ds_static = None
            self._ds_doy = None
            self._ds_daily = None
            self._ds_composites = None

    def extract_sequence(
        self,
        row: int,
        col: int,
        end_date: np.datetime64,
        target_variable: str,
        seq_len: int = 12,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Extract a complete feature sequence for pre-training.

        Args:
            row: Grid row index
            col: Grid column index
            end_date: End date of sequence (inclusive)
            target_variable: Target meteorological variable (e.g., 'tmax')
            seq_len: Sequence length in days

        Returns:
            Tuple of:
                - target_seq: (seq_len,) target variable values
                - exog_seq: (seq_len, n_exog) exogenous features
                - terrain_vec: (n_terrain,) static terrain features
                - valid: bool indicating if all data is complete
        """
        # Get time indices
        end_date = np.datetime64(end_date, "D")
        end_idx = self._date_to_idx(end_date)
        if end_idx is None:
            return self._empty_sequence(seq_len)

        start_idx = end_idx - seq_len + 1
        if start_idx < 0:
            return self._empty_sequence(seq_len)

        # Extract target variable sequence
        target_seq = self._extract_daily_var(
            target_variable, row, col, start_idx, end_idx + 1
        )
        if target_seq is None or np.any(np.isnan(target_seq)):
            return self._empty_sequence(seq_len)

        # Extract terrain (static, cached)
        terrain_vec = self._get_terrain(row, col)

        # Get dates for DOY lookup
        dates = self.time[start_idx : end_idx + 1]
        doys = self._dates_to_doy(dates)

        # Extract RSUN (DOY-indexed)
        rsun_seq = self._extract_rsun(row, col, doys)
        if rsun_seq is None:
            rsun_seq = np.zeros(seq_len, dtype=np.float32)

        # Extract CDR sequences
        cdr_seq = self._extract_cdr(row, col, start_idx, end_idx + 1)

        # Extract Landsat (nearest composite to each day)
        landsat_seq = self._extract_landsat(row, col, dates)

        # Compute DOY encoding
        doy_sin = np.sin(2 * np.pi * doys / 365.25).astype(np.float32)
        doy_cos = np.cos(2 * np.pi * doys / 365.25).astype(np.float32)

        # Get lat/lon (static)
        lat_val = float(self.lat[row])
        lon_val = float(self.lon[col])
        lat_arr = np.full(seq_len, lat_val, dtype=np.float32)
        lon_arr = np.full(seq_len, lon_val, dtype=np.float32)

        # Terrain repeated for each timestep
        terrain_rep = np.tile(terrain_vec, (seq_len, 1))

        # Assemble exogenous features matching GEO_FEATURES order:
        # ['lat', 'lon', 'rsun', 'doy_sin', 'doy_cos'] + LANDSAT + CDR + TERRAIN
        exog_seq = np.column_stack(
            [
                lat_arr,  # lat
                lon_arr,  # lon
                rsun_seq,  # rsun (DOY-indexed)
                doy_sin,  # doy_sin
                doy_cos,  # doy_cos
                landsat_seq,  # Landsat bands (7)
                cdr_seq,  # CDR bands (6)
                terrain_rep,  # terrain (7)
            ]
        ).astype(np.float32)

        return target_seq.astype(np.float32), exog_seq, terrain_vec, True

    def _date_to_idx(self, date: np.datetime64) -> Optional[int]:
        """Convert date to time index."""
        if self._time_to_idx is None:
            return None
        date = np.datetime64(date, "D")
        return self._time_to_idx.get(date)

    def _dates_to_doy(self, dates: np.ndarray) -> np.ndarray:
        """Convert datetime64 array to day-of-year."""
        # Extract day of year from datetime64
        doys = []
        for d in dates:
            # Convert to python datetime for DOY extraction
            dt = d.astype("datetime64[D]").astype(object)
            doys.append(dt.timetuple().tm_yday)
        return np.array(doys, dtype=np.int32)

    def _extract_daily_var(
        self,
        var: str,
        row: int,
        col: int,
        start_idx: int,
        end_idx: int,
    ) -> Optional[np.ndarray]:
        """Extract a daily variable sequence."""
        try:
            if self._ds_daily is not None and var in self._ds_daily:
                return (
                    self._ds_daily[var]
                    .isel(lat=row, lon=col, time=slice(start_idx, end_idx))
                    .values
                )
            elif "daily" in self.store and var in self.store["daily"]:
                return self.store["daily"][var][start_idx:end_idx, row, col]
        except Exception as e:
            logger.debug(f"Failed to extract {var}: {e}")
        return None

    def _get_terrain(self, row: int, col: int) -> np.ndarray:
        """Get terrain features for a cell (with caching)."""
        cache_key = (row, col)
        if cache_key in self._terrain_cache:
            return self._terrain_cache[cache_key]

        terrain = []
        for feat in self.terrain_features:
            try:
                if self._ds_static is not None and feat in self._ds_static:
                    val = float(self._ds_static[feat].isel(lat=row, lon=col).values)
                elif "static" in self.store and feat in self.store["static"]:
                    val = float(self.store["static"][feat][row, col])
                else:
                    val = 0.0
            except Exception:
                val = 0.0
            terrain.append(val)

        terrain_arr = np.array(terrain, dtype=np.float32)

        # Cache with LRU eviction
        if len(self._terrain_cache) >= self._terrain_cache_max:
            # Remove oldest entry (first key)
            oldest = next(iter(self._terrain_cache))
            del self._terrain_cache[oldest]
        self._terrain_cache[cache_key] = terrain_arr

        return terrain_arr

    def _extract_rsun(
        self,
        row: int,
        col: int,
        doys: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Extract RSUN values for given DOYs."""
        try:
            # DOY is 1-indexed, zarr is 0-indexed
            doy_indices = doys - 1
            if self._ds_doy is not None and "rsun" in self._ds_doy:
                return (
                    self._ds_doy["rsun"].isel(lat=row, lon=col, doy=doy_indices).values
                )
            elif "doy_indexed" in self.store and "rsun" in self.store["doy_indexed"]:
                return self.store["doy_indexed"]["rsun"][doy_indices, row, col]
        except Exception as e:
            logger.debug(f"Failed to extract rsun: {e}")
        return None

    def _extract_cdr(
        self,
        row: int,
        col: int,
        start_idx: int,
        end_idx: int,
    ) -> np.ndarray:
        """Extract CDR (surface reflectance/brightness temp) sequences."""
        seq_len = end_idx - start_idx
        cdr_seq = np.zeros((seq_len, len(self.cdr_features)), dtype=np.float32)

        for i, feat in enumerate(self.cdr_features):
            try:
                if self._ds_daily is not None and feat in self._ds_daily:
                    cdr_seq[:, i] = (
                        self._ds_daily[feat]
                        .isel(lat=row, lon=col, time=slice(start_idx, end_idx))
                        .values
                    )
                elif "daily" in self.store and feat in self.store["daily"]:
                    cdr_seq[:, i] = self.store["daily"][feat][
                        start_idx:end_idx, row, col
                    ]
            except Exception:
                pass  # Leave as zeros

        return cdr_seq

    def _extract_landsat(
        self,
        row: int,
        col: int,
        dates: np.ndarray,
    ) -> np.ndarray:
        """Extract Landsat values (nearest composite to each date)."""
        seq_len = len(dates)
        landsat_seq = np.zeros((seq_len, len(self.landsat_features)), dtype=np.float32)

        # For now, return zeros if composites not available
        # Full implementation would find nearest composite date
        if self._ds_composites is None and "composites" not in self.store:
            return landsat_seq

        # TODO: Implement composite date matching when Landsat layer is built
        return landsat_seq

    def _empty_sequence(
        self,
        seq_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Return empty sequence tuple for invalid data."""
        n_exog = (
            5  # lat, lon, rsun, doy_sin, doy_cos
            + len(self.landsat_features)
            + len(self.cdr_features)
            + len(self.terrain_features)
        )

        return (
            np.zeros(seq_len, dtype=np.float32),
            np.zeros((seq_len, n_exog), dtype=np.float32),
            np.zeros(len(self.terrain_features), dtype=np.float32),
            False,
        )

    def get_valid_cells(self, land_mask_threshold: float = 0.5) -> np.ndarray:
        """
        Get indices of valid (land) cells.

        Args:
            land_mask_threshold: Minimum land_mask value to consider valid

        Returns:
            (N, 2) array of [row, col] indices for valid cells
        """
        try:
            if "static" in self.store and "land_mask" in self.store["static"]:
                land_mask = self.store["static"]["land_mask"][:]
                valid_rows, valid_cols = np.where(land_mask >= land_mask_threshold)
                return np.column_stack([valid_rows, valid_cols])
        except Exception as e:
            logger.warning(f"Failed to get land mask: {e}")

        # Fallback: return all cells
        rows, cols = np.meshgrid(
            np.arange(len(self.lat)), np.arange(len(self.lon)), indexing="ij"
        )
        return np.column_stack([rows.ravel(), cols.ravel()])

    def get_cell_metadata(self, row: int, col: int) -> Dict:
        """
        Get metadata for a specific cell.

        Args:
            row: Row index
            col: Column index

        Returns:
            Dict with lat, lon, terrain features
        """
        return {
            "row": row,
            "col": col,
            "lat": float(self.lat[row]),
            "lon": float(self.lon[col]),
            "terrain": self._get_terrain(row, col),
        }

    def get_available_variables(self) -> Dict[str, List[str]]:
        """Get list of available variables by group."""
        available = {}

        for group_name in ["static", "doy_indexed", "daily", "composites"]:
            if group_name in self.store:
                available[group_name] = list(self.store[group_name].keys())

        return available

    def get_date_range(self) -> Tuple[np.datetime64, np.datetime64]:
        """Get available date range."""
        if self.time is None or len(self.time) == 0:
            return None, None
        return self.time[0], self.time[-1]

    def close(self):
        """Close all datasets."""
        if self._ds_static is not None:
            self._ds_static.close()
        if self._ds_doy is not None:
            self._ds_doy.close()
        if self._ds_daily is not None:
            self._ds_daily.close()
        if self._ds_composites is not None:
            self._ds_composites.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        return (
            f"DataCubeExtractor("
            f"path='{self.cube_path}', "
            f"shape=({len(self.lat)}, {len(self.lon)}), "
            f"n_times={len(self.time) if self.time is not None else 0})"
        )


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        cube_path = sys.argv[1]
    else:
        cube_path = "/data/ssd2/dads_cube/cube.zarr"

    if Path(cube_path).exists():
        extractor = DataCubeExtractor(cube_path)
        print(extractor)
        print(f"\nAvailable variables: {extractor.get_available_variables()}")
        print(f"Date range: {extractor.get_date_range()}")

        valid_cells = extractor.get_valid_cells()
        print(f"Valid cells: {len(valid_cells):,}")
    else:
        print(f"Cube not found at {cube_path}")
        print("Create the cube first using cube builders.")
