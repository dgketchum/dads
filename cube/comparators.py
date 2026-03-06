"""
Station-level comparator loading for DADS validation.

Loads gridded meteorological product values at station locations from existing
processed parquets. Provides alignment with observation time series and metric
computation for evaluating DADS predictions against established products.

No gridded zarr store — all comparisons are station-to-station using the
per-station parquets already produced by `process/gridded/proc_gridded.py`.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SOURCES = ["gridmet", "nldas2", "prism"]

# Variables available per source (from processed parquets)
SOURCE_VARIABLES = {
    "gridmet": ["tmax", "tmin", "rsds", "prcp", "vpd", "wind", "ea", "eto"],
    "nldas2": ["tmax", "tmin", "rsds", "prcp", "vpd", "wind", "ea", "eto"],
    "prism": ["tmax", "tmin", "prcp", "vpd", "tmean"],
    "era5land": ["tmax", "tmin", "rsds", "prcp", "vpd", "wind", "ea", "eto"],
}

# File patterns per source
_FILE_PATTERNS = {
    "gridmet": "{fid}.parquet",
    "nldas2": "{fid}.parquet",
    "prism": "{fid}.parquet",
    "era5land": "{fid}.parquet",
}


class StationComparators:
    """Load gridded product values at station locations for comparison.

    Reads from processed parquets produced by `process/gridded/proc_gridded.py`.
    Each source has a subdirectory under the root containing one parquet per station.

    Example layout::

        processed_parquet_root/
        ├── gridmet/
        │   ├── 0001P.parquet
        │   └── ...
        ├── nldas2/
        │   ├── 0001P.parquet
        │   └── ...
        └── prism/
            ├── 0001P.parquet
            └── ...
    """

    def __init__(
        self,
        processed_parquet_root: Union[str, Path],
        sources: Optional[List[str]] = None,
    ):
        """
        Args:
            processed_parquet_root: Root directory containing source subdirectories.
            sources: Which sources to use. Defaults to ['gridmet', 'nldas2', 'prism'].
        """
        self.root = Path(processed_parquet_root)
        self.sources = sources or list(DEFAULT_SOURCES)

    def _parquet_path(self, source: str, fid: str) -> Path:
        """Build the parquet path for a given source and station."""
        pattern = _FILE_PATTERNS.get(source, "{fid}.parquet")
        return self.root / source / pattern.format(fid=fid)

    def _read_source(
        self,
        source: str,
        fid: str,
        variable: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.Series]:
        """Read a single variable from one source for one station.

        Returns:
            Series with DatetimeIndex, or None if unavailable.
        """
        if variable not in SOURCE_VARIABLES.get(source, []):
            return None

        path = self._parquet_path(source, fid)
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            logger.warning(f"Failed to read {path}: {exc}")
            return None

        if variable not in df.columns:
            return None

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        s = df[variable].astype(np.float32)
        s.name = source

        if start_date is not None:
            s = s.loc[start_date:]
        if end_date is not None:
            s = s.loc[:end_date]

        return s

    def load_station(
        self,
        fid: str,
        variable: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load comparator values for one station, one variable, all sources.

        Args:
            fid: Station identifier (e.g. '0001P', 'USC00456789').
            variable: Canonical variable name (tmax, tmin, rsds, prcp, vpd, wind, ea).
            start_date: Optional start date string for slicing (inclusive).
            end_date: Optional end date string for slicing (inclusive).

        Returns:
            DataFrame with DatetimeIndex, one column per source that has data.
            Empty DataFrame if no sources have data for this station/variable.
        """
        series = {}
        for source in self.sources:
            s = self._read_source(source, fid, variable, start_date, end_date)
            if s is not None and not s.empty:
                series[source] = s

        if not series:
            return pd.DataFrame()

        return pd.DataFrame(series)

    def load_stations(
        self,
        fids: List[str],
        variable: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load comparator values for multiple stations.

        Args:
            fids: List of station identifiers.
            variable: Canonical variable name.
            start_date: Optional start date.
            end_date: Optional end date.

        Returns:
            Dict mapping fid to DataFrame. Stations with no data are omitted.
        """
        result = {}
        for fid in fids:
            df = self.load_station(fid, variable, start_date, end_date)
            if not df.empty:
                result[fid] = df
        return result

    def compare_to_obs(
        self,
        fid: str,
        variable: str,
        obs: pd.Series,
    ) -> Dict[str, Dict[str, float]]:
        """Align comparator data to an observation series and compute metrics.

        Args:
            fid: Station identifier.
            variable: Canonical variable name.
            obs: Observation Series with DatetimeIndex.

        Returns:
            Dict[source_name, Dict[metric_name, value]] with:
                rmse, bias, mae, r2, n
            Sources with no overlapping data are omitted.
        """
        comparators = self.load_station(fid, variable)
        if comparators.empty:
            return {}

        if not isinstance(obs.index, pd.DatetimeIndex):
            obs = obs.copy()
            obs.index = pd.to_datetime(obs.index)

        results = {}
        for source in comparators.columns:
            pred = comparators[source]
            # Align on shared valid dates
            aligned = pd.DataFrame({"obs": obs, "pred": pred}).dropna()
            if len(aligned) < 2:
                continue
            results[source] = _compute_metrics(
                aligned["obs"].values, aligned["pred"].values
            )

        return results

    def compare_all_stations(
        self,
        fids: List[str],
        variable: str,
        obs_loader: Callable[[str, str], Optional[pd.Series]],
    ) -> pd.DataFrame:
        """Run compare_to_obs for many stations, return summary DataFrame.

        Args:
            fids: List of station identifiers.
            variable: Canonical variable name.
            obs_loader: Callable(fid, variable) -> pd.Series or None.
                Loads the observation time series for a given station/variable.

        Returns:
            DataFrame with columns: fid, source, rmse, bias, mae, r2, n
        """
        rows = []
        for fid in fids:
            obs = obs_loader(fid, variable)
            if obs is None or obs.empty:
                continue
            metrics = self.compare_to_obs(fid, variable, obs)
            for source, m in metrics.items():
                rows.append({"fid": fid, "source": source, **m})

        if not rows:
            return pd.DataFrame(
                columns=["fid", "source", "rmse", "bias", "mae", "r2", "n"]
            )

        return pd.DataFrame(rows)


def _compute_metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute comparison metrics between observation and prediction arrays.

    Args:
        obs: Observation values.
        pred: Prediction values (same length as obs).

    Returns:
        Dict with rmse, bias, mae, r2, n.
    """
    diff = pred - obs
    n = len(obs)
    bias = float(np.mean(diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    ss_res = np.sum(diff**2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"rmse": rmse, "bias": bias, "mae": mae, "r2": r2, "n": n}
