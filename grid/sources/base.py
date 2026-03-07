"""Base interface for gridded meteorology sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class GriddedSource(ABC):
    """Abstract base for gridded data extraction at station locations.

    Subclasses represent a single gridded product (RTMA, HRRR, ERA5, etc.)
    and provide a uniform station-extraction interface so that downstream
    graph-builders and table-builders can treat all sources the same way.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Short identifier used as column suffix, e.g. 'rtma', 'hrrr', 'era5'."""
        ...

    @property
    @abstractmethod
    def COLUMN_MAP(self) -> dict[str, str]:
        """Map source-native variable names to canonical short names.

        Example: {"TMP": "tmp", "DPT": "dpt", "UGRD": "ugrd", ...}
        Canonical names are suffixed with ``_{source_name}`` in output DataFrames.
        """
        ...

    @abstractmethod
    def extract_station_daily(
        self,
        d: date,
        station_meta: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Extract daily values at station locations for one day.

        Parameters
        ----------
        d : date
            Calendar day to extract.
        station_meta : DataFrame
            Must have columns ``fid``, ``latitude``, ``longitude``.

        Returns
        -------
        DataFrame with columns ``fid``, ``date``, plus
        ``{canonical_col}_{source_name}`` for each variable, or None
        if no data is available for the requested day.
        """
        ...

    @abstractmethod
    def available_range(self) -> tuple[date, date]:
        """Return (earliest, latest) dates available in the archive."""
        ...

    def canonical_columns(self) -> list[str]:
        """Output column names produced by extract_station_daily."""
        return [f"{v}_{self.source_name}" for v in self.COLUMN_MAP.values()]
