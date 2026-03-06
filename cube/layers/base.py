"""
Abstract base class for data cube layers.

All layer builders inherit from BaseLayer and implement the build() method
to construct their specific data layer from source files.
"""

import importlib.util
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cube.config import COMPRESSION, CubeConfig
from cube.grid import MasterGrid

try:
    import zarr

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

HAS_XARRAY = importlib.util.find_spec("xarray") is not None

logger = logging.getLogger(__name__)


class BaseLayer(ABC):
    """
    Abstract base class for all data cube layers.

    Each layer represents a group of related variables in the cube zarr store.
    Subclasses implement the build() method to construct the layer from source data.

    Attributes:
        config: CubeConfig instance
        grid: MasterGrid instance
        store_path: Path to the zarr store
    """

    def __init__(self, config: CubeConfig, grid: Optional[MasterGrid] = None):
        """
        Initialize layer builder.

        Args:
            config: CubeConfig with paths and parameters
            grid: Optional MasterGrid (created from config if not provided)
        """
        if not HAS_ZARR:
            raise ImportError("zarr is required for layer building")

        self.config = config
        self.grid = (
            grid
            if grid is not None
            else MasterGrid(
                bounds=config.bounds,
                resolution=config.resolution,
                crs=config.crs,
            )
        )
        self.store_path = config.cube_path

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Layer name (zarr group path).

        Examples: 'static', 'daily', 'doy_indexed', 'composites'
        """
        pass

    @property
    @abstractmethod
    def variables(self) -> List[str]:
        """List of variable names in this layer."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> Tuple[str, ...]:
        """
        Dimension names for this layer.

        Examples:
            - ('y', 'x') for static layers
            - ('time', 'y', 'x') for daily time series
            - ('doy', 'y', 'x') for DOY-indexed layers
        """
        pass

    @property
    @abstractmethod
    def chunks(self) -> Dict[str, int]:
        """Chunk sizes for each dimension."""
        pass

    @property
    def dtype(self) -> str:
        """Data type for layer arrays."""
        return "float32"

    @property
    def compression(self) -> Optional[Any]:
        """Compression configuration for zarr arrays."""
        return COMPRESSION.get("float")

    @property
    def fill_value(self) -> float:
        """Fill value for missing data."""
        return np.nan

    @abstractmethod
    def build(
        self,
        source_paths: Optional[Dict[str, Path]] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        """
        Build this layer from source data.

        Args:
            source_paths: Dict mapping source names to paths
                         (uses config.source_paths if not provided)
            overwrite: Whether to overwrite existing data
            **kwargs: Additional layer-specific arguments
        """
        pass

    def validate(self) -> Dict[str, bool]:
        """
        Validate layer integrity.

        Returns:
            Dict mapping check names to pass/fail status
        """
        checks = {}

        if not self.store_path.exists():
            checks["store_exists"] = False
            return checks

        try:
            store = zarr.open(str(self.store_path), mode="r")

            # Check group exists
            if self.name not in store:
                checks["group_exists"] = False
                return checks
            checks["group_exists"] = True

            group = store[self.name]

            # Check each variable exists
            for var in self.variables:
                var_exists = var in group
                checks[f"{var}_exists"] = var_exists

                if var_exists:
                    arr = group[var]
                    # Check shape
                    expected_shape = self._get_expected_shape()
                    checks[f"{var}_shape"] = arr.shape == expected_shape

                    # Check dtype
                    checks[f"{var}_dtype"] = str(arr.dtype) == self.dtype

        except Exception as e:
            logger.error(f"Validation error: {e}")
            checks["error"] = False

        return checks

    def _get_expected_shape(self) -> Tuple[int, ...]:
        """Get expected array shape based on dimensions."""
        shape = []
        for dim in self.dimensions:
            if dim == "y":
                shape.append(self.grid.n_y)
            elif dim == "x":
                shape.append(self.grid.n_x)
            elif dim == "doy":
                shape.append(365)
            elif dim == "time":
                # Estimate from date range
                import pandas as pd

                start = pd.Timestamp(self.config.start_date)
                end = pd.Timestamp(self.config.end_date)
                shape.append((end - start).days + 1)
            elif dim == "composite_time":
                import pandas as pd

                from cube.config import N_PERIODS_PER_YEAR

                start = pd.Timestamp(self.config.start_date)
                end = pd.Timestamp(self.config.end_date)
                n_years = end.year - start.year + 1
                shape.append(n_years * N_PERIODS_PER_YEAR)
            else:
                raise ValueError(f"Unknown dimension: {dim}")
        return tuple(shape)

    def _open_store(self, mode: str = "a") -> "zarr.Group":
        """
        Open or create the zarr store.

        Args:
            mode: 'r' for read, 'w' for write (overwrite), 'a' for append

        Returns:
            Zarr root group
        """
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        return zarr.open(str(self.store_path), mode=mode)

    def _ensure_group(self, store: "zarr.Group") -> "zarr.Group":
        """
        Ensure layer group exists in store.

        Args:
            store: Root zarr group

        Returns:
            Layer group
        """
        return store.require_group(self.name)

    def _create_array(
        self,
        group: "zarr.Group",
        name: str,
        shape: Tuple[int, ...],
        overwrite: bool = False,
    ) -> "zarr.Array":
        """
        Create a zarr array in the group.

        Args:
            group: Zarr group to create array in
            name: Array name
            shape: Array shape
            overwrite: Whether to overwrite existing

        Returns:
            Zarr array
        """
        chunks = tuple(
            self.chunks.get(dim, shape[i]) for i, dim in enumerate(self.dimensions)
        )

        return group.create_dataset(
            name,
            shape=shape,
            chunks=chunks,
            dtype=self.dtype,
            compressors=self.compression,
            fill_value=self.fill_value,
            overwrite=overwrite,
        )

    def _write_array(
        self,
        group: "zarr.Group",
        name: str,
        data: np.ndarray,
        overwrite: bool = False,
    ) -> "zarr.Array":
        """
        Write data to a zarr array.

        Args:
            group: Zarr group
            name: Array name
            data: Data to write
            overwrite: Whether to overwrite existing

        Returns:
            Zarr array
        """
        chunks = tuple(
            self.chunks.get(dim, data.shape[i]) for i, dim in enumerate(self.dimensions)
        )

        return group.create_dataset(
            name,
            shape=data.shape,
            data=data,
            chunks=chunks,
            dtype=self.dtype,
            compressors=self.compression,
            fill_value=self.fill_value,
            overwrite=overwrite,
        )

    def _write_coords(self, store: "zarr.Group") -> None:
        """
        Write coordinate arrays to store root.

        Writes 1D y/x as dimension coords and 2D lat/lon as auxiliary coords.

        Args:
            store: Root zarr group
        """
        # 1D projected dimension coords
        if "y" not in store:
            arr = store.create_dataset(
                "y",
                shape=self.grid.y.astype(np.float64).shape,
                data=self.grid.y.astype(np.float64),
                chunks=(len(self.grid.y),),
                dtype="float64",
            )
            arr.attrs["standard_name"] = "projection_y_coordinate"
            arr.attrs["units"] = "m"
            arr.attrs["axis"] = "Y"

        if "x" not in store:
            arr = store.create_dataset(
                "x",
                shape=self.grid.x.astype(np.float64).shape,
                data=self.grid.x.astype(np.float64),
                chunks=(len(self.grid.x),),
                dtype="float64",
            )
            arr.attrs["standard_name"] = "projection_x_coordinate"
            arr.attrs["units"] = "m"
            arr.attrs["axis"] = "X"

        # 2D auxiliary geographic coords
        if "lat" not in store:
            arr = store.create_dataset(
                "lat",
                shape=self.grid.lat.astype(np.float64).shape,
                data=self.grid.lat.astype(np.float64),
                chunks=(min(256, self.grid.n_y), min(256, self.grid.n_x)),
                dtype="float64",
            )
            arr.attrs["standard_name"] = "latitude"
            arr.attrs["units"] = "degrees_north"

        if "lon" not in store:
            arr = store.create_dataset(
                "lon",
                shape=self.grid.lon.astype(np.float64).shape,
                data=self.grid.lon.astype(np.float64),
                chunks=(min(256, self.grid.n_y), min(256, self.grid.n_x)),
                dtype="float64",
            )
            arr.attrs["standard_name"] = "longitude"
            arr.attrs["units"] = "degrees_east"

    def _write_time_coord(
        self,
        store: "zarr.Group",
        times: np.ndarray,
    ) -> None:
        """
        Write time coordinate array.

        Args:
            store: Root zarr group
            times: Array of datetime64 values
        """
        if "time" not in store:
            # Store as int64 (days since epoch) for zarr compatibility
            times_int = times.astype("datetime64[D]").astype(np.int64)
            store.create_dataset(
                "time",
                shape=times_int.shape,
                data=times_int,
                chunks=(365,),
                dtype="int64",
            )
            store["time"].attrs["units"] = "days since 1970-01-01"
            store["time"].attrs["calendar"] = "standard"

    def _write_doy_coord(self, store: "zarr.Group") -> None:
        """
        Write DOY coordinate array.

        Args:
            store: Root zarr group
        """
        if "doy" not in store:
            doy_data = np.arange(1, 366, dtype=np.int16)
            store.create_dataset(
                "doy",
                shape=doy_data.shape,
                data=doy_data,
                chunks=(365,),
                dtype="int16",
            )

    def _write_composite_time_coord(
        self,
        store: "zarr.Group",
        start_year: int,
        end_year: int,
    ) -> None:
        """
        Write composite_time coordinate as mid-date of each seasonal period.

        Args:
            store: Root zarr group
            start_year: First year (inclusive)
            end_year: Last year (inclusive)
        """
        import pandas as pd

        from cube.config import SEASONAL_PERIODS

        if "composite_time" not in store:
            dates = []
            for yr in range(start_year, end_year + 1):
                for _, start_mmdd, end_mmdd in SEASONAL_PERIODS:
                    s = pd.Timestamp(f"{yr}-{start_mmdd}")
                    e = pd.Timestamp(f"{yr}-{end_mmdd}")
                    dates.append(s + (e - s) / 2)

            times = np.array(dates, dtype="datetime64[D]")
            arr = store.create_dataset(
                "composite_time",
                shape=times.astype(np.int64).shape,
                data=times.astype(np.int64),
                chunks=(len(times),),
                dtype="int64",
            )
            arr.attrs["units"] = "days since 1970-01-01"
            arr.attrs["calendar"] = "standard"

    def exists(self) -> bool:
        """Check if this layer already exists in the store."""
        if not self.store_path.exists():
            return False

        try:
            store = zarr.open(str(self.store_path), mode="r")
            if self.name not in store:
                return False

            group = store[self.name]
            return all(var in group for var in self.variables)
        except Exception:
            return False

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"variables={self.variables}, "
            f"dimensions={self.dimensions})"
        )
