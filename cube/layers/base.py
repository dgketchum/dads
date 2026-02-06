"""
Abstract base class for data cube layers.

All layer builders inherit from BaseLayer and implement the build() method
to construct their specific data layer from source files.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
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
from cube.config import CubeConfig, COMPRESSION

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
        self.grid = grid if grid is not None else MasterGrid(
            bounds=config.bounds,
            resolution_deg=config.resolution_deg,
            crs=config.crs,
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
            - ('lat', 'lon') for static layers
            - ('time', 'lat', 'lon') for daily time series
            - ('doy', 'lat', 'lon') for DOY-indexed layers
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
        return 'float32'

    @property
    def compression(self) -> Optional[Any]:
        """Compression configuration for zarr arrays."""
        return COMPRESSION.get('float')

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
            checks['store_exists'] = False
            return checks

        try:
            store = zarr.open(str(self.store_path), mode='r')

            # Check group exists
            if self.name not in store:
                checks['group_exists'] = False
                return checks
            checks['group_exists'] = True

            group = store[self.name]

            # Check each variable exists
            for var in self.variables:
                var_exists = var in group
                checks[f'{var}_exists'] = var_exists

                if var_exists:
                    arr = group[var]
                    # Check shape
                    expected_shape = self._get_expected_shape()
                    checks[f'{var}_shape'] = arr.shape == expected_shape

                    # Check dtype
                    checks[f'{var}_dtype'] = str(arr.dtype) == self.dtype

        except Exception as e:
            logger.error(f"Validation error: {e}")
            checks['error'] = False

        return checks

    def _get_expected_shape(self) -> Tuple[int, ...]:
        """Get expected array shape based on dimensions."""
        shape = []
        for dim in self.dimensions:
            if dim == 'lat':
                shape.append(self.grid.n_lat)
            elif dim == 'lon':
                shape.append(self.grid.n_lon)
            elif dim == 'doy':
                shape.append(365)
            elif dim == 'time':
                # Estimate from date range
                import pandas as pd
                start = pd.Timestamp(self.config.start_date)
                end = pd.Timestamp(self.config.end_date)
                shape.append((end - start).days + 1)
            elif dim == 'composite_time':
                # 16-day composites
                import pandas as pd
                start = pd.Timestamp(self.config.start_date)
                end = pd.Timestamp(self.config.end_date)
                n_days = (end - start).days + 1
                shape.append(n_days // 16 + 1)
            else:
                raise ValueError(f"Unknown dimension: {dim}")
        return tuple(shape)

    def _open_store(self, mode: str = 'a') -> zarr.hierarchy.Group:
        """
        Open or create the zarr store.

        Args:
            mode: 'r' for read, 'w' for write (overwrite), 'a' for append

        Returns:
            Zarr root group
        """
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        return zarr.open(str(self.store_path), mode=mode)

    def _ensure_group(self, store: zarr.hierarchy.Group) -> zarr.hierarchy.Group:
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
        group: zarr.hierarchy.Group,
        name: str,
        shape: Tuple[int, ...],
        overwrite: bool = False,
    ) -> zarr.Array:
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
        chunks = tuple(self.chunks.get(dim, shape[i])
                      for i, dim in enumerate(self.dimensions))

        return group.create_dataset(
            name,
            shape=shape,
            chunks=chunks,
            dtype=self.dtype,
            compressor=self.compression,
            fill_value=self.fill_value,
            overwrite=overwrite,
        )

    def _write_array(
        self,
        group: zarr.hierarchy.Group,
        name: str,
        data: np.ndarray,
        overwrite: bool = False,
    ) -> zarr.Array:
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
        chunks = tuple(self.chunks.get(dim, data.shape[i])
                      for i, dim in enumerate(self.dimensions))

        return group.create_dataset(
            name,
            data=data,
            chunks=chunks,
            dtype=self.dtype,
            compressor=self.compression,
            fill_value=self.fill_value,
            overwrite=overwrite,
        )

    def _write_coords(self, store: zarr.hierarchy.Group) -> None:
        """
        Write coordinate arrays to store root.

        Args:
            store: Root zarr group
        """
        # Latitude
        if 'lat' not in store:
            store.create_dataset(
                'lat',
                data=self.grid.lat.astype(np.float64),
                chunks=(len(self.grid.lat),),
                dtype='float64',
            )

        # Longitude
        if 'lon' not in store:
            store.create_dataset(
                'lon',
                data=self.grid.lon.astype(np.float64),
                chunks=(len(self.grid.lon),),
                dtype='float64',
            )

    def _write_time_coord(
        self,
        store: zarr.hierarchy.Group,
        times: np.ndarray,
    ) -> None:
        """
        Write time coordinate array.

        Args:
            store: Root zarr group
            times: Array of datetime64 values
        """
        if 'time' not in store:
            # Store as int64 (days since epoch) for zarr compatibility
            times_int = times.astype('datetime64[D]').astype(np.int64)
            store.create_dataset(
                'time',
                data=times_int,
                chunks=(365,),
                dtype='int64',
            )
            store['time'].attrs['units'] = 'days since 1970-01-01'
            store['time'].attrs['calendar'] = 'standard'

    def _write_doy_coord(self, store: zarr.hierarchy.Group) -> None:
        """
        Write DOY coordinate array.

        Args:
            store: Root zarr group
        """
        if 'doy' not in store:
            store.create_dataset(
                'doy',
                data=np.arange(1, 366, dtype=np.int16),
                chunks=(365,),
                dtype='int16',
            )

    def exists(self) -> bool:
        """Check if this layer already exists in the store."""
        if not self.store_path.exists():
            return False

        try:
            store = zarr.open(str(self.store_path), mode='r')
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
