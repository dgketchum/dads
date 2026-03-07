"""
Configuration dataclasses for pre-training DADS on gridded data.

Uses existing extracted data where available (parquet files in
extract/met_data/grid outputs), plus native Zarr/NetCDF for
direct grid sampling.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class GridSource:
    """Describes a gridded data source.

    Attributes:
        name: Source identifier (e.g., 'prism', 'gridmet', 'era5_land')
        zarr_path: Path to Zarr store (preferred for random access)
        netcdf_dir: Directory of yearly NetCDF files (fallback)
        variables: Variable names available in this source
        resolution_deg: Native spatial resolution in degrees
        temporal_resolution: 'daily' or 'hourly'
        lat_var: Name of latitude coordinate variable
        lon_var: Name of longitude coordinate variable
        time_var: Name of time coordinate variable
    """

    name: str
    zarr_path: Optional[Path] = None
    netcdf_dir: Optional[Path] = None
    variables: List[str] = field(default_factory=list)
    resolution_deg: float = 0.04166667  # ~4km (PRISM resolution)
    temporal_resolution: str = "daily"
    lat_var: str = "lat"
    lon_var: str = "lon"
    time_var: str = "time"

    def __post_init__(self):
        if self.zarr_path is not None and not isinstance(self.zarr_path, Path):
            self.zarr_path = Path(self.zarr_path)
        if self.netcdf_dir is not None and not isinstance(self.netcdf_dir, Path):
            self.netcdf_dir = Path(self.netcdf_dir)


@dataclass
class PretrainConfig:
    """Configuration for DADS pre-training on gridded data.

    Attributes:
        bounds: Spatial domain as (west, south, east, north) in degrees
        n_cells_per_epoch: Number of grid cells to sample each epoch
        sampling_strategy: 'uniform', 'poisson_disk', or 'stratified'
        min_cell_spacing_km: Minimum spacing for poisson_disk sampling
        resample_each_epoch: Whether to resample grid cells each epoch
        n_neighbors: Number of neighbors per target in graph
        neighbor_pool_factor: Oversample candidates before filtering
        max_distance_km: Maximum neighbor distance
        seq_len: Days in temporal window (matches observation system)
        date_range: (start_date, end_date) as ISO strings
        neighbor_drop_prob: Probability of dropping a neighbor entirely
        timestep_mask_prob: Probability of masking individual timesteps
        sources: List of GridSource configurations
        dem_path: Path to DEM for terrain feature extraction
        terrain_cache: Path to cache pre-computed terrain features
        cache_dir: Directory for spatial index and other caches
        variable_map: Map target variable to source name
    """

    # Spatial domain (CONUS default)
    bounds: Tuple[float, float, float, float] = (-125.0, 25.0, -67.0, 49.5)

    # Grid sampling
    n_cells_per_epoch: int = 8000
    sampling_strategy: str = "poisson_disk"
    min_cell_spacing_km: float = 20.0
    resample_each_epoch: bool = True

    # Graph construction
    n_neighbors: int = 10
    neighbor_pool_factor: int = 3
    max_distance_km: float = 500.0

    # Temporal
    seq_len: int = 12
    date_range: Tuple[str, str] = ("1990-01-01", "2023-12-31")

    # Synthetic missingness injection
    neighbor_drop_prob: float = 0.1
    timestep_mask_prob: float = 0.05

    # Gridded data sources
    sources: List[GridSource] = field(default_factory=list)

    # Terrain
    dem_path: Optional[Path] = None
    terrain_cache: Optional[Path] = None

    # Output/cache
    cache_dir: Optional[Path] = None

    # Variable to source mapping
    variable_map: Dict[str, str] = field(
        default_factory=lambda: {
            "tmax": "prism",
            "tmin": "prism",
            "ppt": "prism",
            "rsds": "gridmet",
            "srad": "gridmet",
            "ea": "gridmet",
            "vpd": "gridmet",
            "wind": "gridmet",
        }
    )

    def __post_init__(self):
        if self.dem_path is not None and not isinstance(self.dem_path, Path):
            self.dem_path = Path(self.dem_path)
        if self.terrain_cache is not None and not isinstance(self.terrain_cache, Path):
            self.terrain_cache = Path(self.terrain_cache)
        if self.cache_dir is not None and not isinstance(self.cache_dir, Path):
            self.cache_dir = Path(self.cache_dir)


def default_western_us_config(
    prism_zarr: Optional[str] = None,
    gridmet_zarr: Optional[str] = None,
    dem_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> PretrainConfig:
    """Create default configuration for Western US pre-training.

    Args:
        prism_zarr: Path to PRISM Zarr store
        gridmet_zarr: Path to GridMET Zarr store
        dem_path: Path to DEM raster
        cache_dir: Directory for caching spatial index

    Returns:
        PretrainConfig with sensible defaults for Western US
    """
    sources = []

    if prism_zarr:
        sources.append(
            GridSource(
                name="prism",
                zarr_path=Path(prism_zarr),
                variables=["tmax", "tmin", "ppt"],
                resolution_deg=0.04166667,  # 4km
                temporal_resolution="daily",
            )
        )

    if gridmet_zarr:
        sources.append(
            GridSource(
                name="gridmet",
                zarr_path=Path(gridmet_zarr),
                variables=["srad", "vpd", "th", "vs"],  # GridMET variable names
                resolution_deg=0.04166667,  # 4km
                temporal_resolution="daily",
            )
        )

    return PretrainConfig(
        bounds=(-125.0, 31.0, -102.0, 49.0),  # Western US
        n_cells_per_epoch=8000,
        sampling_strategy="poisson_disk",
        min_cell_spacing_km=20.0,
        n_neighbors=10,
        max_distance_km=400.0,
        seq_len=12,
        date_range=("1990-01-01", "2023-12-31"),
        sources=sources,
        dem_path=Path(dem_path) if dem_path else None,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )


if __name__ == "__main__":
    # Example usage
    config = default_western_us_config(
        prism_zarr="/data/gridded/prism.zarr",
        gridmet_zarr="/data/gridded/gridmet.zarr",
        dem_path="/data/dem/conus_dem.tif",
        cache_dir="/data/pretrain_cache",
    )
    print(config)
