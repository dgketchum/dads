"""
Configuration for DADS data cube.

Defines the cube schema, chunking strategy, compression settings,
and feature definitions that align with the observation pipeline.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from zarr.codecs import BloscCodec, BloscShuffle

    HAS_BLOSC = True
except ImportError:
    HAS_BLOSC = False


# Chunking strategy optimized for spatial random access during training
CHUNKS = {
    # Static layers: larger spatial chunks (no time access pattern)
    "static": {"y": 256, "x": 256},
    # DOY-indexed: full DOY dimension, moderate spatial
    "doy_indexed": {"doy": 365, "y": 128, "x": 128},
    # Daily time series: balance temporal and spatial access
    # Small spatial chunks for random cell access, moderate temporal for sequences
    "daily": {"time": 365, "y": 100, "x": 100},
    # Composites: aligned to seasonal periods (5 per year)
    "composites": {"composite_time": 5, "y": 100, "x": 100},
    # CDR native: one year of full spatial per chunk (~0.05° grid)
    "cdr_native": {"time": 365, "cdr_lat": 140, "cdr_lon": 420},
}

# Seasonal compositing periods (period_id, start_mmdd, end_mmdd)
SEASONAL_PERIODS = [
    ("0", "01-01", "03-01"),  # Winter
    ("1", "03-01", "05-01"),  # Spring
    ("2", "05-01", "07-15"),  # Late spring
    ("3", "07-15", "09-30"),  # Summer
    ("4", "09-30", "12-31"),  # Fall
]
N_PERIODS_PER_YEAR = len(SEASONAL_PERIODS)


def get_compression(layer_type: str = "float"):
    """
    Get compression codec for zarr arrays (zarr v3 native).

    Args:
        layer_type: One of 'float', 'int', 'archive'

    Returns:
        zarr BloscCodec instance or None
    """
    if not HAS_BLOSC:
        return None

    codecs = {
        # For continuous float data (temperature, radiation, reflectance)
        "float": BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle),
        # For integer/categorical data (masks, flags)
        "int": BloscCodec(cname="lz4", clevel=5, shuffle=BloscShuffle.bitshuffle),
        # For rarely-accessed archive data (higher compression)
        "archive": BloscCodec(cname="zstd", clevel=9, shuffle=BloscShuffle.shuffle),
    }
    return codecs.get(layer_type)


# Compression presets
COMPRESSION = {
    "float": get_compression("float"),
    "int": get_compression("int"),
    "archive": get_compression("archive"),
}


# Feature definitions aligned with prep/columns_desc.py
CDR_FEATURES = ["sr1", "sr2", "sr3", "bt1", "bt2", "bt3"]
CDR_MISS_FEATURES = [
    "sr1_miss",
    "sr2_miss",
    "sr3_miss",
    "bt1_miss",
    "bt2_miss",
    "bt3_miss",
]
CDR_NATIVE_FEATURES = CDR_FEATURES + CDR_MISS_FEATURES
LANDSAT_FEATURES = [
    "landsat_b2",
    "landsat_b3",
    "landsat_b4",
    "landsat_b5",
    "landsat_b6",
    "landsat_b7",
    "landsat_b10",
]
TERRAIN_FEATURES = [
    "elevation",
    "slope",
    "aspect",
    "tpi_500",
    "tpi_2500",
    "tpi_10000",
    "tpi_22500",
]
MET_FEATURES = ["tmax", "tmin", "prcp", "rsds", "vpd", "wind", "ea"]

# Static feature order: matches GEO_FEATURES from columns_desc.py
# ['lat', 'lon', 'rsun', 'doy_sin', 'doy_cos'] + LANDSAT + CDR + TERRAIN
# Note: 'lat'/'lon' here are model feature names, not dimension names
GEO_FEATURE_ORDER = (
    ["lat", "lon", "rsun", "doy_sin", "doy_cos"]
    + LANDSAT_FEATURES
    + CDR_FEATURES
    + TERRAIN_FEATURES
)


@dataclass
class CubeConfig:
    """
    Configuration for building and accessing the data cube.

    Attributes:
        cube_path: Path to the zarr store (e.g., /data/ssd2/dads_cube/cube.zarr)
        bounds: Spatial domain as (x_min, y_min, x_max, y_max) in EPSG:5070 meters
        resolution: Grid resolution in meters (default 1000.0)
        crs: Coordinate reference system (EPSG:5070)
        start_date: Start of temporal coverage
        end_date: End of temporal coverage
        source_paths: Mapping of source names to data paths
    """

    cube_path: Path = field(
        default_factory=lambda: Path("/data/ssd2/dads_cube/cube.zarr")
    )

    # Spatial domain (EPSG:5070 meters)
    bounds: Tuple[float, float, float, float] = (
        -2361000.0,
        258000.0,
        2264000.0,
        3177000.0,
    )
    resolution: float = 1000.0
    crs: str = "EPSG:5070"

    # Temporal extent
    start_date: str = "1990-01-01"
    end_date: str = "2025-12-31"

    # Source data paths (user-configurable)
    source_paths: Dict[str, Path] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.cube_path, Path):
            self.cube_path = Path(self.cube_path)

        # Convert source paths to Path objects
        self.source_paths = {
            k: Path(v) if not isinstance(v, Path) else v
            for k, v in self.source_paths.items()
        }

    @classmethod
    def from_toml(cls, path: str | Path) -> "CubeConfig":
        """Load a CubeConfig from a TOML project file.

        Args:
            path: Path to the TOML file (e.g., 'projects/pnw.toml')

        Returns:
            Fully populated CubeConfig with projected bounds
        """
        from cube.grid import _geographic_bounds_to_albers

        with open(path, "rb") as f:
            cfg = tomllib.load(f)

        domain = cfg["domain"]
        w, s, e, n = domain["geographic_bounds"]
        resolution = domain.get("resolution", 1000.0)
        crs = domain.get("crs", "EPSG:5070")

        bounds = _geographic_bounds_to_albers(w, s, e, n, resolution=resolution)

        temporal = cfg.get("temporal", {})
        output = cfg.get("output", {})
        sources = cfg.get("sources", {})

        return cls(
            cube_path=Path(output.get("cube_path", "/data/ssd2/dads_cube/cube.zarr")),
            bounds=bounds,
            resolution=resolution,
            crs=crs,
            start_date=temporal.get("start_date", "1990-01-01"),
            end_date=temporal.get("end_date", "2025-12-31"),
            source_paths=sources,
        )

    @property
    def n_y(self) -> int:
        """Number of rows (northing cells)."""
        x_min, y_min, x_max, y_max = self.bounds
        return int(round((y_max - y_min) / self.resolution))

    @property
    def n_x(self) -> int:
        """Number of columns (easting cells)."""
        x_min, y_min, x_max, y_max = self.bounds
        return int(round((x_max - x_min) / self.resolution))

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape as (n_y, n_x)."""
        return (self.n_y, self.n_x)

    @property
    def n_cells(self) -> int:
        """Total number of grid cells."""
        return self.n_y * self.n_x


@dataclass
class LayerSpec:
    """
    Specification for a single data layer.

    Attributes:
        name: Layer name (zarr group path)
        variables: List of variable names in this layer
        dimensions: Dimension names (e.g., ('time', 'y', 'x'))
        chunks: Chunk sizes per dimension
        dtype: Data type (e.g., 'float32')
        compression: Compression type ('float', 'int', 'archive')
        source_key: Key in CubeConfig.source_paths for source data
    """

    name: str
    variables: List[str]
    dimensions: Tuple[str, ...]
    chunks: Dict[str, int]
    dtype: str = "float32"
    compression: str = "float"
    source_key: Optional[str] = None


# Layer specifications
LAYER_SPECS = {
    "terrain": LayerSpec(
        name="static",
        variables=TERRAIN_FEATURES + ["land_mask"],
        dimensions=("y", "x"),
        chunks=CHUNKS["static"],
        dtype="float32",
        compression="float",
        source_key="dem",
    ),
    "rsun": LayerSpec(
        name="doy_indexed",
        variables=["rsun"],
        dimensions=("doy", "y", "x"),
        chunks=CHUNKS["doy_indexed"],
        dtype="float32",
        compression="float",
        source_key="rsun_dir",
    ),
    "cdr": LayerSpec(
        name="daily",
        variables=CDR_FEATURES,
        dimensions=("time", "y", "x"),
        chunks=CHUNKS["daily"],
        dtype="float32",
        compression="float",
        source_key="cdr_dir",
    ),
    "met": LayerSpec(
        name="daily",
        variables=MET_FEATURES,
        dimensions=("time", "y", "x"),
        chunks=CHUNKS["daily"],
        dtype="float32",
        compression="float",
        source_key="nldas_dir",  # Primary source; falls back to others
    ),
    "landsat": LayerSpec(
        name="composites",
        variables=LANDSAT_FEATURES,
        dimensions=("composite_time", "y", "x"),
        chunks=CHUNKS["composites"],
        dtype="float32",
        compression="float",
        source_key="landsat_dir",
    ),
    "cdr_native": LayerSpec(
        name="cdr_native",
        variables=CDR_NATIVE_FEATURES,
        dimensions=("time", "cdr_lat", "cdr_lon"),
        chunks=CHUNKS["cdr_native"],
        dtype="float32",
        compression="float",
        source_key="cdr_dir",
    ),
}


def default_conus_config(
    cube_path: str = "/data/ssd2/dads_cube/cube.zarr",
    dem_path: Optional[str] = None,
    rsun_dir: Optional[str] = None,
    cdr_dir: Optional[str] = None,
    nldas_dir: Optional[str] = None,
    prism_dir: Optional[str] = None,
    gridmet_dir: Optional[str] = None,
    landsat_dir: Optional[str] = None,
) -> CubeConfig:
    """
    Create default configuration for CONUS data cube.

    Args:
        cube_path: Output path for zarr store
        dem_path: Path to DEM GeoTIFF(s)
        rsun_dir: Directory with 365 rsun GeoTIFFs
        cdr_dir: Directory with NOAA CDR NetCDF files
        nldas_dir: Directory with NLDAS NetCDF files
        prism_dir: Directory with PRISM NetCDF files
        gridmet_dir: Directory with GridMET NetCDF files
        landsat_dir: Directory with Landsat composite GeoTIFFs

    Returns:
        CubeConfig instance
    """
    from cube.grid import _geographic_bounds_to_albers

    source_paths = {}
    if dem_path:
        source_paths["dem"] = dem_path
    if rsun_dir:
        source_paths["rsun_dir"] = rsun_dir
    if cdr_dir:
        source_paths["cdr_dir"] = cdr_dir
    if nldas_dir:
        source_paths["nldas_dir"] = nldas_dir
    if prism_dir:
        source_paths["prism_dir"] = prism_dir
    if gridmet_dir:
        source_paths["gridmet_dir"] = gridmet_dir
    if landsat_dir:
        source_paths["landsat_dir"] = landsat_dir

    bounds = _geographic_bounds_to_albers(-125.0, 24.0, -66.0, 50.0)

    return CubeConfig(
        cube_path=Path(cube_path),
        bounds=bounds,
        resolution=1000.0,
        crs="EPSG:5070",
        start_date="1990-01-01",
        end_date="2025-12-31",
        source_paths=source_paths,
    )


if __name__ == "__main__":
    # Example usage
    config = default_conus_config(
        dem_path="/data/dem/conus_dem.tif",
        nldas_dir="/data/ssd1/nldas2/netcdf/",
    )
    print(f"Grid shape: {config.shape}")
    print(f"Total cells: {config.n_cells:,}")
    print(f"Cube path: {config.cube_path}")
