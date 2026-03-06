"""
Configuration for DADS data cube.

Defines the cube schema, chunking strategy, compression settings,
and feature definitions that align with the observation pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict, List, Optional

try:
    from numcodecs import Blosc

    HAS_NUMCODECS = True
except ImportError:
    HAS_NUMCODECS = False


# Chunking strategy optimized for spatial random access during training
CHUNKS = {
    # Static layers: larger spatial chunks (no time access pattern)
    "static": {"lat": 256, "lon": 256},
    # DOY-indexed: full DOY dimension, moderate spatial
    "doy_indexed": {"doy": 365, "lat": 128, "lon": 128},
    # Daily time series: balance temporal and spatial access
    # Small spatial chunks for random cell access, moderate temporal for sequences
    "daily": {"time": 365, "lat": 100, "lon": 100},
    # Composites: similar to daily but aligned to composite periods
    "composites": {"composite_time": 23, "lat": 100, "lon": 100},
}


def get_compression(layer_type: str = "float") -> Optional[dict]:
    """
    Get compression configuration for zarr arrays.

    Args:
        layer_type: One of 'float', 'int', 'archive'

    Returns:
        Compressor configuration dict or None if numcodecs unavailable
    """
    if not HAS_NUMCODECS:
        return None

    compressors = {
        # For continuous float data (temperature, radiation, reflectance)
        "float": Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE),
        # For integer/categorical data (masks, flags)
        "int": Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE),
        # For rarely-accessed archive data (higher compression)
        "archive": Blosc(cname="zstd", clevel=9, shuffle=Blosc.SHUFFLE),
    }
    return compressors.get(layer_type)


# Compression presets
COMPRESSION = {
    "float": get_compression("float"),
    "int": get_compression("int"),
    "archive": get_compression("archive"),
}


# Feature definitions aligned with prep/columns_desc.py
CDR_FEATURES = ["sr1", "sr2", "sr3", "bt1", "bt2", "bt3"]
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
        bounds: Spatial domain as (west, south, east, north) in degrees
        resolution_deg: Grid resolution in degrees (~0.00833 for 1km)
        crs: Coordinate reference system (WGS84)
        start_date: Start of temporal coverage
        end_date: End of temporal coverage
        source_paths: Mapping of source names to data paths
    """

    cube_path: Path = field(
        default_factory=lambda: Path("/data/ssd2/dads_cube/cube.zarr")
    )

    # Spatial domain (CONUS)
    bounds: Tuple[float, float, float, float] = (-125.0, 24.0, -66.0, 50.0)
    resolution_deg: float = 0.008333333  # ~1km at mid-latitudes (1/120 degree)
    crs: str = "EPSG:4326"

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

    @property
    def n_lat(self) -> int:
        """Number of latitude cells."""
        s, n = self.bounds[1], self.bounds[3]
        return int(round((n - s) / self.resolution_deg))

    @property
    def n_lon(self) -> int:
        """Number of longitude cells."""
        w, e = self.bounds[0], self.bounds[2]
        return int(round((e - w) / self.resolution_deg))

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape as (n_lat, n_lon)."""
        return (self.n_lat, self.n_lon)

    @property
    def n_cells(self) -> int:
        """Total number of grid cells."""
        return self.n_lat * self.n_lon


@dataclass
class LayerSpec:
    """
    Specification for a single data layer.

    Attributes:
        name: Layer name (zarr group path)
        variables: List of variable names in this layer
        dimensions: Dimension names (e.g., ('time', 'lat', 'lon'))
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
        dimensions=("lat", "lon"),
        chunks=CHUNKS["static"],
        dtype="float32",
        compression="float",
        source_key="dem",
    ),
    "rsun": LayerSpec(
        name="doy_indexed",
        variables=["rsun"],
        dimensions=("doy", "lat", "lon"),
        chunks=CHUNKS["doy_indexed"],
        dtype="float32",
        compression="float",
        source_key="rsun_dir",
    ),
    "cdr": LayerSpec(
        name="daily",
        variables=CDR_FEATURES,
        dimensions=("time", "lat", "lon"),
        chunks=CHUNKS["daily"],
        dtype="float32",
        compression="float",
        source_key="cdr_dir",
    ),
    "met": LayerSpec(
        name="daily",
        variables=MET_FEATURES,
        dimensions=("time", "lat", "lon"),
        chunks=CHUNKS["daily"],
        dtype="float32",
        compression="float",
        source_key="nldas_dir",  # Primary source; falls back to others
    ),
    "landsat": LayerSpec(
        name="composites",
        variables=LANDSAT_FEATURES,
        dimensions=("composite_time", "lat", "lon"),
        chunks=CHUNKS["composites"],
        dtype="float32",
        compression="float",
        source_key="landsat_dir",
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

    return CubeConfig(
        cube_path=Path(cube_path),
        bounds=(-125.0, 24.0, -66.0, 50.0),  # CONUS
        resolution_deg=0.008333333,  # ~1km
        crs="EPSG:4326",
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
