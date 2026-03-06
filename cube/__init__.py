"""
DADS Unified Data Cube

A zarr/xarray-based data cube for pre-training DADS on gridded climate data.
Provides unified access to meteorological, remote sensing, and terrain data
at 1km resolution over CONUS in EPSG:5070 (Albers Equal Area Conic).

Main components:
    - MasterGrid: Defines the 1km EPSG:5070 coordinate system
    - DataCubeExtractor: Extracts training sequences from the cube
    - Layer classes: Build individual data layers (terrain, CDR, met, etc.)
    - Adapters: Drop-in replacements for pretrain_build compatibility

Example usage:
    from cube import DataCubeExtractor, MasterGrid

    # Extract training sequences
    extractor = DataCubeExtractor('/data/ssd2/dads_cube/cube.zarr')
    target_seq, exog_seq, terrain, valid = extractor.extract_sequence(
        row=1000, col=2000,
        end_date=np.datetime64('2020-06-15'),
        target_variable='tmax',
        seq_len=12
    )

    # Build cube layers
    from cube.layers import TerrainLayer, RSUNLayer, CDRLayer, MetLayer
    from cube.config import default_conus_config

    config = default_conus_config(
        cube_path='/data/ssd2/dads_cube/cube.zarr',
        dem_path='/data/dem/conus_dem.tif',
    )
    terrain = TerrainLayer(config)
    terrain.build()
"""

from cube.config import CHUNKS, COMPRESSION, CubeConfig, default_conus_config
from cube.extractors.data_cube_extractor import DataCubeExtractor
from cube.extractors.pretrain_adapter import CubeGridIndex, CubeSequenceExtractor
from cube.grid import MasterGrid, create_conus_grid, create_test_region_grid

__version__ = "0.2.0"

__all__ = [
    # Config
    "CubeConfig",
    "CHUNKS",
    "COMPRESSION",
    "default_conus_config",
    # Grid
    "MasterGrid",
    "create_conus_grid",
    "create_test_region_grid",
    # Extractors
    "DataCubeExtractor",
    "CubeSequenceExtractor",
    "CubeGridIndex",
]
