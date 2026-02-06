"""
Pre-training DADS on gridded reanalysis data.

This package provides utilities to pre-train the DADS meteorological GNN on
dense gridded reanalysis products (ERA5, PRISM, GridMET, etc.) before fine-tuning
on sparse observational station data.

The key insight is that grid cells can be treated as synthetic weather stations,
allowing the model to learn spatial meteorological transfer patterns from dense,
complete data before adapting to the sparse, irregular real station network.

Modules:
    config: Configuration dataclasses for grid sources and pre-training parameters
    terrain: DEM-derived terrain features at grid cell centroids
    grid_index: Zarr-backed spatial index of valid grid cells
    sampler: Epoch-level random grid cell sampling for graph construction
    sequences: Temporal sequence extraction from gridded Zarr/NetCDF sources
    dataset: PyTorch Dataset matching DadsDataset output format
    train: Pre-training loop with epoch-wise graph resampling
"""

from pretrain_build.config import PretrainConfig, GridSource
from pretrain_build.terrain import extract_terrain_at_points
from pretrain_build.grid_index import GridIndex
from pretrain_build.sampler import EpochSampler, EpochSample
from pretrain_build.sequences import SequenceExtractor
from pretrain_build.dataset import PretrainDataset

__all__ = [
    'PretrainConfig',
    'GridSource',
    'GridIndex',
    'EpochSampler',
    'EpochSample',
    'SequenceExtractor',
    'PretrainDataset',
    'extract_terrain_at_points',
]
