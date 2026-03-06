"""
PyTorch Dataset for pre-training.

Mirrors the interface of DadsDataset but sources data from gridded
products rather than station parquets.

Key differences from DadsDataset:
    1. No station embeddings - uses terrain-derived identity vectors
    2. Graph structure comes from EpochSample (rebuilt each epoch)
    3. Sequences extracted on-the-fly from Zarr/NetCDF
    4. Synthetic missingness injection for robustness

The output format matches DadsDataset exactly:
    (graph, y, neighbor_seq, neighbor_mask, target_seq)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from models.components.scalers import MinMaxScaler

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data

    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

from pretrain_build.config import PretrainConfig
from pretrain_build.grid_index import GridIndex
from pretrain_build.sampler import EpochSample
from pretrain_build.sequences import SequenceExtractor


class PretrainDataset(Dataset):
    """
    Dataset for DADS pre-training on gridded data.

    Each sample is a (graph, y, neighbor_seq, neighbor_mask, target_seq) tuple
    matching the DadsDataset output format, allowing DadsMetGNN to be used
    without modification.

    The dataset generates random temporal windows for each cell, with optional
    synthetic missingness injection to prepare the model for real-world data gaps.
    """

    def __init__(
        self,
        epoch_sample: EpochSample,
        grid_index: GridIndex,
        seq_extractor: SequenceExtractor,
        config: PretrainConfig,
        variable: str,
        scaler: "MinMaxScaler",
        windows_per_cell: int = 100,
        seed: Optional[int] = None,
        inject_missingness: bool = True,
    ):
        """
        Initialize dataset for one epoch.

        Args:
            epoch_sample: Graph structure for this epoch (from EpochSampler)
            grid_index: Spatial index with coordinates and terrain
            seq_extractor: Extracts temporal sequences from gridded data
            config: Pre-training configuration
            variable: Target variable (e.g., 'tmax')
            scaler: Fitted MinMaxScaler for normalization
            windows_per_cell: Number of random temporal windows per cell
            seed: Random seed for reproducibility
            inject_missingness: Whether to inject synthetic missingness
        """
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric is required for PretrainDataset")

        self.sample = epoch_sample
        self.index = grid_index
        self.extractor = seq_extractor
        self.config = config
        self.variable = variable
        self.scaler = scaler
        self.inject_missingness = inject_missingness

        self.rng = np.random.default_rng(seed)

        # Dimensions
        self.terrain_dim = grid_index.terrain.shape[1]
        self.emb_dim = self.terrain_dim  # Use terrain as pseudo-embedding
        # Exog: rsun + doy_sin + doy_cos + terrain
        self.exog_dim = 3 + self.terrain_dim
        # Sequence channels: target + exog
        self.seq_in_channels = 1 + self.exog_dim

        # Build sample index: (target_cell_idx, end_date)
        self._build_index(windows_per_cell)

        # Edge attribute dimension: terrain_delta + bearing(sin,cos) + distance + exog_delta
        self.edge_dim = self.terrain_dim + 2 + 1 + self.exog_dim

    def _build_index(self, windows_per_cell: int):
        """Build (cell_idx, end_date) pairs for all samples."""
        start_str, end_str = self.config.date_range
        start_ord = np.datetime64(start_str, "D").astype(int)
        end_ord = np.datetime64(end_str, "D").astype(int)

        # Need seq_len days before end_date
        valid_start = start_ord + self.config.seq_len

        self.samples = []
        for cell_idx in self.sample.target_indices:
            # Sample random end dates within valid range
            end_dates = self.rng.integers(valid_start, end_ord, size=windows_per_cell)
            for ed in end_dates:
                end_date = np.datetime64(ed, "D")
                self.samples.append((int(cell_idx), end_date))

    def __len__(self):
        return len(self.samples)

    def __getitem__(
        self, idx
    ) -> Tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single training sample.

        Returns:
            (graph, y, neighbor_seq, neighbor_mask, target_seq) tuple matching DadsDataset
        """
        cell_idx, end_date = self.samples[idx]

        # Get target cell info
        tgt_lat, tgt_lon = self.index.coords[cell_idx]
        tgt_terrain = self.index.terrain[cell_idx]
        tgt_row, tgt_col = self.index.grid_indices[cell_idx]

        # Extract target sequence
        tgt_seq, tgt_valid = self.extractor.extract_sequence(
            tgt_lat,
            tgt_lon,
            end_date,
            self.variable,
            tgt_terrain,
            row=int(tgt_row),
            col=int(tgt_col),
        )

        if not tgt_valid:
            return self._empty_sample()

        # Scale target sequence
        tgt_seq_scaled = self._scale_sequence(tgt_seq)
        # y is the scaled target variable sequence
        y = torch.from_numpy(tgt_seq_scaled[:, 0])

        # Get neighbors from epoch sample
        neighbor_indices, bearings, distances, terrain_deltas = (
            self.sample.get_neighbors(cell_idx)
        )

        if len(neighbor_indices) < self.config.n_neighbors:
            return self._empty_sample()

        # Extract neighbor sequences
        neighbor_seqs = []
        neighbor_masks = []
        neighbor_obs_today = []
        neighbor_exog_today = []

        for ni in neighbor_indices:
            nbr_lat, nbr_lon = self.index.coords[ni]
            nbr_terrain = self.index.terrain[ni]
            nbr_row, nbr_col = self.index.grid_indices[ni]

            nbr_seq, nbr_valid = self.extractor.extract_sequence(
                nbr_lat,
                nbr_lon,
                end_date,
                self.variable,
                nbr_terrain,
                row=int(nbr_row),
                col=int(nbr_col),
            )

            # Inject synthetic missingness
            if self.inject_missingness:
                if self.rng.random() < self.config.neighbor_drop_prob:
                    nbr_valid = False

            if nbr_valid:
                nbr_seq_scaled = self._scale_sequence(nbr_seq)

                # Apply timestep masking
                if self.inject_missingness:
                    mask = (
                        self.rng.random(self.config.seq_len)
                        > self.config.timestep_mask_prob
                    )
                    nbr_seq_scaled[~mask] = 0.0

                neighbor_seqs.append(torch.from_numpy(nbr_seq_scaled))
                neighbor_masks.append(True)
                neighbor_obs_today.append(float(nbr_seq_scaled[-1, 0]))
                neighbor_exog_today.append(
                    torch.from_numpy(nbr_seq_scaled[-1, 1:].astype(np.float32))
                )
            else:
                neighbor_seqs.append(
                    torch.zeros(self.config.seq_len, self.seq_in_channels)
                )
                neighbor_masks.append(False)
                neighbor_obs_today.append(0.0)
                neighbor_exog_today.append(torch.zeros(self.exog_dim))

        neighbor_seq = torch.stack(neighbor_seqs, dim=0)  # (k, T, C)
        neighbor_mask = torch.tensor(neighbor_masks, dtype=torch.bool)

        # Build node features matching DadsDataset format:
        # Target: [zeros(emb), exog_today, 0]
        # Neighbors: [terrain_emb, exog_today, obs_today]
        tgt_exog = torch.from_numpy(tgt_seq_scaled[-1, 1:].astype(np.float32))

        # Target row: embedding(zeros) + exog + obs(zero for target)
        tgt_row_feat = torch.cat(
            [
                torch.zeros(self.emb_dim, dtype=torch.float32),
                tgt_exog,
                torch.zeros(1, dtype=torch.float32),
            ]
        )

        # Neighbor rows: terrain_embedding + exog + obs
        nbr_embs = torch.from_numpy(
            self.index.terrain[neighbor_indices].astype(np.float32)
        )
        nbr_exog_mat = torch.stack(neighbor_exog_today, dim=0)
        nbr_obs = torch.tensor(neighbor_obs_today, dtype=torch.float32).unsqueeze(1)
        nbr_rows = torch.cat([nbr_embs, nbr_exog_mat, nbr_obs], dim=1)

        x = torch.cat([tgt_row_feat.unsqueeze(0), nbr_rows], dim=0)

        # Build edge attributes
        terrain_delta = torch.from_numpy(terrain_deltas.astype(np.float32))

        # Bearing: sin/cos encoding
        bearing_rad = torch.tensor(
            [np.radians(b) for b in bearings], dtype=torch.float32
        )
        bearing_sc = torch.stack(
            [torch.sin(bearing_rad), torch.cos(bearing_rad)], dim=1
        )

        # Distance: scaled to ~[0,1]
        dist_scaled = torch.tensor(distances, dtype=torch.float32).unsqueeze(1) / 500.0

        # Exog delta: target - neighbor
        exog_delta = tgt_exog.unsqueeze(0) - nbr_exog_mat

        edge_attr = torch.cat(
            [terrain_delta, bearing_sc, dist_scaled, exog_delta], dim=1
        )

        # Edge index: neighbors (1..k) -> target (0)
        n_neighbors = len(neighbor_indices)
        edge_index = torch.stack(
            [
                torch.arange(1, n_neighbors + 1),
                torch.zeros(n_neighbors, dtype=torch.long),
            ],
            dim=0,
        )

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Target exog-only temporal sequence (pad target channel with zeros)
        target_seq = torch.zeros(
            self.config.seq_len, self.seq_in_channels, dtype=torch.float32
        )
        target_seq[:, 1:] = torch.from_numpy(tgt_seq_scaled[:, 1:].astype(np.float32))

        return graph, y, neighbor_seq, neighbor_mask, target_seq

    def _scale_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Apply MinMax scaling to sequence."""
        # scaler.bias and scaler.scale are (1, n_features)
        bias = np.asarray(self.scaler.bias).reshape(-1)
        scale = np.asarray(self.scaler.scale).reshape(-1)

        # Handle dimension mismatch - only scale features that exist in scaler
        n_seq_features = seq.shape[1]
        n_scaler_features = len(bias)

        if n_seq_features <= n_scaler_features:
            # Use first n_seq_features from scaler
            bias = bias[:n_seq_features]
            scale = scale[:n_seq_features]
        else:
            # Pad scaler with identity transform (0 bias, 1 scale)
            pad_size = n_seq_features - n_scaler_features
            bias = np.concatenate([bias, np.zeros(pad_size)])
            scale = np.concatenate([scale, np.ones(pad_size)])

        scaled = (seq - bias) / scale + 5e-8
        return scaled.astype(np.float32)

    def _empty_sample(
        self,
    ) -> Tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a zeroed sample for invalid data."""
        n = self.config.n_neighbors
        node_dim = self.emb_dim + self.exog_dim + 1

        graph = Data(
            x=torch.zeros(n + 1, node_dim),
            edge_index=torch.zeros(2, n, dtype=torch.long),
            edge_attr=torch.zeros(n, self.edge_dim),
        )
        y = torch.zeros(self.config.seq_len)
        neighbor_seq = torch.zeros(n, self.config.seq_len, self.seq_in_channels)
        neighbor_mask = torch.zeros(n, dtype=torch.bool)
        target_seq = torch.zeros(self.config.seq_len, self.seq_in_channels)

        return graph, y, neighbor_seq, neighbor_mask, target_seq


def pretrain_collate_fn(batch):
    """
    Custom collate function matching DadsDataset collation.

    Filters out invalid samples (all-zero y) and batches the rest.
    """
    from torch_geometric.data import Batch

    graphs, ys, neighbor_seqs, neighbor_masks, target_seqs = zip(*batch)

    # Filter out invalid samples
    valid = [i for i, y in enumerate(ys) if y.abs().sum() > 0]
    if not valid:
        valid = [0]  # Keep at least one

    graphs = [graphs[i] for i in valid]
    ys = [ys[i] for i in valid]
    neighbor_seqs = [neighbor_seqs[i] for i in valid]
    neighbor_masks = [neighbor_masks[i] for i in valid]
    target_seqs = [target_seqs[i] for i in valid]

    batched_graph = Batch.from_data_list(graphs)
    y_batch = torch.stack(ys, dim=0)
    neighbor_seq_batch = torch.stack(neighbor_seqs, dim=0)
    neighbor_mask_batch = torch.stack(neighbor_masks, dim=0)
    target_seq_batch = torch.stack(target_seqs, dim=0)

    return (
        batched_graph,
        y_batch,
        neighbor_seq_batch,
        neighbor_mask_batch,
        target_seq_batch,
    )


class IterablePretrainDataset(torch.utils.data.IterableDataset):
    """
    Iterable version of PretrainDataset for efficient streaming.

    This is more memory-efficient for large datasets as it generates
    samples on-the-fly rather than pre-indexing all windows.
    """

    def __init__(
        self,
        epoch_sample: EpochSample,
        grid_index: GridIndex,
        seq_extractor: SequenceExtractor,
        config: PretrainConfig,
        variable: str,
        scaler: "MinMaxScaler",
        samples_per_epoch: int = 100000,
        seed: Optional[int] = None,
        inject_missingness: bool = True,
    ):
        """
        Initialize iterable dataset.

        Args:
            epoch_sample: Graph structure for this epoch
            grid_index: Spatial index
            seq_extractor: Sequence extractor
            config: Configuration
            variable: Target variable
            scaler: Scaler for normalization
            samples_per_epoch: Number of samples to generate per epoch
            seed: Random seed
            inject_missingness: Whether to inject synthetic missingness
        """
        self.sample = epoch_sample
        self.index = grid_index
        self.extractor = seq_extractor
        self.config = config
        self.variable = variable
        self.scaler = scaler
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.inject_missingness = inject_missingness

        # Dimensions
        self.terrain_dim = grid_index.terrain.shape[1]
        self.emb_dim = self.terrain_dim
        self.exog_dim = 3 + self.terrain_dim
        self.seq_in_channels = 1 + self.exog_dim
        self.edge_dim = self.terrain_dim + 2 + 1 + self.exog_dim

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single worker
            seed = self.seed
            n_samples = self.samples_per_epoch
        else:
            # Multiple workers - split samples
            per_worker = self.samples_per_epoch // worker_info.num_workers
            n_samples = per_worker
            seed = (self.seed or 0) + worker_info.id

        rng = np.random.default_rng(seed)

        start_str, end_str = self.config.date_range
        start_ord = np.datetime64(start_str, "D").astype(int) + self.config.seq_len
        end_ord = np.datetime64(end_str, "D").astype(int)

        for _ in range(n_samples):
            # Random target cell
            target_idx = rng.choice(self.sample.target_indices)
            end_date = np.datetime64(rng.integers(start_ord, end_ord), "D")

            # Generate sample (similar to PretrainDataset.__getitem__)
            sample = self._generate_sample(target_idx, end_date, rng)
            if sample is not None:
                yield sample

    def _generate_sample(self, cell_idx, end_date, rng):
        """Generate a single sample."""
        tgt_lat, tgt_lon = self.index.coords[cell_idx]
        tgt_terrain = self.index.terrain[cell_idx]
        tgt_row, tgt_col = self.index.grid_indices[cell_idx]

        tgt_seq, tgt_valid = self.extractor.extract_sequence(
            tgt_lat,
            tgt_lon,
            end_date,
            self.variable,
            tgt_terrain,
            row=int(tgt_row),
            col=int(tgt_col),
        )

        if not tgt_valid:
            return None

        # Scale and process (abbreviated - similar to PretrainDataset)
        tgt_seq_scaled = self._scale_sequence(tgt_seq)
        y = torch.from_numpy(tgt_seq_scaled[:, 0])

        neighbor_indices, bearings, distances, terrain_deltas = (
            self.sample.get_neighbors(cell_idx)
        )

        if len(neighbor_indices) < self.config.n_neighbors:
            return None

        # Build the rest of the sample...
        # (Following same logic as PretrainDataset.__getitem__)

        neighbor_seqs = []
        neighbor_masks = []
        neighbor_obs_today = []
        neighbor_exog_today = []

        for ni in neighbor_indices:
            nbr_lat, nbr_lon = self.index.coords[ni]
            nbr_terrain = self.index.terrain[ni]
            nbr_row, nbr_col = self.index.grid_indices[ni]

            nbr_seq, nbr_valid = self.extractor.extract_sequence(
                nbr_lat,
                nbr_lon,
                end_date,
                self.variable,
                nbr_terrain,
                row=int(nbr_row),
                col=int(nbr_col),
            )

            if (
                self.inject_missingness
                and rng.random() < self.config.neighbor_drop_prob
            ):
                nbr_valid = False

            if nbr_valid:
                nbr_seq_scaled = self._scale_sequence(nbr_seq)
                if self.inject_missingness:
                    mask = (
                        rng.random(self.config.seq_len) > self.config.timestep_mask_prob
                    )
                    nbr_seq_scaled[~mask] = 0.0

                neighbor_seqs.append(torch.from_numpy(nbr_seq_scaled))
                neighbor_masks.append(True)
                neighbor_obs_today.append(float(nbr_seq_scaled[-1, 0]))
                neighbor_exog_today.append(
                    torch.from_numpy(nbr_seq_scaled[-1, 1:].astype(np.float32))
                )
            else:
                neighbor_seqs.append(
                    torch.zeros(self.config.seq_len, self.seq_in_channels)
                )
                neighbor_masks.append(False)
                neighbor_obs_today.append(0.0)
                neighbor_exog_today.append(torch.zeros(self.exog_dim))

        neighbor_seq = torch.stack(neighbor_seqs, dim=0)
        neighbor_mask = torch.tensor(neighbor_masks, dtype=torch.bool)

        tgt_exog = torch.from_numpy(tgt_seq_scaled[-1, 1:].astype(np.float32))
        tgt_row_feat = torch.cat(
            [
                torch.zeros(self.emb_dim, dtype=torch.float32),
                tgt_exog,
                torch.zeros(1, dtype=torch.float32),
            ]
        )

        nbr_embs = torch.from_numpy(
            self.index.terrain[neighbor_indices].astype(np.float32)
        )
        nbr_exog_mat = torch.stack(neighbor_exog_today, dim=0)
        nbr_obs = torch.tensor(neighbor_obs_today, dtype=torch.float32).unsqueeze(1)
        nbr_rows = torch.cat([nbr_embs, nbr_exog_mat, nbr_obs], dim=1)

        x = torch.cat([tgt_row_feat.unsqueeze(0), nbr_rows], dim=0)

        terrain_delta = torch.from_numpy(terrain_deltas.astype(np.float32))
        bearing_rad = torch.tensor(
            [np.radians(b) for b in bearings], dtype=torch.float32
        )
        bearing_sc = torch.stack(
            [torch.sin(bearing_rad), torch.cos(bearing_rad)], dim=1
        )
        dist_scaled = torch.tensor(distances, dtype=torch.float32).unsqueeze(1) / 500.0
        exog_delta = tgt_exog.unsqueeze(0) - nbr_exog_mat

        edge_attr = torch.cat(
            [terrain_delta, bearing_sc, dist_scaled, exog_delta], dim=1
        )

        n_neighbors = len(neighbor_indices)
        edge_index = torch.stack(
            [
                torch.arange(1, n_neighbors + 1),
                torch.zeros(n_neighbors, dtype=torch.long),
            ],
            dim=0,
        )

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        target_seq = torch.zeros(
            self.config.seq_len, self.seq_in_channels, dtype=torch.float32
        )
        target_seq[:, 1:] = torch.from_numpy(tgt_seq_scaled[:, 1:].astype(np.float32))

        return graph, y, neighbor_seq, neighbor_mask, target_seq

    def _scale_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Apply MinMax scaling to sequence."""
        bias = np.asarray(self.scaler.bias).reshape(-1)
        scale = np.asarray(self.scaler.scale).reshape(-1)

        n_seq_features = seq.shape[1]
        n_scaler_features = len(bias)

        if n_seq_features <= n_scaler_features:
            bias = bias[:n_seq_features]
            scale = scale[:n_seq_features]
        else:
            pad_size = n_seq_features - n_scaler_features
            bias = np.concatenate([bias, np.zeros(pad_size)])
            scale = np.concatenate([scale, np.ones(pad_size)])

        scaled = (seq - bias) / scale + 5e-8
        return scaled.astype(np.float32)


if __name__ == "__main__":
    print("PretrainDataset module")
    print("Use with EpochSampler, GridIndex, and SequenceExtractor")
