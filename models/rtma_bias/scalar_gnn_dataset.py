"""
Precomputed graph dataset for scalar bias-correction GNNs.

Loads per-day .pt graphs from disk (built by prep/build_graphs.py).
Applies z-score normalization at __getitem__ time.

Spatial holdout is **transductive**: all nodes stay in the graph for message
passing; a ``loss_mask`` boolean tensor controls which nodes contribute to the
loss.  This preserves realistic graph topology for both train and val.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None


class PrecomputedGraphDataset(Dataset):
    """Load precomputed per-day .pt graphs from disk.

    Parameters
    ----------
    graph_dir : str
        Directory containing .pt files and meta.json.
    use_graph : bool
        If False, strip edges (MLP mode).
    norm_stats : dict or None
        Precomputed {col: {mean, std}}. If None, computed from data.
    train_days : set or None
        Restrict to these days only.
    loss_fids : set or None
        Station fids whose nodes contribute to loss. If None, all nodes
        contribute.  All nodes remain in the graph regardless of this setting
        (transductive holdout).
    """

    def __init__(
        self,
        graph_dir: str,
        use_graph: bool = True,
        norm_stats: dict | None = None,
        train_days: set | None = None,
        loss_fids: set | None = None,
    ):
        super().__init__()
        self.use_graph = use_graph
        self.loss_fids = loss_fids

        # Load metadata
        with open(os.path.join(graph_dir, "meta.json")) as f:
            meta = json.load(f)
        self.feature_cols: list[str] = meta["all_feature_cols"]
        self.target_cols: list[str] = meta["target_cols"]

        # Glob and filter .pt files
        pt_files = sorted(Path(graph_dir).glob("*.pt"))
        if train_days is not None:
            train_day_strs = {pd.Timestamp(d).strftime("%Y-%m-%d") for d in train_days}
            pt_files = [p for p in pt_files if p.stem in train_day_strs]

        # Preload all graphs into RAM
        self._graphs: list[Data] = []
        for p in pt_files:
            self._graphs.append(torch.load(p, weights_only=False))

        # Compute norm stats from loaded data if not provided
        if norm_stats is None:
            self.norm_stats = self._compute_norm_stats()
        else:
            self.norm_stats = norm_stats

    def _compute_norm_stats(self) -> dict[str, dict[str, float]]:
        """Compute z-score stats from the loaded (raw) data."""
        xs = [g.x for g in self._graphs]
        if not xs:
            return {}
        all_x = torch.cat(xs, dim=0)
        stats = {}
        for i, c in enumerate(self.feature_cols):
            vals = all_x[:, i]
            stats[c] = {
                "mean": float(vals.mean()),
                "std": float(max(vals.std().item(), 1e-8)),
            }
        return stats

    def __len__(self) -> int:
        return len(self._graphs)

    @property
    def node_dim(self) -> int:
        return len(self.feature_cols)

    @property
    def edge_dim(self) -> int:
        return 7

    def __getitem__(self, idx: int):
        g = self._graphs[idx]

        x = g.x.clone()
        y = g.y.clone()
        edge_index = g.edge_index.clone() if self.use_graph else None
        edge_attr = g.edge_attr.clone() if self.use_graph else None
        fids = g.fids

        # Transductive loss mask: all nodes stay, mask selects loss contributors
        if self.loss_fids is not None:
            loss_mask = torch.tensor(
                [f in self.loss_fids for f in fids], dtype=torch.bool
            )
        else:
            loss_mask = torch.ones(len(fids), dtype=torch.bool)

        # Apply z-score normalization
        for i, c in enumerate(self.feature_cols):
            if c in self.norm_stats:
                x[:, i] = (x[:, i] - self.norm_stats[c]["mean"]) / self.norm_stats[c][
                    "std"
                ]

        n_nodes = x.shape[0]
        if not self.use_graph or edge_index is None:
            return Data(x=x, y=y, loss_mask=loss_mask, num_nodes=n_nodes)

        return Data(
            x=x,
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr,
            loss_mask=loss_mask,
            num_nodes=n_nodes,
        )

    def save_norm_stats(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "feature_cols": self.feature_cols,
                    "norm_stats": self.norm_stats,
                },
                f,
                indent=2,
            )

    @staticmethod
    def load_norm_stats(path: str) -> dict:
        with open(path) as f:
            return json.load(f)
