"""
Dataset loader for DA benchmark HeteroData graphs (da-graph-v0).

Loads pre-built HeteroData .pt files with source/query node separation.
Applies z-score normalization to query and source features separately.
Attaches loss_mask on query nodes for transductive holdout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


class DAGraphDataset(Dataset):
    """Load precomputed DA hetero graphs from disk.

    Parameters
    ----------
    graph_dir : str
        Directory with .pt HeteroData files and meta.json.
    train_days : set | None
        Filter to these days only.
    loss_fids : set | None
        Query fids that contribute to loss (non-holdout for train, holdout for val).
    norm_stats : dict | None
        Pre-computed {col: {mean, std}} for query and source features.
    target_index : int | None
        Select a single target column from multi-target y. None = all targets.
    """

    def __init__(
        self,
        graph_dir: str,
        train_days: set | None = None,
        loss_fids: set | None = None,
        norm_stats: dict | None = None,
        target_index: int | None = None,
    ):
        super().__init__()
        self.loss_fids = loss_fids
        self.target_index = target_index

        # Load and verify metadata
        meta_path = os.path.join(graph_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No meta.json in {graph_dir}")
        with open(meta_path) as f:
            meta = json.load(f)

        if meta.get("family") != "da-graph-v0":
            raise ValueError(f"Expected da-graph-v0 family, got {meta.get('family')}")

        self.query_feature_cols: list[str] = meta["query_feature_cols"]
        self.source_feature_cols: list[str] = meta["source_feature_cols"]
        self.target_cols: list[str] = meta["target_cols"]

        # Load graphs
        pt_files = sorted(Path(graph_dir).glob("*.pt"))
        if train_days is not None:
            day_strs = {pd.Timestamp(d).strftime("%Y-%m-%d") for d in train_days}
            pt_files = [p for p in pt_files if p.stem in day_strs]

        self._graphs: list[HeteroData] = []
        for p in pt_files:
            self._graphs.append(torch.load(p, weights_only=False))

        # Compute or load normalization stats
        if norm_stats is None:
            self.norm_stats = self._compute_norm_stats()
        else:
            self.norm_stats = norm_stats

    @property
    def query_node_dim(self) -> int:
        return len(self.query_feature_cols)

    @property
    def source_node_dim(self) -> int:
        return len(self.source_feature_cols)

    @property
    def edge_dim(self) -> int:
        return 7

    def _compute_norm_stats(self) -> dict:
        """Compute z-score stats from loss-eligible (non-holdout) query nodes
        and all source nodes (source nodes are already non-holdout by construction)."""
        query_xs = []
        source_xs = []
        for g in self._graphs:
            qx = g["query"].x
            if self.loss_fids is not None and hasattr(g["query"], "fids"):
                mask = torch.tensor(
                    [f in self.loss_fids for f in g["query"].fids], dtype=torch.bool
                )
                if mask.any():
                    query_xs.append(qx[mask])
            else:
                query_xs.append(qx)
            source_xs.append(g["source"].x)

        stats = {"query": {}, "source": {}}
        if query_xs:
            all_qx = torch.cat(query_xs, dim=0)
            for i, c in enumerate(self.query_feature_cols):
                vals = all_qx[:, i]
                stats["query"][c] = {
                    "mean": float(vals.mean()),
                    "std": float(max(vals.std().item(), 1e-8)),
                }
        if source_xs:
            all_sx = torch.cat(source_xs, dim=0)
            for i, c in enumerate(self.source_feature_cols):
                vals = all_sx[:, i]
                stats["source"][c] = {
                    "mean": float(vals.mean()),
                    "std": float(max(vals.std().item(), 1e-8)),
                }
        return stats

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx: int) -> HeteroData:
        g = self._graphs[idx]

        # Clone tensors
        qx = g["query"].x.clone()
        qy = g["query"].y.clone()
        q_valid = g["query"].valid_mask.clone()
        sx = g["source"].x.clone()

        # Normalize query features
        for i, c in enumerate(self.query_feature_cols):
            s = self.norm_stats.get("query", {}).get(c)
            if s:
                qx[:, i] = (qx[:, i] - s["mean"]) / s["std"]

        # Normalize source features
        for i, c in enumerate(self.source_feature_cols):
            s = self.norm_stats.get("source", {}).get(c)
            if s:
                sx[:, i] = (sx[:, i] - s["mean"]) / s["std"]

        # Target index selection for single-head
        if self.target_index is not None and qy.ndim == 2:
            qy = qy[:, self.target_index]
            q_valid = q_valid[:, self.target_index]

        # Loss mask on query nodes
        if self.loss_fids is not None and hasattr(g["query"], "fids"):
            loss_mask = torch.tensor(
                [f in self.loss_fids for f in g["query"].fids], dtype=torch.bool
            )
            # For single-head, combine with valid_mask
            if q_valid.ndim == 1:
                loss_mask = loss_mask & q_valid
        else:
            loss_mask = torch.ones(qx.shape[0], dtype=torch.bool)

        # Build output HeteroData
        out = HeteroData()
        out["query"].x = qx
        out["query"].y = qy
        out["query"].loss_mask = loss_mask

        # For multitask: attach valid_mask separately
        if q_valid.ndim == 2:
            out["query"].valid_mask = q_valid

        out["source"].x = sx

        out["source", "influences", "query"].edge_index = g[
            "source", "influences", "query"
        ].edge_index.clone()
        out["source", "influences", "query"].edge_attr = g[
            "source", "influences", "query"
        ].edge_attr.clone()
        out["query", "neighbors", "query"].edge_index = g[
            "query", "neighbors", "query"
        ].edge_index.clone()
        out["query", "neighbors", "query"].edge_attr = g[
            "query", "neighbors", "query"
        ].edge_attr.clone()

        return out

    def save_norm_stats(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.norm_stats, f, indent=2)
