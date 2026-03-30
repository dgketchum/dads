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

        family = meta.get("family", "")
        if family not in ("da-graph-v0", "da-graph-v1", "da-graph-v2"):
            raise ValueError(f"Expected da-graph-v0/v1/v2 family, got {family}")
        self._family = family

        self.query_feature_cols: list[str] = meta["query_feature_cols"]
        self.target_cols: list[str] = meta["target_cols"]

        # v1: separate context/payload; v0: combined source_feature_cols
        if family in ("da-graph-v1", "da-graph-v2"):
            self.source_context_feature_cols: list[str] = meta[
                "source_context_feature_cols"
            ]
            self.source_payload_feature_cols: list[str] = meta[
                "source_payload_feature_cols"
            ]
        else:
            self.source_context_feature_cols = []
            self.source_payload_feature_cols = []
            self._source_feature_cols_v0: list[str] = meta.get(
                "source_feature_cols", []
            )

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
    def source_context_dim(self) -> int:
        return len(self.source_context_feature_cols)

    @property
    def source_payload_dim(self) -> int:
        return len(self.source_payload_feature_cols)

    @property
    def edge_dim(self) -> int:
        return 7

    def _compute_norm_stats(self) -> dict:
        """Compute z-score stats from loss-eligible (non-holdout) query nodes
        and all source nodes (source nodes are already non-holdout by construction)."""
        query_xs = []
        source_ctx_xs = []
        source_pay_xs = []
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
            if self._family in ("da-graph-v1", "da-graph-v2"):
                source_ctx_xs.append(g["source"].context_x)
                source_pay_xs.append(g["source"].payload_x)
            else:
                source_ctx_xs.append(g["source"].x)

        stats = {"query": {}, "source_context": {}, "source_payload": {}}
        if query_xs:
            all_qx = torch.cat(query_xs, dim=0)
            for i, c in enumerate(self.query_feature_cols):
                vals = all_qx[:, i]
                stats["query"][c] = {
                    "mean": float(vals.mean()),
                    "std": float(max(vals.std().item(), 1e-8)),
                }
        if source_ctx_xs:
            all_ctx = torch.cat(source_ctx_xs, dim=0)
            cols = self.source_context_feature_cols or self._source_feature_cols_v0
            for i, c in enumerate(cols):
                if i < all_ctx.shape[1]:
                    vals = all_ctx[:, i]
                    stats["source_context"][c] = {
                        "mean": float(vals.mean()),
                        "std": float(max(vals.std().item(), 1e-8)),
                    }
        if source_pay_xs:
            all_pay = torch.cat(source_pay_xs, dim=0)
            for i, c in enumerate(self.source_payload_feature_cols):
                if i < all_pay.shape[1]:
                    # Don't z-score valid flags (binary 0/1)
                    if c.startswith("valid_"):
                        continue
                    vals = all_pay[:, i]
                    stats["source_payload"][c] = {
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

        if self._family in ("da-graph-v1", "da-graph-v2"):
            s_ctx = g["source"].context_x.clone()
            s_pay = g["source"].payload_x.clone()
        else:
            s_ctx = g["source"].x.clone()
            s_pay = None

        # Normalize query features
        for i, c in enumerate(self.query_feature_cols):
            s = self.norm_stats.get("query", {}).get(c)
            if s:
                qx[:, i] = (qx[:, i] - s["mean"]) / s["std"]

        # Normalize source context
        ctx_cols = self.source_context_feature_cols or getattr(
            self, "_source_feature_cols_v0", []
        )
        for i, c in enumerate(ctx_cols):
            ns = self.norm_stats.get("source_context", {}).get(c)
            if ns and i < s_ctx.shape[1]:
                s_ctx[:, i] = (s_ctx[:, i] - ns["mean"]) / ns["std"]

        # Normalize source payload (skip valid flags)
        if s_pay is not None:
            for i, c in enumerate(self.source_payload_feature_cols):
                if c.startswith("valid_"):
                    continue
                ns = self.norm_stats.get("source_payload", {}).get(c)
                if ns and i < s_pay.shape[1]:
                    s_pay[:, i] = (s_pay[:, i] - ns["mean"]) / ns["std"]

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

        if self._family in ("da-graph-v1", "da-graph-v2"):
            out["source"].context_x = s_ctx
            out["source"].payload_x = s_pay
            out["source"].num_nodes = s_ctx.shape[0]
        else:
            out["source"].x = s_ctx  # v0 compat

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
