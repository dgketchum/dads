"""
Dataset loader for DA benchmark HeteroData graphs.

Loads pre-built HeteroData .pt files with source/query node separation.
Applies z-score normalization to query and source features separately.
Attaches loss_mask on query nodes for transductive holdout.

Optionally applies a runtime source/query partition for training so that
no station can simultaneously be a DA source and a supervised query.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
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
    is_train : bool
        Whether this is a training dataset (enables runtime split).
    da_split_enabled : bool
        Enable runtime source/query partition for training.
    da_source_fraction : float
        Fraction of non-holdout stations assigned as sources (rest are queries).
    da_exclude_radius_km : float
        Remove source→query edges shorter than this distance.
    da_target_source_k : int | None
        After runtime split/radius filtering, keep at most the nearest K
        surviving source neighbors per query. ``None`` disables this cap.
    split_seed : int
        Base seed for deterministic per-epoch split re-randomization.
    da_mixed_local_enabled : bool
        Allow a small number of sub-exclusion-radius edges on a fraction of
        training graphs.
    da_mixed_local_graph_fraction : float
        Fraction of training graphs that use mixed-local mode.
    da_mixed_local_max_edges_per_query : int
        Max local (< exclusion radius) edges kept per query in mixed-local mode.
        The local/far boundary is always ``da_exclude_radius_km``.
    """

    def __init__(
        self,
        graph_dir: str,
        train_days: set | None = None,
        loss_fids: set | None = None,
        norm_stats: dict | None = None,
        target_index: int | None = None,
        is_train: bool = False,
        da_split_enabled: bool = False,
        da_source_fraction: float = 0.5,
        da_exclude_radius_km: float = 20.0,
        da_target_source_k: int | None = 16,
        split_seed: int = 42,
        da_mixed_local_enabled: bool = False,
        da_mixed_local_graph_fraction: float = 0.0,
        da_mixed_local_max_edges_per_query: int = 0,
    ):
        super().__init__()
        self.loss_fids = loss_fids
        self.target_index = target_index
        self.is_train = is_train
        self.da_split_enabled = da_split_enabled
        self.da_source_fraction = da_source_fraction
        self.da_exclude_radius_km = da_exclude_radius_km
        self.da_target_source_k = (
            int(da_target_source_k)
            if da_target_source_k is not None and int(da_target_source_k) > 0
            else None
        )
        self.split_seed = split_seed
        self.da_mixed_local_enabled = da_mixed_local_enabled
        self.da_mixed_local_graph_fraction = da_mixed_local_graph_fraction
        self.da_mixed_local_max_edges_per_query = da_mixed_local_max_edges_per_query
        self._epoch = 0

        # Load and verify metadata
        meta_path = os.path.join(graph_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No meta.json in {graph_dir}")
        with open(meta_path) as f:
            meta = json.load(f)

        family = meta.get("family", "")
        if family not in ("da-graph-v0", "da-graph-v1", "da-graph-v2", "da-graph-v3"):
            raise ValueError(f"Expected da-graph-v0/v1/v2/v3 family, got {family}")
        self._family = family

        self.query_feature_cols: list[str] = meta["query_feature_cols"]
        self.target_cols: list[str] = meta["target_cols"]
        self._sq_edge_norm = meta.get("source_query_edge_norm", {})
        self.source_candidate_k = meta.get("source_k")
        self.source_candidate_radius_km = meta.get(
            "source_max_radius_km",
            meta.get("max_radius_km"),
        )

        # v1: separate context/payload; v0: combined source_feature_cols
        if family in ("da-graph-v1", "da-graph-v2", "da-graph-v3"):
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
    def effective_family(self) -> str:
        """Runtime family label reflecting source masking policy."""
        if self.da_split_enabled and self.da_mixed_local_enabled:
            return "da-graph-v3-mixed-local"
        if self.da_split_enabled:
            return "da-graph-v3"
        return self._family

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
            if self._family in ("da-graph-v1", "da-graph-v2", "da-graph-v3"):
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

    def set_epoch(self, epoch: int) -> None:
        """Update epoch for per-epoch split re-randomization."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self._graphs)

    def _apply_split(
        self,
        g: HeteroData,
        idx: int,
        s_ctx: torch.Tensor,
        s_pay: torch.Tensor | None,
        sq_edge_index: torch.Tensor,
        sq_edge_attr: torch.Tensor,
        loss_mask: torch.Tensor,
        q_valid: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Apply runtime source/query partition for training.

        Returns updated (s_ctx, s_pay, sq_edge_index, sq_edge_attr, loss_mask).
        """
        q_fids = list(g["query"].fids)
        s_fids = list(g["source"].fids)
        n_src = len(s_fids)

        # Identify non-holdout query fids that also exist as sources
        holdout_fids = (
            set() if self.loss_fids is None else (set(q_fids) - self.loss_fids)
        )
        eligible = [i for i, f in enumerate(s_fids) if f not in holdout_fids]
        if len(eligible) < 2:
            # Sparse day: not enough stations for a split.  Keep sources as-is
            # but zero out the loss_mask so no station is both source and
            # supervised query (bg-only supervision via holdout-only loss).
            loss_mask = torch.zeros_like(loss_mask)
            return s_ctx, s_pay, sq_edge_index, sq_edge_attr, loss_mask

        # Deterministic per-graph, per-epoch partition
        rng = np.random.default_rng(self.split_seed + self._epoch * 100_000 + idx)
        n_source = max(1, int(round(len(eligible) * self.da_source_fraction)))
        n_source = min(n_source, len(eligible) - 1)
        perm = rng.permutation(len(eligible))
        source_eligible_idx = set(perm[:n_source].tolist())

        # Map back to source-node indices
        source_keep = set()  # source node indices to keep
        query_designated_fids = set()  # fids that should be supervised as queries
        for ei, si in enumerate(eligible):
            if ei in source_eligible_idx:
                source_keep.add(si)
            else:
                query_designated_fids.add(s_fids[si])

        # Build source node mask and reindex
        src_mask = torch.zeros(n_src, dtype=torch.bool)
        for si in source_keep:
            src_mask[si] = True
        new_src_idx = torch.full((n_src,), -1, dtype=torch.long)
        new_src_idx[src_mask] = torch.arange(src_mask.sum())

        # Filter source tensors
        s_ctx = s_ctx[src_mask]
        if s_pay is not None:
            s_pay = s_pay[src_mask]

        # Reindex source→query edges
        old_src, old_dst = sq_edge_index[0], sq_edge_index[1]
        edge_keep = src_mask[old_src]
        sq_edge_index = torch.stack(
            [new_src_idx[old_src[edge_keep]], old_dst[edge_keep]]
        )
        sq_edge_attr = sq_edge_attr[edge_keep]

        # Decide whether this graph uses mixed-local mode
        use_mixed_local = self.da_mixed_local_enabled and (
            rng.random() < self.da_mixed_local_graph_fraction
        )

        sq_edge_index, sq_edge_attr = self._prune_source_edges(
            sq_edge_index, sq_edge_attr, mixed_local=use_mixed_local
        )

        # Update loss_mask: only query-designated non-holdout stations get train loss
        q_fid_to_idx = {}
        for qi, f in enumerate(q_fids):
            q_fid_to_idx.setdefault(f, []).append(qi)

        new_loss_mask = torch.zeros_like(loss_mask)
        for f in query_designated_fids:
            for qi in q_fid_to_idx.get(f, []):
                if q_valid[qi] if q_valid.ndim == 1 else q_valid[qi].any():
                    new_loss_mask[qi] = True
        loss_mask = new_loss_mask

        return s_ctx, s_pay, sq_edge_index, sq_edge_attr, loss_mask

    def _prune_source_edges(
        self,
        sq_edge_index: torch.Tensor,
        sq_edge_attr: torch.Tensor,
        mixed_local: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply distance floor and nearest-surviving-K cap to source edges.

        When *mixed_local* is True, a limited number of sub-exclusion-radius
        ("local") edges are retained per query before filling the remaining
        budget from far-field edges.
        """
        if sq_edge_index.numel() == 0:
            return sq_edge_index, sq_edge_attr

        dist_mean = self._sq_edge_norm.get("dist_mean", 0.0)
        dist_std = self._sq_edge_norm.get("dist_std", 1.0)
        dist_km = sq_edge_attr[:, 0] * dist_std + dist_mean

        if mixed_local and self.da_mixed_local_max_edges_per_query > 0:
            return self._prune_mixed_local(sq_edge_index, sq_edge_attr, dist_km)

        # --- strict mode: remove all edges below exclusion radius ---
        if self.da_exclude_radius_km > 0:
            radius_keep = dist_km >= self.da_exclude_radius_km
            sq_edge_index = sq_edge_index[:, radius_keep]
            sq_edge_attr = sq_edge_attr[radius_keep]
            dist_km = dist_km[radius_keep]
            if sq_edge_index.numel() == 0:
                return sq_edge_index, sq_edge_attr

        if self.da_target_source_k is None:
            return sq_edge_index, sq_edge_attr

        old_dst = sq_edge_index[1]
        keep_mask = torch.zeros(old_dst.shape[0], dtype=torch.bool)
        unique_dst = torch.unique(old_dst)
        for qi in unique_dst.tolist():
            edge_idx = torch.nonzero(old_dst == qi, as_tuple=False).squeeze(1)
            if edge_idx.numel() <= self.da_target_source_k:
                keep_mask[edge_idx] = True
                continue
            nearest = torch.argsort(dist_km[edge_idx])[: self.da_target_source_k]
            keep_mask[edge_idx[nearest]] = True

        return sq_edge_index[:, keep_mask], sq_edge_attr[keep_mask]

    def _prune_mixed_local(
        self,
        sq_edge_index: torch.Tensor,
        sq_edge_attr: torch.Tensor,
        dist_km: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep up to *max_local* local edges + fill from far up to K total."""
        local_thresh = self.da_exclude_radius_km
        max_local = self.da_mixed_local_max_edges_per_query
        k_total = self.da_target_source_k

        is_local = dist_km < local_thresh
        is_far = ~is_local

        old_dst = sq_edge_index[1]
        keep_mask = torch.zeros(old_dst.shape[0], dtype=torch.bool)

        for qi in torch.unique(old_dst).tolist():
            qi_edges = torch.nonzero(old_dst == qi, as_tuple=False).squeeze(1)
            local_edges = qi_edges[is_local[qi_edges]]
            far_edges = qi_edges[is_far[qi_edges]]

            # Nearest local edges, capped
            kept_local = torch.tensor([], dtype=torch.long)
            if local_edges.numel() > 0:
                order = torch.argsort(dist_km[local_edges])
                kept_local = local_edges[order[:max_local]]

            # Fill remaining budget from far edges
            remaining = (k_total - kept_local.numel()) if k_total else far_edges.numel()
            kept_far = torch.tensor([], dtype=torch.long)
            if far_edges.numel() > 0 and remaining > 0:
                order = torch.argsort(dist_km[far_edges])
                kept_far = far_edges[order[:remaining]]

            if kept_local.numel() > 0:
                keep_mask[kept_local] = True
            if kept_far.numel() > 0:
                keep_mask[kept_far] = True

        return sq_edge_index[:, keep_mask], sq_edge_attr[keep_mask]

    def __getitem__(self, idx: int) -> HeteroData:
        g = self._graphs[idx]

        # Clone tensors
        qx = g["query"].x.clone()
        qy = g["query"].y.clone()
        q_valid = g["query"].valid_mask.clone()

        if self._family in ("da-graph-v1", "da-graph-v2", "da-graph-v3"):
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

        # Clone edge tensors
        sq_edge_index = g["source", "influences", "query"].edge_index.clone()
        sq_edge_attr = g["source", "influences", "query"].edge_attr.clone()

        # Runtime source/query split (train only)
        if self.is_train and self.da_split_enabled:
            s_ctx, s_pay, sq_edge_index, sq_edge_attr, loss_mask = self._apply_split(
                g, idx, s_ctx, s_pay, sq_edge_index, sq_edge_attr, loss_mask, q_valid
            )

        # Build output HeteroData
        out = HeteroData()
        out["query"].x = qx
        out["query"].y = qy
        out["query"].loss_mask = loss_mask

        # For multitask: attach valid_mask separately
        if q_valid.ndim == 2:
            out["query"].valid_mask = q_valid

        if self._family in ("da-graph-v1", "da-graph-v2", "da-graph-v3"):
            out["source"].context_x = s_ctx
            out["source"].payload_x = s_pay
            out["source"].num_nodes = s_ctx.shape[0]
        else:
            out["source"].x = s_ctx  # v0 compat

        out["source", "influences", "query"].edge_index = sq_edge_index
        out["source", "influences", "query"].edge_attr = sq_edge_attr
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
