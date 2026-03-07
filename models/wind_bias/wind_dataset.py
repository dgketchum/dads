"""
Wind graph dataset — one PyG Data object per day.

Loads the full station-day Parquet into memory at init, groups by day,
and builds per-day graphs from the precomputed k-NN neighbor map.
Also provides PrecomputedWindDataset for loading pre-built .pt graphs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from prep.graph_utils import (
    build_edges_for_day,
    build_knn_map,
    build_static_edge_attrs,
    compute_edge_norm,
)

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None


# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

RTMA_WEATHER_COLS = [
    "ugrd_rtma",
    "vgrd_rtma",
    "wind_rtma",
    "tmp_rtma",
    "dpt_rtma",
    "pres_rtma",
    "tcdc_rtma",
    "prcp_rtma",
    "ea_rtma",
    "tmp_dpt_diff",
]

TERRAIN_COLS = [
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "tpi_4",
    "tpi_10",
]

SX_DERIVED_COLS = ["terrain_openness", "terrain_directionality"]

FLOW_TERRAIN_COLS = ["flow_upslope", "flow_cross", "wind_aligned_sx"]

TEMPORAL_COLS = ["doy_sin", "doy_cos"]

LOCATION_COLS = ["latitude", "longitude"]

TARGET_COLS = ["delta_w_par", "delta_w_perp"]


def _get_sx_cols(df: pd.DataFrame) -> list[str]:
    return sorted(
        c for c in df.columns if c.startswith("sx_") and ("_2k" in c or "_10k" in c)
    )


def _get_feature_cols(
    df: pd.DataFrame,
    use_sx: bool = True,
    use_flow_terrain: bool = True,
) -> list[str]:
    """Build ordered list of node feature columns."""
    cols = []
    for c in RTMA_WEATHER_COLS:
        if c in df.columns:
            cols.append(c)
    for c in TERRAIN_COLS:
        if c in df.columns:
            cols.append(c)
    if use_sx:
        cols.extend(_get_sx_cols(df))
        for c in SX_DERIVED_COLS:
            if c in df.columns:
                cols.append(c)
    if use_flow_terrain:
        for c in FLOW_TERRAIN_COLS:
            if c in df.columns:
                cols.append(c)
    for c in TEMPORAL_COLS:
        if c in df.columns:
            cols.append(c)
    for c in LOCATION_COLS:
        if c in df.columns:
            cols.append(c)
    return cols


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def compute_norm_stats(
    df: pd.DataFrame, feature_cols: list[str]
) -> dict[str, dict[str, float]]:
    """Compute per-column mean/std for z-score normalisation."""
    stats = {}
    for c in feature_cols:
        if c in df.columns:
            vals = df[c].dropna()
            stats[c] = {
                "mean": float(vals.mean()),
                "std": float(max(vals.std(), 1e-8)),
            }
    return stats


def apply_norm(arr: np.ndarray, cols: list[str], stats: dict) -> np.ndarray:
    """Z-score normalise an array using precomputed stats."""
    out = arr.copy().astype("float32")
    for i, c in enumerate(cols):
        if c in stats:
            out[:, i] = (out[:, i] - stats[c]["mean"]) / stats[c]["std"]
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class WindGraphDataset(Dataset):
    """One PyG Data per day.

    Parameters
    ----------
    table_path : str
        Path to station_day_wind Parquet.
    stations_csv : str
        Station inventory CSV.
    k : int
        k-NN neighbors.
    use_graph : bool
        If False, return Data without edges (MLP-only mode).
    use_sx : bool
        Include Sx features.
    use_flow_terrain : bool
        Include flow-terrain interaction features.
    norm_stats : dict or None
        Precomputed normalisation stats.  Computed from data if None.
    train_days : set or None
        If provided, only include these days.
    exclude_fids : set or None
        Station fids to exclude (spatial holdout).
    """

    def __init__(
        self,
        table_path: str,
        stations_csv: str,
        k: int = 16,
        use_graph: bool = True,
        use_sx: bool = True,
        use_flow_terrain: bool = True,
        norm_stats: dict | None = None,
        train_days: set | None = None,
        exclude_fids: set | None = None,
    ):
        super().__init__()
        self.use_graph = use_graph
        self.use_sx = use_sx
        self.use_flow_terrain = use_flow_terrain
        self.k = k

        # Load full table
        df = pd.read_parquet(table_path)
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        df["fid"] = df["fid"].astype(str)
        df["day"] = pd.to_datetime(df["day"])

        if train_days is not None:
            df = df[df["day"].isin(train_days)]
        if exclude_fids is not None:
            df = df[~df["fid"].isin(exclude_fids)]

        self.feature_cols = _get_feature_cols(
            df, use_sx=use_sx, use_flow_terrain=use_flow_terrain
        )
        self.target_cols = [c for c in TARGET_COLS if c in df.columns]

        # Fill NaN features
        for c in self.feature_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0.0)

        # Normalisation
        if norm_stats is None:
            self.norm_stats = compute_norm_stats(df, self.feature_cols)
        else:
            self.norm_stats = norm_stats

        # Group by day
        self.day_groups: dict[str, pd.DataFrame] = {}
        for day, grp in df.groupby("day"):
            self.day_groups[str(day)] = grp
        self.days = sorted(self.day_groups.keys())

        # k-NN and edge attrs
        if use_graph:
            self.knn_map = build_knn_map(stations_csv, k=k)
            self.static_edges = build_static_edge_attrs(stations_csv, self.knn_map)
            self._edge_norm = compute_edge_norm(self.static_edges)
        else:
            self.knn_map = {}
            self.static_edges = {}

    def __len__(self) -> int:
        return len(self.days)

    @property
    def node_dim(self) -> int:
        return len(self.feature_cols)

    @property
    def edge_dim(self) -> int:
        return 7  # distance_norm, bearing_sin, bearing_cos, delta_elev, delta_tpi, upwind_cos, upwind_sin

    def __getitem__(self, idx: int) -> Any:
        day_key = self.days[idx]
        day_df = self.day_groups[day_key]
        fids = day_df["fid"].values

        # Node features
        x_raw = day_df[self.feature_cols].values.astype("float32")
        x = apply_norm(x_raw, self.feature_cols, self.norm_stats)

        # Targets
        y = day_df[self.target_cols].values.astype("float32")

        # Baseline wind for metrics/loss weighting
        baseline_wind = (
            np.sqrt(
                day_df["ugrd_rtma"].values ** 2 + day_df["vgrd_rtma"].values ** 2
            ).astype("float32")
            if "ugrd_rtma" in day_df.columns
            else np.zeros(len(fids), dtype="float32")
        )

        # Raw u/v for speed/direction evaluation metrics
        u_obs = (
            day_df["u_obs"].values.astype("float32")
            if "u_obs" in day_df.columns
            else np.zeros(len(fids), dtype="float32")
        )
        v_obs = (
            day_df["v_obs"].values.astype("float32")
            if "v_obs" in day_df.columns
            else np.zeros(len(fids), dtype="float32")
        )
        ugrd_baseline = (
            day_df["ugrd_rtma"].values.astype("float32")
            if "ugrd_rtma" in day_df.columns
            else np.zeros(len(fids), dtype="float32")
        )
        vgrd_baseline = (
            day_df["vgrd_rtma"].values.astype("float32")
            if "vgrd_rtma" in day_df.columns
            else np.zeros(len(fids), dtype="float32")
        )

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        baseline_wind_t = torch.from_numpy(baseline_wind)
        u_obs_t = torch.from_numpy(u_obs)
        v_obs_t = torch.from_numpy(v_obs)
        ugrd_baseline_t = torch.from_numpy(ugrd_baseline)
        vgrd_baseline_t = torch.from_numpy(vgrd_baseline)

        if not self.use_graph:
            return Data(
                x=x_t,
                y=y_t,
                baseline_wind=baseline_wind_t,
                u_obs=u_obs_t,
                v_obs=v_obs_t,
                ugrd_baseline=ugrd_baseline_t,
                vgrd_baseline=vgrd_baseline_t,
                num_nodes=len(fids),
            )

        # Build edges via shared helper
        ugrd = day_df["ugrd_rtma"].values if "ugrd_rtma" in day_df.columns else None
        vgrd = day_df["vgrd_rtma"].values if "vgrd_rtma" in day_df.columns else None
        edge_index, edge_attr = build_edges_for_day(
            fids=list(fids),
            ugrd=ugrd,
            vgrd=vgrd,
            knn_map=self.knn_map,
            static_edges=self.static_edges,
            edge_norm=self._edge_norm,
        )

        return Data(
            x=x_t,
            y=y_t,
            edge_index=edge_index,
            edge_attr=edge_attr,
            baseline_wind=baseline_wind_t,
            u_obs=u_obs_t,
            v_obs=v_obs_t,
            ugrd_baseline=ugrd_baseline_t,
            vgrd_baseline=vgrd_baseline_t,
            num_nodes=len(fids),
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


# ---------------------------------------------------------------------------
# Precomputed graph dataset
# ---------------------------------------------------------------------------


class PrecomputedWindDataset(Dataset):
    """Load precomputed per-day .pt graphs from disk.

    All .pt files contain raw (unnormalized) features for ALL stations.
    Normalization, feature column selection, and spatial holdout filtering
    are applied at __getitem__ time, keeping precomputation split-agnostic.

    Parameters
    ----------
    graph_dir : str
        Directory containing .pt files and meta.json.
    use_sx : bool
        Include Sx features in node input.
    use_flow_terrain : bool
        Include flow-terrain interaction features.
    use_graph : bool
        If False, strip edges (MLP mode).
    norm_stats : dict or None
        Precomputed {col: {mean, std}}. If None, computed from data.
    train_days : set or None
        Restrict to these days only.
    exclude_fids : set or None
        Station fids to exclude (spatial holdout).
    """

    def __init__(
        self,
        graph_dir: str,
        use_sx: bool = True,
        use_flow_terrain: bool = True,
        use_graph: bool = True,
        norm_stats: dict | None = None,
        train_days: set | None = None,
        exclude_fids: set | None = None,
    ):
        super().__init__()
        self.use_sx = use_sx
        self.use_flow_terrain = use_flow_terrain
        self.use_graph = use_graph
        self.exclude_fids = exclude_fids

        # Load metadata
        with open(os.path.join(graph_dir, "meta.json")) as f:
            meta = json.load(f)
        self._all_feature_cols: list[str] = meta["all_feature_cols"]
        self.target_cols: list[str] = meta["target_cols"]

        # Determine which feature columns to use (subset of all_feature_cols)
        self.feature_cols = self._select_feature_cols()
        self._col_indices = [self._all_feature_cols.index(c) for c in self.feature_cols]

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

    def _select_feature_cols(self) -> list[str]:
        """Select feature columns based on use_sx / use_flow_terrain flags."""
        sx_cols_set = set()
        for c in self._all_feature_cols:
            if c.startswith("sx_") and ("_2k" in c or "_10k" in c):
                sx_cols_set.add(c)
        sx_cols_set.update(SX_DERIVED_COLS)
        flow_cols_set = set(FLOW_TERRAIN_COLS)

        cols = []
        for c in self._all_feature_cols:
            if not self.use_sx and c in sx_cols_set:
                continue
            if not self.use_flow_terrain and c in flow_cols_set:
                continue
            cols.append(c)
        return cols

    def _compute_norm_stats(self) -> dict[str, dict[str, float]]:
        """Compute z-score stats from the loaded (raw) training data."""
        # Stack all x tensors, select feature columns
        col_idx = torch.tensor(self._col_indices, dtype=torch.long)
        xs = []
        for g in self._graphs:
            xs.append(g.x[:, col_idx])
        if not xs:
            return {}
        all_x = torch.cat(xs, dim=0)  # (total_nodes, n_features)
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

    def __getitem__(self, idx: int) -> Any:
        g = self._graphs[idx]

        # Select feature columns
        col_idx = self._col_indices
        x = g.x[:, col_idx].clone()
        y = g.y.clone()
        baseline_wind = g.baseline_wind.clone()
        edge_index = g.edge_index.clone() if self.use_graph else None
        edge_attr = g.edge_attr.clone() if self.use_graph else None
        fids = g.fids  # list[str]

        # Spatial holdout: filter out excluded fids
        if self.exclude_fids:
            keep = torch.tensor(
                [f not in self.exclude_fids for f in fids], dtype=torch.bool
            )
            if not keep.all():
                x = x[keep]
                y = y[keep]
                baseline_wind = baseline_wind[keep]
                if edge_index is not None and edge_index.numel() > 0:
                    # Remap node indices
                    old_to_new = torch.full((len(fids),), -1, dtype=torch.long)
                    old_to_new[keep] = torch.arange(keep.sum(), dtype=torch.long)
                    src, dst = edge_index[0], edge_index[1]
                    valid = keep[src] & keep[dst]
                    edge_index = torch.stack(
                        [old_to_new[src[valid]], old_to_new[dst[valid]]]
                    )
                    edge_attr = edge_attr[valid]

        # Apply z-score normalization
        for i, c in enumerate(self.feature_cols):
            if c in self.norm_stats:
                x[:, i] = (x[:, i] - self.norm_stats[c]["mean"]) / self.norm_stats[c][
                    "std"
                ]

        n_nodes = x.shape[0]
        if not self.use_graph or edge_index is None:
            return Data(x=x, y=y, baseline_wind=baseline_wind, num_nodes=n_nodes)

        return Data(
            x=x,
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr,
            baseline_wind=baseline_wind,
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
