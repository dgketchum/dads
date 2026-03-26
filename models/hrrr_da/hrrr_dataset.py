"""
HRRR wind graph dataset — one PyG Data object per day.

Mirrors wind_dataset.py but swaps RTMA columns for HRRR columns.
Loads station-day parquets + obs, builds PyG Data objects using
prep/graph_utils.py for edge construction.
"""

from __future__ import annotations

import json
from pathlib import Path

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

HRRR_WEATHER_COLS = [
    "ugrd_hrrr",
    "vgrd_hrrr",
    "wind_hrrr",
    "tmp_hrrr",
    "dpt_hrrr",
    "pres_hrrr",
    "tcdc_hrrr",
    "ea_hrrr",
    "dswrf_hrrr",
    "hpbl_hrrr",
    "spfh_hrrr",
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
TARGET_COLS = [
    "delta_tmax",  # 0
    "delta_tmin",  # 1
    "delta_ea",  # 2
    "delta_rsds",  # 3
    "delta_w_par",  # 4  ← wind heads start here
    "delta_w_perp",  # 5
]


def _get_sx_cols(df: pd.DataFrame) -> list[str]:
    return sorted(
        c for c in df.columns if c.startswith("sx_") and ("_2k" in c or "_10k" in c)
    )


def _get_feature_cols(
    df: pd.DataFrame,
    use_sx: bool = True,
    use_flow_terrain: bool = True,
    use_innovations: bool = False,
) -> list[str]:
    """Build ordered list of node feature columns."""
    cols = []
    for c in HRRR_WEATHER_COLS:
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
    if use_innovations:
        for c in TARGET_COLS:
            if c in df.columns:
                cols.append(c)
    for c in TEMPORAL_COLS:
        if c in df.columns:
            cols.append(c)
    for c in LOCATION_COLS:
        if c in df.columns:
            cols.append(c)
    return cols


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


class HRRRGraphDataset(Dataset):
    """One PyG Data per day, using HRRR as the gridded baseline.

    Parameters
    ----------
    table_path : str
        Path to station_day_hrrr Parquet.
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
        use_innovations: bool = False,
        is_val: bool = False,
        norm_stats: dict | None = None,
        train_days: set | None = None,
        exclude_fids: set | None = None,
    ):
        super().__init__()
        self.use_graph = use_graph
        self.use_sx = use_sx
        self.use_flow_terrain = use_flow_terrain
        self.use_innovations = use_innovations
        self._is_val = is_val
        self.k = k
        # When using innovations, keep holdout stations in graph (transductive)
        # but mask their innovation features. Store holdout set for masking.
        self._holdout_fids: set[str] = set()
        if use_innovations and exclude_fids:
            self._holdout_fids = set(exclude_fids)

        df = pd.read_parquet(table_path)
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        df["fid"] = df["fid"].astype(str)
        df["day"] = pd.to_datetime(df["day"])

        if train_days is not None:
            df = df[df["day"].isin(train_days)]
        # Inductive holdout: remove stations entirely (unless using innovations)
        if exclude_fids is not None and not use_innovations:
            df = df[~df["fid"].isin(exclude_fids)]

        # Warn if fewer than 4 of 6 canonical targets are present (likely old table)
        found_targets = [c for c in TARGET_COLS if c in df.columns]
        if len(found_targets) < 4:
            import warnings

            warnings.warn(
                f"Only {len(found_targets)}/{len(TARGET_COLS)} TARGET_COLS found in table "
                f"({found_targets}). Run build_hrrr_station_day_table with the new "
                "multivariable builder to include all targets.",
                stacklevel=2,
            )

        self.feature_cols = _get_feature_cols(
            df,
            use_sx=use_sx,
            use_flow_terrain=use_flow_terrain,
            use_innovations=use_innovations,
        )
        self.node_dim = len(self.feature_cols)

        # Norm stats
        if norm_stats is None:
            self.norm_stats = compute_norm_stats(df, self.feature_cols)
        else:
            self.norm_stats = norm_stats

        # Station metadata for graph construction
        self.knn_map = build_knn_map(stations_csv, k=k)
        self.static_edges = build_static_edge_attrs(stations_csv, self.knn_map)
        self.edge_norm = compute_edge_norm(self.static_edges)
        self.edge_dim = 7

        # Group by day
        self._days = sorted(df["day"].unique())
        self._day_groups = {d: g for d, g in df.groupby("day")}

    def __len__(self) -> int:
        return len(self._days)

    def __getitem__(self, idx: int) -> Data:
        day = self._days[idx]
        day_df = self._day_groups[day]

        fids = day_df["fid"].tolist()
        features = day_df[self.feature_cols].values
        features = apply_norm(features, self.feature_cols, self.norm_stats)

        # Mask holdout stations' innovation features to 0 (pre-normalization value)
        if self.use_innovations and self._holdout_fids:
            inn_indices = [
                i for i, c in enumerate(self.feature_cols) if c in TARGET_COLS
            ]
            holdout_mask = np.array([f in self._holdout_fids for f in fids], dtype=bool)
            for ci in inn_indices:
                # Set to the normalized value of 0 = (0 - mean) / std
                stats = self.norm_stats.get(self.feature_cols[ci])
                if stats:
                    features[holdout_mask, ci] = -stats["mean"] / stats["std"]
                else:
                    features[holdout_mask, ci] = 0.0

        x = torch.from_numpy(features)

        # Build y and valid_mask across all canonical targets.
        # WARNING: NaN targets are filled with 0.0. Always gate on valid_mask before loss.
        n_nodes = len(day_df)
        y_raw = np.full((n_nodes, len(TARGET_COLS)), np.nan, dtype="float32")
        for col_idx, col in enumerate(TARGET_COLS):
            if col in day_df.columns:
                y_raw[:, col_idx] = day_df[col].values.astype("float32")
        valid_mask_np = ~np.isnan(y_raw)
        y_raw[np.isnan(y_raw)] = 0.0
        y = torch.from_numpy(y_raw)  # (N, 6) float32
        valid_mask = torch.from_numpy(valid_mask_np)  # (N, 6) bool

        # Edges
        if self.use_graph:
            ugrd = day_df["ugrd_hrrr"].values if "ugrd_hrrr" in day_df.columns else None
            vgrd = day_df["vgrd_hrrr"].values if "vgrd_hrrr" in day_df.columns else None
            edge_index, edge_attr = build_edges_for_day(
                fids, ugrd, vgrd, self.knn_map, self.static_edges, self.edge_norm
            )
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, self.edge_dim)

        # Extra attributes for wind metrics
        hrrr_wind = (
            day_df["wind_hrrr"].values if "wind_hrrr" in day_df.columns else None
        )

        # Transductive loss_mask for innovation mode:
        #   train: loss on non-holdout only (holdout features masked, no supervision)
        #   val:   loss on holdout only (test generalization to masked stations)
        if self.use_innovations and self._holdout_fids:
            is_holdout = [f in self._holdout_fids for f in fids]
            if self._is_val:
                loss_mask = torch.tensor(is_holdout, dtype=torch.bool)
            else:
                loss_mask = torch.tensor([not h for h in is_holdout], dtype=torch.bool)
        else:
            loss_mask = torch.ones(n_nodes, dtype=torch.bool)

        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr,
            fids=fids,
            valid_mask=valid_mask,
            loss_mask=loss_mask,
        )
        if hrrr_wind is not None:
            data.baseline_wind = torch.from_numpy(hrrr_wind.astype("float32"))
        if "ugrd_hrrr" in day_df.columns:
            data.ugrd_baseline = torch.from_numpy(
                day_df["ugrd_hrrr"].values.astype("float32")
            )
            data.vgrd_baseline = torch.from_numpy(
                day_df["vgrd_hrrr"].values.astype("float32")
            )
        if "u_obs" in day_df.columns:
            data.u_obs = torch.from_numpy(day_df["u_obs"].values.astype("float32"))
            data.v_obs = torch.from_numpy(day_df["v_obs"].values.astype("float32"))

        return data

    def save_norm_stats(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.norm_stats, f, indent=2)


class PrecomputedHRRRDataset(Dataset):
    """Load pre-built .pt graph files from a directory.

    valid_mask is embedded at .pt build time via HRRRGraphDataset.__getitem__.
    Pre-built files without valid_mask (built before the multivariable rewrite)
    are stale and must be rebuilt to use task="multitask".
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
        graph_dir = Path(graph_dir)
        files = sorted(graph_dir.glob("*.pt"))
        if train_days is not None:
            train_day_strs = {pd.Timestamp(d).strftime("%Y-%m-%d") for d in train_days}
            files = [f for f in files if f.stem in train_day_strs]

        self._graphs = [torch.load(f, weights_only=False) for f in files]
        if not self._graphs:
            raise ValueError(f"No .pt files found in {graph_dir}")

        sample = self._graphs[0]
        self.node_dim = sample.x.shape[1]
        self.edge_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0
        self.norm_stats = norm_stats or {}
        self.loss_fids: set[str] | None = exclude_fids

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx: int) -> Data:
        return self._graphs[idx]

    def save_norm_stats(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.norm_stats, f, indent=2)
