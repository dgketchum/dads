"""
Precompute per-day PyG graph files for scalar bias-correction GNNs.

Reads the station-day Parquet + station CSV + static TIFs, builds per-day
graphs with raw (unnormalized) node features for ALL stations, and saves
each as a .pt file. Normalization, feature selection, and holdout filtering
happen at load time in PrecomputedGraphDataset.

Weather columns are discovered dynamically from --model-prefix (e.g. "urma"
finds all *_urma columns, "rtma" finds all *_rtma columns). Columns listed
in --drop-features are excluded.

Target: configurable via --target-col (default: delta_tmax).
When --log-target is set, computes log(y_obs) - log({prefix}_col) in-place.

Edge construction: reuses build_knn_map, build_static_edge_attrs,
build_edges_for_day from models/wind_bias/wind_dataset.py.

Output structure:
    {out_dir}/
        meta.json       # {all_feature_cols, target_cols, edge_dim, n_days}
        2024-01-01.pt   # {x, y, edge_index, edge_attr, fids}
        ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import rasterio
import torch
from pyproj import Transformer

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None

from models.rtma_bias.patch_dataset import _date_to_period
from models.wind_bias.wind_dataset import (
    build_edges_for_day,
    build_knn_map,
    build_static_edge_attrs,
)

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

TERRAIN_COLS = [
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "tpi_4",
    "tpi_10",
]

TEMPORAL_COLS = ["doy_sin", "doy_cos"]
LOCATION_COLS = ["latitude", "longitude"]
RSUN_COL = "rsun"
LANDSAT_COLS = ["ls_b2", "ls_b3", "ls_b4", "ls_b5", "ls_b6", "ls_b7", "ls_b10"]


def _discover_weather_cols(
    df: pd.DataFrame, prefix: str, drop: list[str] | None = None
) -> list[str]:
    """Find all _{prefix} columns in df, excluding any in *drop*."""
    suffix = f"_{prefix}"
    cols = sorted(c for c in df.columns if c.endswith(suffix))
    if drop:
        cols = [c for c in cols if c not in drop]
    return cols


# ---------------------------------------------------------------------------
# Static raster helpers
# ---------------------------------------------------------------------------


def _load_terrain(tif_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load 6-band terrain TIF. Returns (data (6, H, W), transform (6,))."""
    with rasterio.open(tif_path) as src:
        data = src.read().astype("float32")
        tf = np.array(src.transform, dtype="float64")
    print(f"Terrain loaded: {data.shape} from {tif_path}", flush=True)
    return data, tf


def _load_rsun(tif_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load 365-band rsun TIF. Returns (data (365, H, W), transform)."""
    with rasterio.open(tif_path) as src:
        data = src.read().astype("float32")
        tf = np.array(src.transform, dtype="float64")
    print(f"RSUN loaded: {data.shape} from {tif_path}", flush=True)
    return data, tf


def _load_landsat(tif_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load 35-band Landsat TIF. Returns (data (35, H, W), transform)."""
    with rasterio.open(tif_path) as src:
        data = src.read().astype("float32")
        tf = np.array(src.transform, dtype="float64")
    print(f"Landsat loaded: {data.shape} from {tif_path}", flush=True)
    return data, tf


def _extract_point(
    raster: np.ndarray,
    transform: np.ndarray,
    x_proj: float,
    y_proj: float,
    band_idx: int | None = None,
) -> float:
    """Extract a single pixel value from a raster at projected coordinates.

    If band_idx is None and raster is 2D, extract directly.
    If band_idx is given, index the first axis.
    """
    a, _b, c, _d, e, f = transform[:6]
    col = int((x_proj - c) / a)
    row = int((y_proj - f) / e)

    if raster.ndim == 2:
        H, W = raster.shape
    else:
        _, H, W = raster.shape

    if 0 <= row < H and 0 <= col < W:
        if band_idx is not None:
            return float(raster[band_idx, row, col])
        elif raster.ndim == 2:
            return float(raster[row, col])
        else:
            return float(raster[0, row, col])
    return 0.0


def _extract_point_bands(
    raster: np.ndarray,
    transform: np.ndarray,
    x_proj: float,
    y_proj: float,
    band_start: int,
    n_bands: int,
) -> np.ndarray:
    """Extract multiple bands at a single point. Returns (n_bands,) float32."""
    a, _b, c, _d, e, f = transform[:6]
    col = int((x_proj - c) / a)
    row = int((y_proj - f) / e)

    _, H, W = raster.shape
    if 0 <= row < H and 0 <= col < W:
        return raster[band_start : band_start + n_bands, row, col].astype("float32")
    return np.zeros(n_bands, dtype="float32")


def _build_covariate_vectors(
    unique_fids: np.ndarray,
    fid_proj: dict[str, tuple[float, float]],
    fid_terrain: dict[str, np.ndarray],
    rsun_data: np.ndarray,
    rsun_tf: np.ndarray,
    landsat_data: np.ndarray,
    landsat_tf: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build min-max normalized 14-dim covariate vectors per station.

    Dimensions: terrain(6) + rsun_annual_mean(1) + landsat_annual_mean(7) = 14.
    """
    raw = {}
    for fid in unique_fids:
        x5070, y5070 = fid_proj[fid]
        terrain = fid_terrain.get(fid, np.zeros(6, dtype="float32"))

        # RSUN annual mean (average over all 365 DOY bands)
        a, _b, c, _d, e, f = rsun_tf[:6]
        col = int((x5070 - c) / a)
        row = int((y5070 - f) / e)
        _, rH, rW = rsun_data.shape
        if 0 <= row < rH and 0 <= col < rW:
            rsun_mean = float(rsun_data[:, row, col].mean())
        else:
            rsun_mean = 0.0

        # Landsat annual mean (5 periods x 7 bands -> mean over periods)
        a, _b, c, _d, e, f = landsat_tf[:6]
        col = int((x5070 - c) / a)
        row = int((y5070 - f) / e)
        _, lH, lW = landsat_data.shape
        if 0 <= row < lH and 0 <= col < lW:
            ls_all = landsat_data[:, row, col].reshape(5, 7)
            ls_mean = ls_all.mean(axis=0).astype("float32")
        else:
            ls_mean = np.zeros(7, dtype="float32")

        raw[fid] = np.concatenate([terrain, [rsun_mean], ls_mean])

    # Min-max normalize each of 14 dims to [0, 1]
    mat = np.stack([raw[fid] for fid in unique_fids])
    mins = mat.min(axis=0)
    maxs = mat.max(axis=0)
    rng = maxs - mins
    rng[rng < 1e-8] = 1.0  # avoid division by zero
    mat_norm = (mat - mins) / rng

    result = {}
    for i, fid in enumerate(unique_fids):
        result[fid] = mat_norm[i].astype("float32")
    return result


def build_covariate_knn_map(
    stations_csv: str,
    fid_covariates: dict[str, np.ndarray],
    k: int = 16,
    max_radius_km: float = 150.0,
    similarity_fraction: float = 0.7,
) -> dict[str, list[str]]:
    """Build k-NN map using covariate similarity/dissimilarity within geo radius.

    For each station, finds all candidates within max_radius_km, then picks
    ~70% by covariate similarity (lowest Euclidean distance) and ~30% by
    covariate dissimilarity (highest distance).
    """
    from sklearn.neighbors import BallTree

    stations = pd.read_csv(stations_csv)
    id_col = "station_id" if "station_id" in stations.columns else "fid"
    fids = stations[id_col].astype(str).values
    coords = np.radians(stations[["latitude", "longitude"]].values)

    tree = BallTree(coords, metric="haversine")
    max_rad = max_radius_km / 6371.0

    # query_radius returns variable-length lists per station
    all_indices, all_dists = tree.query_radius(
        coords, r=max_rad, return_distance=True, sort_results=True
    )

    k_sim = round(k * similarity_fraction)
    k_dissim = k - k_sim

    knn_map: dict[str, list[str]] = {}
    for i, fid in enumerate(fids):
        if fid not in fid_covariates:
            knn_map[str(fid)] = []
            continue

        cov_i = fid_covariates[fid]
        candidates = []
        for j_idx, j in enumerate(all_indices[i]):
            if j == i:
                continue
            nbr_fid = str(fids[j])
            if nbr_fid not in fid_covariates:
                continue
            cov_dist = float(np.linalg.norm(cov_i - fid_covariates[nbr_fid]))
            candidates.append((nbr_fid, cov_dist))

        if len(candidates) <= k:
            # Fewer candidates than k: take all
            knn_map[str(fid)] = [c[0] for c in candidates]
        else:
            # Sort by covariate distance
            candidates.sort(key=lambda c: c[1])
            similar = [c[0] for c in candidates[:k_sim]]
            dissimilar = [c[0] for c in candidates[-k_dissim:]]
            knn_map[str(fid)] = similar + dissimilar

    return knn_map


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute GNN graphs.")
    p.add_argument(
        "--table-path",
        default="/nas/dads/mvp/station_day_e1_tmax_dailyall.parquet",
    )
    p.add_argument(
        "--stations-csv",
        default="/nas/dads/met/stations/madis_02JULY2025_mgrs.csv",
    )
    p.add_argument("--out-dir", default="/nas/dads/mvp/tmax_graphs_pnw_2024")
    p.add_argument("--terrain-tif", default="/nas/dads/mvp/terrain_pnw_1km.tif")
    p.add_argument("--rsun-tif", default="/nas/dads/mvp/rsun_pnw_1km.tif")
    p.add_argument("--landsat-tif", default="/nas/dads/mvp/landsat_pnw_1km.tif")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--max-radius-km", type=float, default=150.0)
    p.add_argument(
        "--neighbor-mode",
        choices=["geo", "covariate"],
        default="geo",
        help="Neighbor selection: pure geographic or covariate-aware",
    )
    p.add_argument(
        "--similarity-fraction",
        type=float,
        default=0.7,
        help="Fraction of k neighbors chosen by covariate similarity (covariate mode)",
    )
    # --- target / variable selection ---
    p.add_argument(
        "--target-col",
        default="delta_tmax",
        help="Name of the target column (default: delta_tmax)",
    )
    p.add_argument(
        "--model-prefix",
        default="urma",
        help="Column suffix for weather vars, e.g. 'urma' or 'rtma'",
    )
    p.add_argument(
        "--drop-features",
        nargs="*",
        default=None,
        help="Weather columns to exclude from node features (e.g. ea_rtma)",
    )
    p.add_argument(
        "--log-target",
        action="store_true",
        help="Compute log-space residual: log(y_obs) - log({base}_col)",
    )
    p.add_argument(
        "--required-col",
        default=None,
        help="Column that must be non-null (default: tmp_{model-prefix})",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    target_col = args.target_col
    prefix = args.model_prefix
    required_col = args.required_col or f"tmp_{prefix}"

    # ------------------------------------------------------------------
    # Load station-day table
    # ------------------------------------------------------------------
    print("Loading station-day table...")
    df = pd.read_parquet(args.table_path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["fid"] = df["fid"].astype(str)
    df["day"] = pd.to_datetime(df["day"])

    # --- Discover weather columns from prefix ---
    weather_cols = _discover_weather_cols(df, prefix, drop=args.drop_features)
    print(f"Weather cols ({prefix}): {weather_cols}")

    # --- Optional log-space target ---
    if args.log_target:
        base_col = f"ea_{prefix}"
        print(f"Computing log-space residual: log(y_obs) - log({base_col})")
        valid = (df["y_obs"] > 1e-4) & (df[base_col] > 1e-4)
        df = df[valid].copy()
        df[target_col] = np.log(df["y_obs"]) - np.log(df[base_col])
        print(f"  {len(df)} rows after filtering y_obs, {base_col} > 1e-4 kPa")

    # Drop rows without required weather data or target
    before = len(df)
    df = df.dropna(subset=[required_col, target_col])
    print(f"Kept {len(df)} of {before} rows with {required_col} + {target_col}")

    # ------------------------------------------------------------------
    # Load station inventory for lat/lon/elevation
    # ------------------------------------------------------------------
    print("Loading station inventory...")
    stations = pd.read_csv(args.stations_csv)
    id_col = "station_id" if "station_id" in stations.columns else "fid"
    stations[id_col] = stations[id_col].astype(str)
    sta_lookup = stations.set_index(id_col)

    # Merge lat/lon/elevation into df
    for col in ["latitude", "longitude", "elevation"]:
        if col not in df.columns:
            df[col] = df["fid"].map(sta_lookup[col])

    # Drop stations not in inventory
    df = df.dropna(subset=["latitude", "longitude"])

    # ------------------------------------------------------------------
    # Project station coords to EPSG:5070 for raster extraction
    # ------------------------------------------------------------------
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    unique_fids = df["fid"].unique()
    fid_proj = {}
    for fid in unique_fids:
        row = df[df["fid"] == fid].iloc[0]
        x5070, y5070 = transformer.transform(row["longitude"], row["latitude"])
        fid_proj[fid] = (x5070, y5070)

    # ------------------------------------------------------------------
    # Load static rasters
    # ------------------------------------------------------------------
    terrain_data, terrain_tf = _load_terrain(args.terrain_tif)
    rsun_data, rsun_tf = _load_rsun(args.rsun_tif)
    landsat_data, landsat_tf = _load_landsat(args.landsat_tif)

    # Precompute per-station static features (terrain + location)
    print("Extracting per-station static features...")
    fid_terrain = {}  # fid -> (6,) array
    for fid in unique_fids:
        x5070, y5070 = fid_proj[fid]
        vals = []
        for b in range(6):
            vals.append(_extract_point(terrain_data, terrain_tf, x5070, y5070, b))
        fid_terrain[fid] = np.array(vals, dtype="float32")

    # ------------------------------------------------------------------
    # Build k-NN map and static edge attrs (from station inventory)
    # ------------------------------------------------------------------
    # First, create a filtered stations CSV with only the active fids
    active_sta = stations[stations[id_col].isin(set(unique_fids))].copy()
    active_csv = os.path.join(args.out_dir, "_active_stations.csv")
    active_sta.to_csv(active_csv, index=False)

    if args.neighbor_mode == "covariate":
        print("Building covariate vectors (14-dim)...")
        fid_covariates = _build_covariate_vectors(
            unique_fids,
            fid_proj,
            fid_terrain,
            rsun_data,
            rsun_tf,
            landsat_data,
            landsat_tf,
        )
        print(
            f"Building covariate k-NN map (k={args.k}, "
            f"max_radius={args.max_radius_km} km, "
            f"sim_frac={args.similarity_fraction})..."
        )
        knn_map = build_covariate_knn_map(
            active_csv,
            fid_covariates,
            k=args.k,
            max_radius_km=args.max_radius_km,
            similarity_fraction=args.similarity_fraction,
        )
    else:
        print(f"Building k-NN map (k={args.k}, max_radius={args.max_radius_km} km)...")
        knn_map = build_knn_map(active_csv, k=args.k, max_radius_km=args.max_radius_km)
    print("Building static edge attributes...")
    static_edges = build_static_edge_attrs(active_csv, knn_map)

    # Edge normalization stats
    all_dists = [
        ea["distance_km"]
        for fid_attrs in static_edges.values()
        for ea in fid_attrs.values()
    ]
    dist_mean = float(np.mean(all_dists)) if all_dists else 1.0
    dist_std = float(max(np.std(all_dists), 1e-6))
    all_delev = [
        ea["delta_elevation"]
        for fid_attrs in static_edges.values()
        for ea in fid_attrs.values()
    ]
    delev_mean = float(np.mean(all_delev)) if all_delev else 0.0
    delev_std = float(max(np.std(all_delev), 1e-6))
    edge_norm = {
        "dist_mean": dist_mean,
        "dist_std": dist_std,
        "delev_mean": delev_mean,
        "delev_std": delev_std,
    }

    # ------------------------------------------------------------------
    # Fill NaN in weather columns
    # ------------------------------------------------------------------
    for c in weather_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # ------------------------------------------------------------------
    # Build all feature column names
    # ------------------------------------------------------------------
    all_feature_cols = (
        weather_cols
        + TERRAIN_COLS
        + [RSUN_COL]
        + LANDSAT_COLS
        + TEMPORAL_COLS
        + LOCATION_COLS
    )

    # ------------------------------------------------------------------
    # Group by day and build graphs
    # ------------------------------------------------------------------
    day_groups = {str(day): grp for day, grp in df.groupby("day")}
    days = sorted(day_groups.keys())
    print(f"Days to process: {len(days)}")

    t0 = time.time()
    for i, day_key in enumerate(days):
        day_df = day_groups[day_key]
        day_ts = pd.Timestamp(day_key)
        fids = day_df["fid"].values
        fid_list = list(fids)
        n_stations = len(fid_list)

        # --- Build node feature matrix ---
        # Weather columns (dynamic, from prefix)
        wx_cols_present = [c for c in weather_cols if c in day_df.columns]
        wx_arr = day_df[wx_cols_present].values.astype("float32")
        # Pad missing columns with zeros
        if len(wx_cols_present) < len(weather_cols):
            full_wx = np.zeros((n_stations, len(weather_cols)), dtype="float32")
            for j, c in enumerate(weather_cols):
                if c in wx_cols_present:
                    full_wx[:, j] = wx_arr[:, wx_cols_present.index(c)]
            wx_arr = full_wx

        # Terrain (6 per station, static)
        terrain_arr = np.stack(
            [fid_terrain.get(f, np.zeros(6, dtype="float32")) for f in fid_list]
        )

        # RSUN (1 per station, DOY-indexed)
        doy_idx = min(day_ts.dayofyear - 1, rsun_data.shape[0] - 1)
        rsun_arr = np.array(
            [
                _extract_point(rsun_data, rsun_tf, *fid_proj[f], band_idx=doy_idx)
                for f in fid_list
            ],
            dtype="float32",
        ).reshape(-1, 1)

        # Landsat (7 per station, period-indexed)
        period = _date_to_period(day_ts)
        b_start = period * 7
        landsat_arr = np.stack(
            [
                _extract_point_bands(landsat_data, landsat_tf, *fid_proj[f], b_start, 7)
                for f in fid_list
            ]
        )

        # Temporal: doy_sin, doy_cos
        doy = day_ts.dayofyear
        doy_sin = np.sin(2 * np.pi * doy / 365.25)
        doy_cos = np.cos(2 * np.pi * doy / 365.25)
        temporal_arr = np.full((n_stations, 2), [doy_sin, doy_cos], dtype="float32")

        # Location: lat, lon
        loc_arr = np.column_stack(
            [
                day_df["latitude"].values.astype("float32"),
                day_df["longitude"].values.astype("float32"),
            ]
        )

        # Concatenate all features
        x = np.concatenate(
            [wx_arr, terrain_arr, rsun_arr, landsat_arr, temporal_arr, loc_arr],
            axis=1,
        )
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Target
        y = day_df[target_col].values.astype("float32")

        # --- Build edges ---
        ugrd_col = f"ugrd_{prefix}"
        vgrd_col = f"vgrd_{prefix}"
        ugrd = day_df[ugrd_col].values if ugrd_col in day_df.columns else None
        vgrd = day_df[vgrd_col].values if vgrd_col in day_df.columns else None
        edge_index, edge_attr = build_edges_for_day(
            fids=fid_list,
            ugrd=ugrd,
            vgrd=vgrd,
            knn_map=knn_map,
            static_edges=static_edges,
            edge_norm=edge_norm,
        )

        data = Data(
            x=torch.from_numpy(x),
            y=torch.from_numpy(y),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=n_stations,
        )
        data.fids = fid_list

        date_str = day_ts.strftime("%Y-%m-%d")
        torch.save(data, os.path.join(args.out_dir, f"{date_str}.pt"))

        if (i + 1) % 50 == 0 or (i + 1) == len(days):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(days) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i + 1}/{len(days)}] {date_str}  "
                f"{rate:.1f} days/s  ETA {eta / 60:.1f} min"
            )

    # Clean up temp file
    if os.path.exists(active_csv):
        os.remove(active_csv)

    # Save metadata
    meta = {
        "all_feature_cols": all_feature_cols,
        "target_cols": [target_col],
        "edge_dim": 7,
        "n_days": len(days),
        "n_features": len(all_feature_cols),
        "k": args.k,
        "max_radius_km": args.max_radius_km,
        "neighbor_mode": args.neighbor_mode,
        "similarity_fraction": args.similarity_fraction,
        "edge_norm": edge_norm,
        "model_prefix": prefix,
        "log_target": args.log_target,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. {len(days)} graphs saved to {args.out_dir}")
    print(f"Features per node: {len(all_feature_cols)}")


if __name__ == "__main__":
    main()
