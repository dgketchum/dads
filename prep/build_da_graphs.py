"""
Build per-day HeteroData graphs for DA benchmark.

Supports da-graph-v1 (37-feature core-graph-v0 context) and da-graph-v2
(enriched context via --extra-feature-cols).

Creates two node types:
  - query: all stations with HRRR background
  - source: non-holdout stations with observed delta values

Source context mirrors query features. Source payload is observation deltas
+ valid flags (4 features).

Edge families:
  - source -> query: observation influence (excludes colocated self-edges)
  - query -> query: k-NN spatial propagation

Output: HeteroData .pt files + meta.json + split_spec.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import functools
import numpy as np
import pandas as pd
import rasterio
import torch
from torch_geometric.data import HeteroData

from models.rtma_bias.patch_dataset import _date_to_period
from prep.graph_utils import (
    build_edges_for_day,
    build_knn_map,
    build_static_edge_attrs,
    compute_edge_norm,
)

# Force flushed output for nohup/log compatibility
print = functools.partial(print, flush=True)  # type: ignore[assignment]

# Reuse feature definitions from build_graphs.py
TERRAIN_COLS = ["elevation", "slope", "aspect_sin", "aspect_cos", "tpi_4", "tpi_10"]
TEMPORAL_COLS = ["doy_sin", "doy_cos"]
LOCATION_COLS = ["latitude", "longitude"]
RSUN_COL = "rsun"
LANDSAT_COLS = ["ls_b2", "ls_b3", "ls_b4", "ls_b5", "ls_b6", "ls_b7", "ls_b10"]


def _discover_weather_cols(df, prefix, drop=None):
    suffix = f"_{prefix}"
    cols = sorted(
        c for c in df.columns if c.endswith(suffix) and not c.startswith("delta_")
    )
    if drop:
        cols = [c for c in cols if c not in drop]
    return cols


def _load_raster(path):
    with rasterio.open(path) as src:
        data = src.read().astype("float32")
        tf = np.array(src.transform, dtype="float64")
    return data, tf


def _extract_point(raster, transform, x_proj, y_proj, band_idx=None):
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


def _extract_point_bands(raster, transform, x_proj, y_proj, band_start, n_bands):
    a, _b, c, _d, e, f = transform[:6]
    col = int((x_proj - c) / a)
    row = int((y_proj - f) / e)
    _, H, W = raster.shape
    if 0 <= row < H and 0 <= col < W:
        return raster[band_start : band_start + n_bands, row, col].astype("float32")
    return np.zeros(n_bands, dtype="float32")


def _parse_args():
    p = argparse.ArgumentParser(description="Build DA hetero graphs.")
    p.add_argument("--table-path", required=True)
    p.add_argument("--stations-csv", required=True)
    p.add_argument("--holdout-fids-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--terrain-tif", required=True)
    p.add_argument("--rsun-tif", required=True)
    p.add_argument("--landsat-tif", required=True)
    p.add_argument("--model-prefix", default="hrrr")
    p.add_argument("--extra-prefix", nargs="*", default=None)
    p.add_argument("--target-cols", nargs="+", default=["delta_tmax", "delta_tmin"])
    p.add_argument("--required-col", default=None)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--source-k", type=int, default=16)
    p.add_argument("--max-radius-km", type=float, default=150.0)
    p.add_argument(
        "--source-max-radius-km",
        type=float,
        default=None,
        help="Optional source->query radius cap. Defaults to --max-radius-km.",
    )
    p.add_argument("--max-abs-target", type=float, default=None)
    p.add_argument(
        "--extra-feature-cols",
        nargs="*",
        default=None,
        help="Explicit extra column names to include from parquet",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    prefix = args.model_prefix
    target_cols = args.target_cols
    required_col = args.required_col or f"tmp_{prefix}"

    # Load holdout
    with open(args.holdout_fids_json) as f:
        holdout_fids = set(str(x) for x in json.load(f))
    print(f"Holdout fids: {len(holdout_fids)}")

    # Load table
    print("Loading station-day table...")
    df = pd.read_parquet(args.table_path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["fid"] = df["fid"].astype(str)
    df["day"] = pd.to_datetime(df["day"])

    # Discover query feature columns (same as core-graph-v0)
    weather_cols = _discover_weather_cols(df, prefix)
    extra_cols = []
    for ep in args.extra_prefix or []:
        extra_cols.extend(_discover_weather_cols(df, ep))
    print(f"Weather cols ({prefix}): {weather_cols}")
    if extra_cols:
        print(f"Extra cols: {extra_cols}")

    # Keep rows with required background
    before = len(df)
    df = df.dropna(subset=[required_col])
    print(f"Kept {len(df)} of {before} rows with {required_col}")

    # Explicit extra feature columns (must all be present — fail hard per policy)
    explicit_extra_cols: list[str] = []
    if args.extra_feature_cols:
        missing = [c for c in args.extra_feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"--extra-feature-cols requested columns not in table: {missing}"
            )
        explicit_extra_cols = list(args.extra_feature_cols)
        print(f"Explicit extra cols: {explicit_extra_cols}")

    # Fill NaN in weather/extra/explicit
    for c in weather_cols + extra_cols + explicit_extra_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # Query feature list
    query_feature_cols = (
        weather_cols
        + extra_cols
        + explicit_extra_cols
        + TERRAIN_COLS
        + [RSUN_COL]
        + LANDSAT_COLS
        + TEMPORAL_COLS
        + LOCATION_COLS
    )

    # Source context: same core-graph-v0 features at source location
    source_context_feature_cols = list(query_feature_cols)
    # Source payload: observation deltas + valid flags
    source_payload_feature_cols = list(target_cols) + [
        f"valid_{tc}" for tc in target_cols
    ]
    print(f"Query features: {len(query_feature_cols)}")
    print(f"Source context features: {len(source_context_feature_cols)}")
    print(f"Source payload features: {len(source_payload_feature_cols)}")

    # Load station inventory and project
    from pyproj import Transformer

    stations = pd.read_csv(args.stations_csv)
    id_col = "station_id" if "station_id" in stations.columns else "fid"
    stations[id_col] = stations[id_col].astype(str)
    sta_lookup = stations.set_index(id_col)

    for col in ["latitude", "longitude", "elevation"]:
        if col not in df.columns:
            df[col] = df["fid"].map(sta_lookup[col])
    df = df.dropna(subset=["latitude", "longitude"])

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    unique_fids = df["fid"].unique()
    fid_proj = {}
    for fid in unique_fids:
        row = df[df["fid"] == fid].iloc[0]
        x5070, y5070 = transformer.transform(row["longitude"], row["latitude"])
        fid_proj[fid] = (x5070, y5070)

    # Load rasters
    terrain_data, terrain_tf = _load_raster(args.terrain_tif)
    rsun_data, rsun_tf = _load_raster(args.rsun_tif)
    landsat_data, landsat_tf = _load_raster(args.landsat_tif)
    print(
        f"Terrain: {terrain_data.shape}, RSUN: {rsun_data.shape}, Landsat: {landsat_data.shape}"
    )

    # Pre-extract static terrain per station
    fid_terrain = {}
    for fid in unique_fids:
        x5070, y5070 = fid_proj[fid]
        vals = [
            _extract_point(terrain_data, terrain_tf, x5070, y5070, b) for b in range(6)
        ]
        fid_terrain[fid] = np.array(vals, dtype="float32")

    # Build query k-NN (all active stations)
    active_fids = set(unique_fids)
    active_sta = stations[stations[id_col].isin(active_fids)]
    active_csv = os.path.join(args.out_dir, "_active_stations.csv")
    active_sta.to_csv(active_csv, index=False)

    print(
        f"Building query k-NN (k={args.k}, all {len(active_fids)} active stations)..."
    )
    query_knn = build_knn_map(active_csv, k=args.k, max_radius_km=args.max_radius_km)
    query_static_edges = build_static_edge_attrs(active_csv, query_knn)
    query_edge_norm = compute_edge_norm(query_static_edges)

    # Build source k-NN (source -> query): from non-holdout stations to all stations
    # We'll build this per-day since the source population varies
    # For static edges, build from the full non-holdout set
    non_holdout_fids = active_fids - holdout_fids
    non_holdout_sta = stations[stations[id_col].isin(non_holdout_fids)]
    source_csv = os.path.join(args.out_dir, "_source_stations.csv")
    non_holdout_sta.to_csv(source_csv, index=False)

    # Source -> query: for each query, find k nearest source stations
    # Use BallTree directly for bipartite k-NN
    from sklearn.neighbors import BallTree

    query_coords = np.radians(active_sta[["latitude", "longitude"]].values)
    source_coords = np.radians(non_holdout_sta[["latitude", "longitude"]].values)
    query_fids_arr = active_sta[id_col].astype(str).values
    source_fids_arr = non_holdout_sta[id_col].astype(str).values

    source_tree = BallTree(source_coords, metric="haversine")
    source_max_radius_km = (
        args.source_max_radius_km
        if args.source_max_radius_km is not None
        else args.max_radius_km
    )
    max_rad = source_max_radius_km / 6371.0
    source_k = min(args.source_k, len(source_fids_arr))
    print(
        "Building source->query candidate graph "
        f"(source_k={source_k}, source_radius={source_max_radius_km} km, "
        f"non_holdout_sources={len(source_fids_arr)})..."
    )

    # Pre-compute source->query k-NN map
    sq_dists, sq_indices = source_tree.query(query_coords, k=source_k)
    sq_knn_map = {}  # query_fid -> list of (source_fid, distance)
    for qi, qfid in enumerate(query_fids_arr):
        nbrs = []
        for si_idx, dist in zip(sq_indices[qi], sq_dists[qi]):
            sfid = str(source_fids_arr[si_idx])
            if dist <= max_rad and sfid != qfid:  # exclude colocated self-edge
                nbrs.append(sfid)
        sq_knn_map[str(qfid)] = nbrs

    # Build source->query static edge attrs directly (not via build_static_edge_attrs,
    # which assumes both endpoints are in the same CSV). Direction: source -> query.
    # Bearing and delta_elevation are from the source station's perspective toward
    # the query station.
    source_sta_lookup = non_holdout_sta.set_index(id_col)
    query_sta_lookup = active_sta.set_index(id_col)

    sq_static_edges: dict[str, dict[str, dict[str, float]]] = {}
    for qfid, nbr_sfids in sq_knn_map.items():
        sq_static_edges[qfid] = {}
        if qfid not in query_sta_lookup.index:
            continue
        q_row = query_sta_lookup.loc[qfid]
        q_lat, q_lon = float(q_row["latitude"]), float(q_row["longitude"])
        q_elev = float(q_row.get("elevation", 0) or 0)
        for sfid in nbr_sfids:
            if sfid not in source_sta_lookup.index:
                continue
            s_row = source_sta_lookup.loc[sfid]
            s_lat, s_lon = float(s_row["latitude"]), float(s_row["longitude"])
            s_elev = float(s_row.get("elevation", 0) or 0)

            # Haversine distance
            lat1, lat2 = np.radians(s_lat), np.radians(q_lat)
            dlon = np.radians(q_lon - s_lon)
            dlat = lat2 - lat1
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            )
            dist_km = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            # Bearing from source to query
            bearing = np.arctan2(
                np.sin(dlon) * np.cos(lat2),
                np.cos(lat1) * np.sin(lat2)
                - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
            )

            sq_static_edges[qfid][sfid] = {
                "distance_km": dist_km,
                "bearing_sin": float(np.sin(bearing)),
                "bearing_cos": float(np.cos(bearing)),
                "delta_elevation": q_elev - s_elev,  # query minus source
            }

    # Compute normalization stats for source->query edges
    all_dists = [
        e["distance_km"] for qe in sq_static_edges.values() for e in qe.values()
    ]
    all_delev = [
        e["delta_elevation"] for qe in sq_static_edges.values() for e in qe.values()
    ]
    sq_edge_norm = {
        "dist_mean": float(np.mean(all_dists)) if all_dists else 0.0,
        "dist_std": float(max(np.std(all_dists), 1e-8)) if all_dists else 1.0,
        "delev_mean": float(np.mean(all_delev)) if all_delev else 0.0,
        "delev_std": float(max(np.std(all_delev), 1e-8)) if all_delev else 1.0,
    }

    if os.path.exists(active_csv):
        os.remove(active_csv)
    if os.path.exists(source_csv):
        os.remove(source_csv)

    # Group by day
    day_groups = {str(day): grp for day, grp in df.groupby("day")}
    days = sorted(day_groups.keys())
    print(f"Days to process: {len(days)}")

    t0 = time.time()
    for i, day_key in enumerate(days):
        day_df = day_groups[day_key]
        day_ts = pd.Timestamp(day_key)
        fids = day_df["fid"].values
        fid_list = list(fids)
        n_query = len(fid_list)

        # === QUERY NODE FEATURES ===
        wx_arr = day_df[weather_cols].values.astype("float32")
        if extra_cols:
            ex_arr = day_df[extra_cols].values.astype("float32")
        else:
            ex_arr = np.zeros((n_query, 0), dtype="float32")
        if explicit_extra_cols:
            expl_arr = day_df[explicit_extra_cols].values.astype("float32")
        else:
            expl_arr = np.zeros((n_query, 0), dtype="float32")

        terrain_arr = np.stack(
            [fid_terrain.get(f, np.zeros(6, dtype="float32")) for f in fid_list]
        )

        doy_idx = min(day_ts.dayofyear - 1, rsun_data.shape[0] - 1)
        rsun_arr = np.array(
            [
                _extract_point(rsun_data, rsun_tf, *fid_proj[f], band_idx=doy_idx)
                for f in fid_list
            ],
            dtype="float32",
        ).reshape(-1, 1)

        period = _date_to_period(day_ts)
        b_start = period * 7
        landsat_arr = np.stack(
            [
                _extract_point_bands(landsat_data, landsat_tf, *fid_proj[f], b_start, 7)
                for f in fid_list
            ]
        )

        doy = day_ts.dayofyear
        doy_sin = np.sin(2 * np.pi * doy / 365.25)
        doy_cos = np.cos(2 * np.pi * doy / 365.25)
        temporal_arr = np.full((n_query, 2), [doy_sin, doy_cos], dtype="float32")

        loc_arr = np.column_stack(
            [
                day_df["latitude"].values.astype("float32"),
                day_df["longitude"].values.astype("float32"),
            ]
        )

        parts = [wx_arr]
        if ex_arr.shape[1] > 0:
            parts.append(ex_arr)
        if expl_arr.shape[1] > 0:
            parts.append(expl_arr)
        parts.extend([terrain_arr, rsun_arr, landsat_arr, temporal_arr, loc_arr])
        query_x = np.concatenate(parts, axis=1)
        query_x = np.nan_to_num(query_x, nan=0.0, posinf=0.0, neginf=0.0)

        # Query targets
        y_raw = np.column_stack(
            [day_df[tc].values.astype("float32") for tc in target_cols]
        )
        valid_mask = np.isfinite(y_raw)
        if args.max_abs_target is not None:
            valid_mask = valid_mask & (np.abs(y_raw) <= args.max_abs_target)
        y = np.where(valid_mask, y_raw, 0.0).astype("float32")

        # === SOURCE NODES (non-holdout with observations) ===
        source_mask = np.array([f not in holdout_fids for f in fid_list])
        # Source must have at least one valid target
        any_valid = valid_mask.any(axis=1)
        source_mask = source_mask & any_valid

        source_indices = np.where(source_mask)[0]
        source_fids = [fid_list[si] for si in source_indices]
        n_source = len(source_fids)

        # Build source features: full query features at source locations + obs payload
        # The query_x array has all stations in fid_list order; slice source indices
        source_background = query_x[source_indices]  # (n_source, 37)

        # Observation payload: deltas + valid flags
        obs_payload = []
        for tc in target_cols:
            vals = day_df[tc].values.astype("float32")[source_indices]
            obs_payload.append(np.nan_to_num(vals, nan=0.0))
        for ti, tc in enumerate(target_cols):
            vals = valid_mask[source_indices, ti].astype("float32")
            obs_payload.append(vals)

        obs_arr = (
            np.column_stack(obs_payload)
            if obs_payload
            else np.zeros((n_source, 0), dtype="float32")
        )
        # da-graph-v1: keep context and payload as separate tensors
        source_context_x = source_background  # (n_source, 37)
        source_payload_x = obs_arr  # (n_source, 4)

        # === QUERY -> QUERY EDGES ===
        ugrd = day_df["ugrd_hrrr"].values if "ugrd_hrrr" in day_df.columns else None
        vgrd = day_df["vgrd_hrrr"].values if "vgrd_hrrr" in day_df.columns else None
        qq_edge_index, qq_edge_attr = build_edges_for_day(
            fid_list, ugrd, vgrd, query_knn, query_static_edges, query_edge_norm
        )

        # === SOURCE -> QUERY EDGES ===
        source_fid_to_idx = {f: si for si, f in enumerate(source_fids)}

        sq_src_list, sq_dst_list, sq_attr_list = [], [], []
        for qi, qfid in enumerate(fid_list):
            nbr_sfids = sq_knn_map.get(qfid, [])
            for sfid in nbr_sfids:
                if sfid not in source_fid_to_idx:
                    continue  # source not active today
                si = source_fid_to_idx[sfid]
                sq_src_list.append(si)
                sq_dst_list.append(qi)

                # Edge attributes from pre-computed static edges
                static = sq_static_edges.get(qfid, {}).get(sfid)
                if static is not None:
                    attr = np.array(
                        [
                            (static["distance_km"] - sq_edge_norm["dist_mean"])
                            / sq_edge_norm["dist_std"],
                            static["bearing_sin"],
                            static["bearing_cos"],
                            (
                                static["delta_elevation"]
                                - sq_edge_norm.get("delev_mean", 0)
                            )
                            / max(sq_edge_norm.get("delev_std", 1), 1e-8),
                            0.0,  # delta_tpi placeholder
                            0.0,  # upwind_cos (could add dynamic)
                            0.0,  # upwind_sin
                        ],
                        dtype="float32",
                    )
                else:
                    attr = np.zeros(7, dtype="float32")
                sq_attr_list.append(attr)

        if sq_src_list:
            sq_edge_index = torch.tensor([sq_src_list, sq_dst_list], dtype=torch.long)
            sq_edge_attr = torch.from_numpy(np.stack(sq_attr_list))
        else:
            sq_edge_index = torch.zeros(2, 0, dtype=torch.long)
            sq_edge_attr = torch.zeros(0, 7, dtype=torch.float32)

        # === BUILD HETERODATA ===
        data = HeteroData()
        data["query"].x = torch.from_numpy(query_x)
        data["query"].y = torch.from_numpy(y)
        data["query"].valid_mask = torch.from_numpy(valid_mask)
        data["query"].fids = fid_list

        data["source"].context_x = torch.from_numpy(source_context_x)
        data["source"].payload_x = torch.from_numpy(source_payload_x)
        data["source"].fids = source_fids

        data["source", "influences", "query"].edge_index = sq_edge_index
        data["source", "influences", "query"].edge_attr = sq_edge_attr

        data["query", "neighbors", "query"].edge_index = qq_edge_index
        data["query", "neighbors", "query"].edge_attr = qq_edge_attr

        date_str = day_ts.strftime("%Y-%m-%d")
        torch.save(data, os.path.join(args.out_dir, f"{date_str}.pt"))

        if (i + 1) % 50 == 0 or (i + 1) == len(days):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(days) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i + 1}/{len(days)}] {date_str}  {rate:.1f} days/s  ETA {eta / 60:.1f} min"
            )

    # Compute effective holdout coverage
    all_query_fids = set()
    val_query_fids = set()
    for day_key in days:
        day_df = day_groups[day_key]
        day_fids = set(day_df["fid"].values)
        all_query_fids.update(day_fids)
        if pd.Timestamp(day_key).year in {2024}:
            val_query_fids.update(day_fids)

    effective_holdout = holdout_fids & all_query_fids
    val_holdout = holdout_fids & val_query_fids
    print(
        f"Holdout: {len(holdout_fids)} requested, "
        f"{len(effective_holdout)} in graphs, "
        f"{len(val_holdout)} on 2024 val days"
    )

    # Save holdout parity artifacts
    with open(os.path.join(args.out_dir, "requested_holdout_fids.json"), "w") as f:
        json.dump(sorted(holdout_fids), f)
    with open(os.path.join(args.out_dir, "effective_holdout_fids.json"), "w") as f:
        json.dump(sorted(effective_holdout), f)
    with open(os.path.join(args.out_dir, "val_holdout_fids.json"), "w") as f:
        json.dump(sorted(val_holdout), f)

    # Save metadata
    # Family version: da-graph-v1 if standard 37-feature context,
    # da-graph-v2 if extra features were added (new family per policy)
    family = "da-graph-v2" if explicit_extra_cols else "da-graph-v1"
    meta = {
        "family": family,
        "query_feature_cols": query_feature_cols,
        "source_context_feature_cols": source_context_feature_cols,
        "source_payload_feature_cols": source_payload_feature_cols,
        "target_cols": target_cols,
        "query_edge_dim": 7,
        "source_query_edge_dim": 7,
        "source_query_edge_active_channels": [
            "distance_norm",
            "bearing_sin",
            "bearing_cos",
            "delta_elevation_norm",
        ],
        "source_query_edge_placeholder_channels": [
            "delta_tpi (zero)",
            "upwind_cos (zero)",
            "upwind_sin (zero)",
        ],
        "n_days": len(days),
        "n_query_features": len(query_feature_cols),
        "n_source_context_features": len(source_context_feature_cols),
        "n_source_payload_features": len(source_payload_feature_cols),
        "k": args.k,
        "source_k": args.source_k,
        "max_radius_km": args.max_radius_km,
        "source_max_radius_km": source_max_radius_km,
        "holdout_fids_json": args.holdout_fids_json,
        "n_holdout_requested": len(holdout_fids),
        "n_holdout_effective": len(effective_holdout),
        "n_holdout_val_days": len(val_holdout),
        "query_edge_norm": query_edge_norm,
        "source_query_edge_norm": sq_edge_norm,
        "extra_feature_cols": explicit_extra_cols,
        "extra_feature_data_class": {
            c: "static"
            if c in TERRAIN_COLS
            or c.startswith("terrain_")
            or c.startswith("sx_")
            or c.startswith("facet_")
            or c.startswith("effective_terrain")
            else "background"
            for c in explicit_extra_cols
        },
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. {len(days)} DA graphs saved to {args.out_dir}")
    print(
        f"Query: {len(query_feature_cols)}, "
        f"Source context: {len(source_context_feature_cols)}, "
        f"Source payload: {len(source_payload_feature_cols)}"
    )


if __name__ == "__main__":
    main()
