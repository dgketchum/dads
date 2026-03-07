"""
Precompute per-day PyG graph files for the wind GNN.

Reads the station-day Parquet + station CSV, builds per-day graphs with
raw (unnormalized) features for ALL stations, and saves each as a .pt file.
Normalization, feature selection, and holdout filtering happen at load time
in PrecomputedWindDataset.

Output structure:
    {out_dir}/
        meta.json       # {all_feature_cols, edge_dim, n_days}
        2018-01-01.pt   # {x, y, edge_index, edge_attr, baseline_wind, fids}
        ...
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None

from models.wind_bias.wind_dataset import TARGET_COLS, _get_feature_cols
from prep.paths import MVP_ROOT
from prep.graph_utils import (
    build_edges_for_day,
    build_knn_map,
    build_static_edge_attrs,
    compute_edge_norm,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute wind GNN graphs.")
    p.add_argument(
        "--table-path",
        default=f"{MVP_ROOT}/station_day_wind_pnw_2018_2024.parquet",
    )
    p.add_argument("--stations-csv", default="artifacts/madis_pnw.csv")
    p.add_argument("--out-dir", default=f"{MVP_ROOT}/wind_graphs_pnw")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--max-radius-km", type=float, default=150.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load table
    print("Loading station-day table...")
    df = pd.read_parquet(args.table_path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["fid"] = df["fid"].astype(str)
    df["day"] = pd.to_datetime(df["day"])

    # Get ALL feature cols (all flags on)
    all_feature_cols = _get_feature_cols(df, use_sx=True, use_flow_terrain=True)
    target_cols = [c for c in TARGET_COLS if c in df.columns]

    # Fill NaN in feature cols
    for c in all_feature_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # Build k-NN map and static edge attrs
    print(f"Building k-NN map (k={args.k}, max_radius={args.max_radius_km} km)...")
    knn_map = build_knn_map(
        args.stations_csv, k=args.k, max_radius_km=args.max_radius_km
    )
    print("Building static edge attributes...")
    static_edges = build_static_edge_attrs(args.stations_csv, knn_map)

    # Edge normalization stats (distance, delta_elevation)
    edge_norm = compute_edge_norm(static_edges)

    # Group by day
    day_groups = {str(day): grp for day, grp in df.groupby("day")}
    days = sorted(day_groups.keys())
    print(f"Days to process: {len(days)}")

    t0 = time.time()
    for i, day_key in enumerate(days):
        day_df = day_groups[day_key]
        fids = day_df["fid"].values
        fid_list = list(fids)

        # Node features (raw, unnormalized)
        x = day_df[all_feature_cols].values.astype("float32")
        y = day_df[target_cols].values.astype("float32")

        # RTMA wind speed
        if "ugrd_rtma" in day_df.columns:
            rtma_wind = np.sqrt(
                day_df["ugrd_rtma"].values ** 2 + day_df["vgrd_rtma"].values ** 2
            ).astype("float32")
        else:
            rtma_wind = np.zeros(len(fids), dtype="float32")

        # Build edges
        ugrd = day_df["ugrd_rtma"].values if "ugrd_rtma" in day_df.columns else None
        vgrd = day_df["vgrd_rtma"].values if "vgrd_rtma" in day_df.columns else None
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
            baseline_wind=torch.from_numpy(rtma_wind),
            num_nodes=len(fids),
        )
        # Store fids as a list attr (not tensor)
        data.fids = fid_list

        # Save with date string as filename
        date_str = pd.Timestamp(day_key).strftime("%Y-%m-%d")
        torch.save(data, os.path.join(args.out_dir, f"{date_str}.pt"))

        if (i + 1) % 100 == 0 or (i + 1) == len(days):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(days) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i + 1}/{len(days)}] {date_str}  "
                f"{rate:.1f} days/s  ETA {eta / 60:.1f} min"
            )

    # Save metadata
    meta = {
        "all_feature_cols": all_feature_cols,
        "target_cols": target_cols,
        "edge_dim": 7,
        "n_days": len(days),
        "k": args.k,
        "max_radius_km": args.max_radius_km,
        "edge_norm": edge_norm,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. {len(days)} graphs saved to {args.out_dir}")


if __name__ == "__main__":
    main()
