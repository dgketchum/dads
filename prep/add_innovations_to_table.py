"""
Add neighbor innovation features to a station-day parquet table.

For each station on each day, computes summary statistics of its k-nearest
non-holdout neighbors' delta_tmax values. Writes new columns to the table.

Columns added:
  inn_mean_hrrr   -- mean of neighbors' delta_tmax
  inn_std_hrrr    -- std of neighbors' delta_tmax
  inn_count_hrrr  -- count of neighbors with valid delta_tmax

The _hrrr suffix allows auto-discovery by build_graphs.py --model-prefix hrrr
and by HRRRGraphDataset's feature column discovery.

Usage:
    uv run python -m prep.add_innovations_to_table \
        --table /nas/dads/mvp/station_day_hrrr_daily_pnw.parquet \
        --stations-csv artifacts/madis_pnw.csv \
        --holdout-fids artifacts/canonical_holdout_fids.json \
        --out /nas/dads/mvp/station_day_hrrr_daily_inn_pnw.parquet
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--table", required=True)
    p.add_argument("--stations-csv", required=True)
    p.add_argument("--holdout-fids", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--max-radius-km", type=float, default=150.0)
    p.add_argument("--target-col", default="delta_tmax")
    a = p.parse_args()

    with open(a.holdout_fids) as f:
        holdout = set(str(x) for x in json.load(f))
    print(f"Holdout fids: {len(holdout)}")

    df = pd.read_parquet(a.table)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["fid"] = df["fid"].astype(str)
    df["day"] = pd.to_datetime(df["day"])
    print(f"Table: {len(df)} rows, {df['fid'].nunique()} fids")

    # Build k-NN map from station inventory
    stations = pd.read_csv(a.stations_csv)
    id_col = "station_id" if "station_id" in stations.columns else "fid"
    stations[id_col] = stations[id_col].astype(str)
    active_fids = set(df["fid"].unique())
    stations = stations[stations[id_col].isin(active_fids)].copy()

    coords = np.radians(stations[["latitude", "longitude"]].values)
    fids_arr = stations[id_col].values
    tree = BallTree(coords, metric="haversine")
    max_rad = a.max_radius_km / 6371.0

    indices, _ = tree.query_radius(
        coords, r=max_rad, return_distance=True, sort_results=True
    )

    # Build k-NN map (exclude self)
    knn_map: dict[str, list[str]] = {}
    for i, fid in enumerate(fids_arr):
        nbrs = [str(fids_arr[j]) for j in indices[i] if j != i][: a.k]
        knn_map[str(fid)] = nbrs
    print(f"k-NN map: {len(knn_map)} stations, k={a.k}")

    # Compute innovations per day
    inn_mean = np.full(len(df), np.nan, dtype="float32")
    inn_std = np.full(len(df), np.nan, dtype="float32")
    inn_count = np.zeros(len(df), dtype="float32")

    day_groups = df.groupby("day")
    n_days = len(day_groups)
    done = 0

    for day, grp in day_groups:
        idx = grp.index.values
        fids = grp["fid"].values
        targets = grp[a.target_col].values.astype("float32")
        fid_to_row = {f: i for i, f in enumerate(fids)}

        for ri, (row_idx, fid) in enumerate(zip(idx, fids)):
            nbr_fids = knn_map.get(fid, [])
            # Exclude holdout fids from innovation computation
            nbr_rows = [
                fid_to_row[f] for f in nbr_fids if f in fid_to_row and f not in holdout
            ]
            if nbr_rows:
                vals = targets[nbr_rows]
                valid = vals[np.isfinite(vals)]
                if len(valid) > 0:
                    inn_mean[row_idx] = np.mean(valid)
                    inn_std[row_idx] = np.std(valid) if len(valid) > 1 else 0.0
                    inn_count[row_idx] = float(len(valid))

        done += 1
        if done % 200 == 0:
            print(f"  {done}/{n_days} days", flush=True)

    df["inn_mean_hrrr"] = inn_mean
    df["inn_std_hrrr"] = inn_std
    df["inn_count_hrrr"] = inn_count

    n_valid = np.isfinite(inn_mean).sum()
    print(f"Innovations computed: {n_valid}/{len(df)} rows ({n_valid / len(df):.1%})")

    df.to_parquet(a.out, index=False)
    print(f"Wrote {a.out}: {len(df)} rows, {len(df.columns)} cols")


if __name__ == "__main__":
    main()
