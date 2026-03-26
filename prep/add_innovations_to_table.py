"""
Add per-variable neighbor innovation features to a station-day parquet table.

For each target variable, for each station on each day, computes summary
statistics of its k-nearest non-holdout neighbors' observed values.
Each variable is handled independently — missingness in one variable
does not affect innovations for another.

Columns added (18 total, 6 variables x 3 stats):
  inn_{var}_mean_hrrr   -- mean of neighbors' delta values
  inn_{var}_std_hrrr    -- std of neighbors' delta values
  inn_{var}_count_hrrr  -- count of neighbors with valid data

The _hrrr suffix enables auto-discovery by HRRRGraphDataset.

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

_TARGET_VARS = [
    ("delta_tmax", "tmax"),
    ("delta_tmin", "tmin"),
    ("delta_ea", "ea"),
    ("delta_rsds", "rsds"),
    ("delta_w_par", "wpar"),
    ("delta_w_perp", "wperp"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--table", required=True)
    p.add_argument("--stations-csv", required=True)
    p.add_argument("--holdout-fids", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--max-radius-km", type=float, default=150.0)
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

    # Determine which target columns exist
    active_vars = [(col, short) for col, short in _TARGET_VARS if col in df.columns]
    print(f"Innovation variables: {[s for _, s in active_vars]}")

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

    knn_map: dict[str, list[str]] = {}
    for i, fid in enumerate(fids_arr):
        nbrs = [str(fids_arr[j]) for j in indices[i] if j != i][: a.k]
        knn_map[str(fid)] = nbrs
    print(f"k-NN map: {len(knn_map)} stations, k={a.k}")

    # Pre-allocate innovation arrays for each variable
    n_rows = len(df)
    inn_arrays: dict[str, dict[str, np.ndarray]] = {}
    for _, short in active_vars:
        inn_arrays[short] = {
            "mean": np.full(n_rows, np.nan, dtype="float32"),
            "std": np.full(n_rows, np.nan, dtype="float32"),
            "count": np.zeros(n_rows, dtype="float32"),
        }

    # Compute innovations per day, per variable
    day_groups = df.groupby("day")
    n_days = len(day_groups)
    done = 0

    for day, grp in day_groups:
        idx = grp.index.values
        fids = grp["fid"].values
        fid_to_row = {f: i for i, f in enumerate(fids)}

        # Pre-extract target arrays for each variable
        var_vals: dict[str, np.ndarray] = {}
        for col, short in active_vars:
            var_vals[short] = grp[col].values.astype("float32")

        for ri, (row_idx, fid) in enumerate(zip(idx, fids)):
            nbr_fids = knn_map.get(fid, [])
            nbr_rows = [
                fid_to_row[f] for f in nbr_fids if f in fid_to_row and f not in holdout
            ]
            if not nbr_rows:
                continue

            for _, short in active_vars:
                vals = var_vals[short][nbr_rows]
                valid = vals[np.isfinite(vals)]
                if len(valid) > 0:
                    inn_arrays[short]["mean"][row_idx] = np.mean(valid)
                    inn_arrays[short]["std"][row_idx] = (
                        np.std(valid) if len(valid) > 1 else 0.0
                    )
                    inn_arrays[short]["count"][row_idx] = float(len(valid))

        done += 1
        if done % 200 == 0:
            print(f"  {done}/{n_days} days", flush=True)

    # Add columns to dataframe
    for _, short in active_vars:
        df[f"inn_{short}_mean_hrrr"] = inn_arrays[short]["mean"]
        df[f"inn_{short}_std_hrrr"] = inn_arrays[short]["std"]
        df[f"inn_{short}_count_hrrr"] = inn_arrays[short]["count"]

    inn_cols = [c for c in df.columns if c.startswith("inn_")]
    print(f"Added {len(inn_cols)} innovation columns: {inn_cols}")

    # Report per-variable coverage
    for _, short in active_vars:
        n_valid = np.isfinite(inn_arrays[short]["mean"]).sum()
        print(f"  {short}: {n_valid}/{n_rows} ({n_valid / n_rows:.1%})")

    df.to_parquet(a.out, index=False)
    print(f"Wrote {a.out}: {len(df)} rows, {len(df.columns)} cols")


if __name__ == "__main__":
    main()
