"""
Precompute grid-core-v0 patch chips from daily HRRR rasters.

Reads the raster stack once per day, extracts all station-centered 64x64
patches, and saves them as per-day .pt files. This eliminates raster I/O
during training.

Output per day:
    {out_dir}/{YYYY-MM-DD}.pt containing:
        x:           (N_stations, C, H, W) float32 — normalized patches
        sta_rows:    (N_stations, max_sta) int64
        sta_cols:    (N_stations, max_sta) int64
        sta_targets: (N_stations, max_sta, n_targets) float32
        sta_valid:   (N_stations, max_sta, n_targets) bool
        sta_holdout: (N_stations, max_sta) bool
        sta_is_center: (N_stations, max_sta) bool
        n_stations:  list of ints (actual station count per patch)
        fids:        list of center-station fids

Also saves:
    meta.json — feature manifest, holdout coverage, norm stats
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time

import pandas as pd
import torch

from models.hrrr_da.patch_assim_dataset import HRRRPatchDataset

print = functools.partial(print, flush=True)  # type: ignore[assignment]

# ── multiprocessing globals and worker ────────────────────────────────────────

_W_DS_KWARGS: dict = {}
_W_OUT_DIR: str = ""


def _process_day_chunk(chunk_days: list[str]) -> int:
    """Process a chunk of days in one worker with its own dataset."""
    worker_ds = HRRRPatchDataset(
        train_days=set(pd.Timestamp(d) for d in chunk_days),
        **_W_DS_KWARGS,
    )
    # Build day-to-index for this worker's dataset
    w_day_idx: dict[str, list[int]] = {}
    for idx in range(len(worker_ds)):
        d = pd.Timestamp(worker_ds.samples.iloc[idx]["day"]).strftime("%Y-%m-%d")
        w_day_idx.setdefault(d, []).append(idx)

    built = 0
    for day_key in chunk_days:
        indices = w_day_idx.get(day_key, [])
        if not indices:
            continue
        out_path = os.path.join(_W_OUT_DIR, f"{day_key}.pt")
        if os.path.exists(out_path):
            continue

        patches = [worker_ds[idx] for idx in indices]
        n_samples = len(patches)
        max_sta = max(p[1].shape[0] for p in patches)
        n_targets = patches[0][3].shape[1]

        x = torch.stack([p[0] for p in patches])
        sta_rows = torch.zeros(n_samples, max_sta, dtype=torch.long)
        sta_cols = torch.zeros(n_samples, max_sta, dtype=torch.long)
        sta_targets = torch.zeros(n_samples, max_sta, n_targets, dtype=torch.float32)
        sta_valid = torch.zeros(n_samples, max_sta, n_targets, dtype=torch.bool)
        sta_holdout = torch.zeros(n_samples, max_sta, dtype=torch.bool)
        sta_is_center = torch.zeros(n_samples, max_sta, dtype=torch.bool)
        n_stations = []
        fids = []

        for i, p in enumerate(patches):
            n = p[1].shape[0]
            n_stations.append(n)
            sta_rows[i, :n] = p[1]
            sta_cols[i, :n] = p[2]
            sta_targets[i, :n] = p[3]
            sta_valid[i, :n] = p[4]
            sta_holdout[i, :n] = p[5]
            sta_is_center[i, :n] = p[6]
            fids.append(str(worker_ds.samples.iloc[indices[i]]["fid"]))

        torch.save(
            {
                "x": x,
                "sta_rows": sta_rows,
                "sta_cols": sta_cols,
                "sta_targets": sta_targets,
                "sta_valid": sta_valid,
                "sta_holdout": sta_holdout,
                "sta_is_center": sta_is_center,
                "n_stations": n_stations,
                "fids": fids,
            },
            out_path,
        )
        built += 1
    return built


def _parse_args():
    p = argparse.ArgumentParser(description="Precompute grid-core-v0 chips.")
    p.add_argument("--table-path", required=True)
    p.add_argument("--background-dir", required=True)
    p.add_argument("--background-pattern", default="HRRR_1km_{date}.tif")
    p.add_argument("--static-tifs", nargs="+", required=True)
    p.add_argument("--landsat-tif", required=True)
    p.add_argument("--rsun-tif", required=True)
    p.add_argument("--cdr-dir", required=True)
    p.add_argument("--cdr-pattern", default="CDR_005deg_{date}.tif")
    p.add_argument("--holdout-fids-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--target-names", nargs="+", default=["delta_tmax", "delta_tmin"])
    p.add_argument("--drop-bands", nargs="*", default=["n_hours"])
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.holdout_fids_json) as f:
        holdout_fids = set(str(x) for x in json.load(f))

    # Build the full dataset (all years) to get norm stats from training data
    # We'll extract patches day by day
    print("Building dataset for norm stats (train years only)...")
    df = pd.read_parquet(args.table_path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["fid"] = df["fid"].astype(str)
    df["day"] = pd.to_datetime(df["day"])
    train_days = set(
        df[df["day"].dt.year.isin([2018, 2019, 2020, 2021, 2022, 2023])]["day"].unique()
    )

    # Build a train dataset to compute norm stats (exclude holdout from stats)
    norm_ds = HRRRPatchDataset(
        table_path=args.table_path,
        background_dir=args.background_dir,
        background_pattern=args.background_pattern,
        static_tifs=args.static_tifs,
        landsat_tif=args.landsat_tif,
        rsun_tif=args.rsun_tif,
        cdr_dir=args.cdr_dir,
        cdr_pattern=args.cdr_pattern,
        target_names=args.target_names,
        train_days=train_days,
        target_exclude_fids=holdout_fids,
        holdout_fids=holdout_fids,
        drop_bands=args.drop_bands,
        patch_size=args.patch_size,
    )
    norm_stats = norm_ds.norm_stats
    feature_names = norm_ds.feature_names
    in_channels = norm_ds.in_channels
    print(f"Norm stats computed: {in_channels} channels")

    # Save metadata
    meta = {
        "family": "grid-core-v0",
        "feature_names": feature_names,
        "target_names": args.target_names,
        "in_channels": in_channels,
        "patch_size": args.patch_size,
        "norm_stats": norm_stats,
        "holdout_fids_json": args.holdout_fids_json,
        "n_holdout_requested": len(holdout_fids),
        "drop_bands": args.drop_bands,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Build full dataset and group by day
    all_days = set(df["day"].unique())
    print(f"Building full dataset ({len(all_days)} days) with frozen norm stats...")
    full_ds = HRRRPatchDataset(
        table_path=args.table_path,
        background_dir=args.background_dir,
        background_pattern=args.background_pattern,
        static_tifs=args.static_tifs,
        landsat_tif=args.landsat_tif,
        rsun_tif=args.rsun_tif,
        cdr_dir=args.cdr_dir,
        cdr_pattern=args.cdr_pattern,
        target_names=args.target_names,
        train_days=all_days,
        holdout_fids=holdout_fids,
        drop_bands=args.drop_bands,
        norm_stats=norm_stats,
        patch_size=args.patch_size,
    )
    print(f"Total samples: {len(full_ds)}")

    day_to_indices: dict[str, list[int]] = {}
    for idx in range(len(full_ds)):
        day = pd.Timestamp(full_ds.samples.iloc[idx]["day"]).strftime("%Y-%m-%d")
        day_to_indices.setdefault(day, []).append(idx)

    days = sorted(day_to_indices.keys())
    # Filter out already-built days
    days_todo = [
        d for d in days if not os.path.exists(os.path.join(args.out_dir, f"{d}.pt"))
    ]
    print(
        f"Days: {len(days)} total, {len(days) - len(days_todo)} cached, {len(days_todo)} to build"
    )
    del full_ds  # free memory before spawning workers

    # Store dataset construction args for workers as module-level global
    global _W_DS_KWARGS, _W_OUT_DIR
    _W_DS_KWARGS = {
        "table_path": args.table_path,
        "background_dir": args.background_dir,
        "background_pattern": args.background_pattern,
        "static_tifs": args.static_tifs,
        "landsat_tif": args.landsat_tif,
        "rsun_tif": args.rsun_tif,
        "cdr_dir": args.cdr_dir,
        "cdr_pattern": args.cdr_pattern,
        "target_names": args.target_names,
        "holdout_fids": holdout_fids,
        "drop_bands": args.drop_bands,
        "norm_stats": norm_stats,
        "patch_size": args.patch_size,
    }
    _W_OUT_DIR = args.out_dir

    import multiprocessing as mp

    n_workers = args.workers
    chunk_size = max(1, len(days_todo) // n_workers)
    chunks = [
        days_todo[i : i + chunk_size] for i in range(0, len(days_todo), chunk_size)
    ]
    print(f"Processing {len(days_todo)} days with {len(chunks)} worker chunks...")

    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = pool.map(_process_day_chunk, chunks)
    total_built = sum(results)
    elapsed = time.time() - t0
    print(
        f"Built {total_built} day files in {elapsed / 60:.1f} min ({total_built / max(elapsed, 1):.1f} days/s)"
    )

    # Build chip index for fast dataset loading
    chip_index = {}
    for d in days:
        pt_path = os.path.join(args.out_dir, f"{d}.pt")
        if os.path.exists(pt_path):
            chip_index[d] = day_to_indices.get(d, [])
            chip_index[d] = len(chip_index[d])
    with open(os.path.join(args.out_dir, "chip_index.json"), "w") as f:
        json.dump(chip_index, f)
    print(
        f"Chip index: {len(chip_index)} days, {sum(chip_index.values())} total samples"
    )

    # Save holdout coverage
    effective_holdout = holdout_fids & set(df["fid"].unique())
    val_df = df[df["day"].dt.year == 2024]
    val_holdout = holdout_fids & set(val_df["fid"].unique())
    with open(os.path.join(args.out_dir, "effective_holdout_fids.json"), "w") as f:
        json.dump(sorted(effective_holdout), f)
    with open(os.path.join(args.out_dir, "val_holdout_fids.json"), "w") as f:
        json.dump(sorted(val_holdout), f)

    print(
        f"Done. {len(days)} chip files saved to {args.out_dir}\n"
        f"Holdout: {len(holdout_fids)} requested, "
        f"{len(effective_holdout)} effective, {len(val_holdout)} on val days"
    )


if __name__ == "__main__":
    main()
