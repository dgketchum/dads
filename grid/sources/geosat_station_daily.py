"""
Extract geostationary satellite features at station locations.

Reads 3-hourly GridSat-B1 / GOES ABI COG GeoTIFFs and extracts 5 daily
features at ~9,000 station lat/lons.  Writes per-station daily Parquets
with the same layout as RTMA/URMA/CDR station daily files.

Features extracted (all float32, _geosat suffix):
  irwin_cdr_mean_geosat  -- daily mean IR brightness temp
  irwin_cdr_min_geosat   -- daily min (thickest cloud)
  irwin_cdr_range_geosat -- daily range (diurnal cloud cycling proxy)
  irwvp_mean_geosat      -- daily mean water vapor channel
  vschn_mean_geosat      -- daily mean visible (daytime-only, nanmean)

Output: one Parquet per station in --out-dir, indexed by date.
Supports resume (skips dates already extracted) and multiprocessing.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import rasterio

from extract.rs.geosat_reader import CHANNELS, resolve_cog_path
from extract.rs.goes_abi.goes_abi_common import HOURS_3H

_COL_NAMES = [
    "irwin_cdr_mean_geosat",
    "irwin_cdr_min_geosat",
    "irwin_cdr_range_geosat",
    "irwvp_mean_geosat",
    "vschn_mean_geosat",
]
_N_COLS = len(_COL_NAMES)

# ── multiprocessing globals ──────────────────────────────────────────────────

_W_ROW_IDX: np.ndarray | None = None
_W_COL_IDX: np.ndarray | None = None
_W_N_STATIONS: int = 0
_W_GRIDSAT_ROOT: str = ""
_W_GOES_ROOT: str = ""


def _init_worker(
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    n_stations: int,
    gridsat_root: str,
    goes_root: str,
) -> None:
    global _W_ROW_IDX, _W_COL_IDX, _W_N_STATIONS, _W_GRIDSAT_ROOT, _W_GOES_ROOT
    _W_ROW_IDX = row_idx
    _W_COL_IDX = col_idx
    _W_N_STATIONS = n_stations
    _W_GRIDSAT_ROOT = gridsat_root
    _W_GOES_ROOT = goes_root


def _process_one_day(day_str: str) -> tuple[str, np.ndarray] | None:
    """Process one day. Returns (date_str, array (n_stations, 5)) or None."""
    d = datetime.strptime(day_str, "%Y-%m-%d").date()
    n = _W_N_STATIONS
    row_idx = _W_ROW_IDX
    col_idx = _W_COL_IDX

    # Collect per-channel stacks: channel -> list of 1-D arrays (n_stations,)
    channel_vals: dict[str, list[np.ndarray]] = {ch: [] for ch in CHANNELS}

    for hour in HOURS_3H:
        for channel in CHANNELS:
            path = resolve_cog_path(d, hour, channel, _W_GRIDSAT_ROOT, _W_GOES_ROOT)
            if path is None:
                continue
            try:
                with rasterio.open(path) as src:
                    band = src.read(1)
                    vals = band[row_idx, col_idx].astype("float32")
                    # Mask nodata
                    if src.nodata is not None:
                        vals[vals == src.nodata] = np.nan
                    channel_vals[channel].append(vals)
            except Exception:
                continue

    # Check if we got any data at all
    total_reads = sum(len(v) for v in channel_vals.values())
    if total_reads == 0:
        return None

    out = np.full((n, _N_COLS), np.nan, dtype="float32")

    # irwin_cdr: mean, min, range
    if channel_vals["irwin_cdr"]:
        stack = np.stack(channel_vals["irwin_cdr"], axis=0)  # (n_hours, n_stations)
        out[:, 0] = np.nanmean(stack, axis=0)
        out[:, 1] = np.nanmin(stack, axis=0)
        out[:, 2] = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)

    # irwvp: mean
    if channel_vals["irwvp"]:
        stack = np.stack(channel_vals["irwvp"], axis=0)
        out[:, 3] = np.nanmean(stack, axis=0)

    # vschn: mean (daytime only, nanmean handles NaN from nighttime)
    if channel_vals["vschn"]:
        stack = np.stack(channel_vals["vschn"], axis=0)
        out[:, 4] = np.nanmean(stack, axis=0)

    return day_str, out


def _compute_pixel_indices(
    lats: np.ndarray,
    lons: np.ndarray,
    gridsat_root: str,
    goes_root: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute row/col pixel indices from the first available COG.

    All COGs share identical grid geometry, so we only need one.
    """
    # Search for any existing COG to get the transform
    sample_cog = None
    for root_dir in (gridsat_root, goes_root):
        if not os.path.isdir(root_dir):
            continue
        for year_dir in sorted(os.listdir(root_dir)):
            year_path = os.path.join(root_dir, year_dir)
            if not os.path.isdir(year_path):
                continue
            for f in os.listdir(year_path):
                if f.endswith(".tif"):
                    sample_cog = os.path.join(year_path, f)
                    break
            if sample_cog:
                break
        if sample_cog:
            break

    if sample_cog is None:
        raise FileNotFoundError(f"No COG files found in {gridsat_root} or {goes_root}")

    with rasterio.open(sample_cog) as src:
        rows_cols = [src.index(lon, lat) for lon, lat in zip(lons, lats)]
        row_idx = np.clip(np.array([rc[0] for rc in rows_cols]), 0, src.height - 1)
        col_idx = np.clip(np.array([rc[1] for rc in rows_cols]), 0, src.width - 1)

    return row_idx.astype(np.intp), col_idx.astype(np.intp)


def extract_geosat_stations(
    out_dir: str,
    stations_csv: str,
    gridsat_root: str = "/nas/dads/rs/gridsat_b1",
    goes_root: str = "/nas/dads/rs/goes_abi",
    start_date: str = "2018-01-01",
    end_date: str = "2024-12-31",
    num_workers: int = 4,
    flush_every: int = 100,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Load stations
    stations = pd.read_csv(stations_csv)
    id_col = "fid" if "fid" in stations.columns else "station_id"
    fids = stations[id_col].astype(str).values
    lats = stations["latitude"].values.astype("float64")
    lons = stations["longitude"].values.astype("float64")
    n_stations = len(fids)

    row_idx, col_idx = _compute_pixel_indices(lats, lons, gridsat_root, goes_root)
    print(f"Stations: {n_stations}", flush=True)

    # Load existing dates per station for resume
    existing_dates: dict[str, set[str]] = {}
    for fid in fids:
        p = os.path.join(out_dir, f"{fid}.parquet")
        if os.path.exists(p):
            try:
                df = pd.read_parquet(p)
                idx = pd.to_datetime(df.index, errors="coerce")
                existing_dates[fid] = set(idx.strftime("%Y-%m-%d"))
            except Exception:
                existing_dates[fid] = set()
        else:
            existing_dates[fid] = set()

    # Find dates already fully extracted (present in ALL stations)
    if existing_dates:
        all_sets = list(existing_dates.values())
        done_dates = set.intersection(*all_sets) if all_sets else set()
    else:
        done_dates = set()
    print(f"Already extracted dates: {len(done_dates)}", flush=True)

    # Build day list
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    all_days: list[str] = []
    cur = start
    while cur <= end:
        ds = cur.strftime("%Y-%m-%d")
        if ds not in done_dates:
            all_days.append(ds)
        cur += timedelta(days=1)

    print(f"Days to process: {len(all_days)}", flush=True)
    if not all_days:
        print("Nothing to do.", flush=True)
        return

    # Per-station accumulators: fid_idx -> list of (date_str, row_array)
    accum: dict[int, list[tuple[str, np.ndarray]]] = {i: [] for i in range(n_stations)}

    t0 = time.time()
    processed = 0

    def _flush():
        """Write accumulated rows to per-station parquets (append-merge)."""
        for i in range(n_stations):
            if not accum[i]:
                continue
            fid = fids[i]
            dates = [r[0] for r in accum[i]]
            vals = np.stack([r[1] for r in accum[i]])
            new_df = pd.DataFrame(vals, index=pd.to_datetime(dates), columns=_COL_NAMES)
            new_df.index.name = "day"

            out_path = os.path.join(out_dir, f"{fid}.parquet")
            if os.path.exists(out_path):
                try:
                    old = pd.read_parquet(out_path)
                    old.index = pd.to_datetime(old.index, errors="coerce")
                    old = old[old.index.notna()]
                    combined = pd.concat([old, new_df]).sort_index()
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined.to_parquet(out_path)
                except Exception:
                    new_df.to_parquet(out_path)
            else:
                new_df.to_parquet(out_path)
            accum[i] = []

    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(row_idx, col_idx, n_stations, gridsat_root, goes_root),
    ) as pool:
        for result in pool.imap_unordered(_process_one_day, all_days, chunksize=4):
            if result is None:
                continue
            date_str, data = result

            for i in range(n_stations):
                accum[i].append((date_str, data[i]))

            processed += 1
            if processed % flush_every == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed
                eta = (len(all_days) - processed) / rate if rate > 0 else 0
                print(
                    f"  [{processed}/{len(all_days)}]  "
                    f"{rate:.1f} days/s  ETA {eta / 60:.1f} min  (flushing...)",
                    flush=True,
                )
                _flush()

    # Final flush
    _flush()

    elapsed = time.time() - t0
    print(
        f"Done: {processed} days -> {out_dir}  ({elapsed / 60:.1f} min)",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract geostationary satellite features at station locations."
    )
    p.add_argument(
        "--out-dir", required=True, help="Output dir for per-station Parquets."
    )
    p.add_argument("--stations-csv", required=True, help="Station inventory CSV.")
    p.add_argument(
        "--gridsat-root",
        default="/nas/dads/rs/gridsat_b1",
        help="Root directory of GridSat-B1 COGs.",
    )
    p.add_argument(
        "--goes-root",
        default="/nas/dads/rs/goes_abi",
        help="Root directory of GOES ABI COGs.",
    )
    p.add_argument("--start-date", default="2018-01-01")
    p.add_argument("--end-date", default="2024-12-31")
    p.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    extract_geosat_stations(
        out_dir=a.out_dir,
        stations_csv=a.stations_csv,
        gridsat_root=a.gridsat_root,
        goes_root=a.goes_root,
        start_date=a.start_date,
        end_date=a.end_date,
        num_workers=a.workers,
    )
