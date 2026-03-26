"""
Build per-station daily Parquets with hourly HRRR values from GRIB2 files.

Same input data as build_hrrr_station_daily.py, but keeps all 24 hourly
values for each of the 9 GRIB variables instead of aggregating to daily.

Output columns (per station, indexed by day):
  {VAR}_{HH}_hrrr  -- 9 vars x 24 hours = 216 columns
  tmax_hrrr         -- daily max TMP (for target computation)
  tmin_hrrr         -- daily min TMP (for target computation)
  n_hours           -- valid hour count

Usage:
    python -m grid.sources.build_hrrr_station_hourly \
        --grib-root /mnt/mco_nas1/shared/hrrr_hourly \
        --stations-csv artifacts/madis_pnw.csv \
        --out-dir /nas/dads/mvp/hrrr_hourly_stations \
        --start 2018-01-01 --end 2024-12-31 \
        --bounds -125.0 42.0 -104.0 49.0 \
        --workers 4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from grid.sources.build_hrrr_station_daily import (
    _MSG_INDEX,
    _N_MSGS,
    _decode_hour,
    _get_grid_coords,
    _grib_path,
)
from grid.station_extract import build_haversine_tree, load_station_csv, query_nearest

_SFX = "_hrrr"

# 9 raw GRIB variable names in output order
_RAW_VARS = ["TMP", "DPT", "UGRD", "VGRD", "DSWRF", "PRES", "TCDC", "HPBL", "SPFH"]
_RAW_COLS = [v.lower() for v in _RAW_VARS]

# Build column names: tmp_00_hrrr, tmp_01_hrrr, ..., spfh_23_hrrr
_HOURLY_COL_NAMES = [f"{col}_{h:02d}{_SFX}" for col in _RAW_COLS for h in range(24)]


def _process_day_hourly(
    grib_root: str,
    day: datetime,
    grid_indices: np.ndarray,
    n_stations: int,
) -> dict[str, np.ndarray] | None:
    """Read 24 hourly GRIBs, sample at stations, return per-hour values."""
    # (24, n_msgs, n_stations) — fill with NaN for missing hours
    cube = np.full((24, _N_MSGS, n_stations), np.nan, dtype=np.float64)
    n_valid = 0

    for hour in range(24):
        path = _grib_path(grib_root, day, hour)
        if path is None:
            continue
        data = _decode_hour(path)
        if data is None:
            continue
        cube[hour, :, :] = data[:, grid_indices]
        n_valid += 1

    if n_valid == 0:
        return None

    # Unit conversions (in-place on cube)
    cube[:, _MSG_INDEX["TMP"], :] -= 273.15  # K -> degC
    cube[:, _MSG_INDEX["DPT"], :] -= 273.15  # K -> degC
    cube[:, _MSG_INDEX["PRES"], :] /= 1000.0  # Pa -> kPa

    result: dict[str, np.ndarray] = {}

    # Per-hour columns: var_HH_hrrr
    for raw_var, col in zip(_RAW_VARS, _RAW_COLS):
        msg_idx = _MSG_INDEX[raw_var]
        for hour in range(24):
            result[f"{col}_{hour:02d}{_SFX}"] = cube[hour, msg_idx, :].astype("float32")

    # Daily tmax/tmin from hourly TMP (needed for target computation)
    tmp_all = cube[:, _MSG_INDEX["TMP"], :]
    result[f"tmax{_SFX}"] = np.nanmax(tmp_all, axis=0).astype("float32")
    result[f"tmin{_SFX}"] = np.nanmin(tmp_all, axis=0).astype("float32")

    result["n_hours"] = np.full(n_stations, n_valid, dtype=np.int16)
    return result


# ── multiprocessing ──────────────────────────────────────────────────────────

_W_GRIB_ROOT: str = ""
_W_GRID_INDICES: np.ndarray | None = None
_W_N_STATIONS: int = 0


def _init_worker(grib_root: str, grid_indices: np.ndarray, n_stations: int) -> None:
    global _W_GRIB_ROOT, _W_GRID_INDICES, _W_N_STATIONS
    _W_GRIB_ROOT = grib_root
    _W_GRID_INDICES = grid_indices
    _W_N_STATIONS = n_stations


def _worker_process_day(
    day: datetime,
) -> tuple[datetime, dict[str, np.ndarray] | None]:
    return day, _process_day_hourly(_W_GRIB_ROOT, day, _W_GRID_INDICES, _W_N_STATIONS)


def build_hrrr_station_hourly(
    grib_root: str,
    stations_csv: str,
    out_dir: str,
    start: datetime,
    end: datetime,
    bounds: tuple[float, float, float, float] | None = None,
    overwrite: bool = False,
    workers: int = 1,
) -> int:
    """Build per-station Parquets with hourly HRRR values."""
    stations = load_station_csv(stations_csv, bounds)
    n = len(stations)
    print(f"Stations loaded: {n}", flush=True)
    os.makedirs(out_dir, exist_ok=True)

    ref_path = None
    d = start
    while d <= end:
        for h in range(24):
            ref_path = _grib_path(grib_root, d, h)
            if ref_path:
                break
        if ref_path:
            break
        d += timedelta(days=1)
    if not ref_path:
        raise FileNotFoundError(f"No HRRR GRIBs found in [{start}, {end}]")

    print(f"Reference GRIB: {ref_path}", flush=True)
    grid_lats, grid_lons = _get_grid_coords(ref_path)
    tree = build_haversine_tree(grid_lats, grid_lons)

    sta_lats = stations["latitude"].values
    sta_lons = stations["longitude"].values % 360.0
    grid_indices = query_nearest(tree, sta_lats, sta_lons)
    print(f"Nearest grid points for {len(grid_indices)} stations", flush=True)

    n_days = (end - start).days + 1
    all_days = [start + timedelta(days=i) for i in range(n_days)]
    days_list: list[pd.Timestamp] = []
    day_arrays: list[dict[str, np.ndarray]] = []

    fids = stations["fid"].values
    workers = max(1, workers)
    print(f"Processing {n_days} days with {workers} worker(s)...", flush=True)

    done = 0
    if workers == 1:
        for day in all_days:
            daily = _process_day_hourly(grib_root, day, grid_indices, n)
            done += 1
            if daily is not None:
                days_list.append(pd.Timestamp(day))
                day_arrays.append(daily)
            if done % 10 == 0:
                print(f"  {done}/{n_days} days ({day:%Y-%m-%d})", flush=True)
    else:
        with mp.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(grib_root, grid_indices, n),
        ) as pool:
            for day, daily in pool.imap_unordered(
                _worker_process_day, all_days, chunksize=4
            ):
                done += 1
                if daily is not None:
                    days_list.append(pd.Timestamp(day))
                    day_arrays.append(daily)
                if done % 10 == 0:
                    print(f"  {done}/{n_days} days", flush=True)

    if not day_arrays:
        raise RuntimeError("No valid days processed.")

    order = sorted(range(len(days_list)), key=lambda i: days_list[i])
    days_list = [days_list[i] for i in order]
    day_arrays = [day_arrays[i] for i in order]

    columns = list(day_arrays[0].keys())
    n_actual = len(days_list)
    print(f"Valid days: {n_actual}, columns: {len(columns)}", flush=True)

    big: dict[str, np.ndarray] = {
        c: np.concatenate([d[c] for d in day_arrays]) for c in columns
    }
    big["fid"] = np.tile(fids, n_actual)
    big["day"] = np.repeat(days_list, n)

    df = pd.DataFrame(big)
    written = 0
    for fid, grp in df.groupby("fid"):
        out_path = os.path.join(out_dir, f"{fid}.parquet")
        if not overwrite and os.path.exists(out_path):
            continue
        grp = grp.drop(columns=["fid"]).set_index("day").sort_index()
        grp.to_parquet(out_path)
        written += 1

    print(f"Wrote {written} station Parquets to {out_dir}", flush=True)
    return written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hourly HRRR GRIB2 -> per-station Parquets with hourly columns."
    )
    p.add_argument("--grib-root", required=True)
    p.add_argument("--stations-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument(
        "--bounds", nargs=4, type=float, metavar=("W", "S", "E", "N"), default=None
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    build_hrrr_station_hourly(
        grib_root=a.grib_root,
        stations_csv=a.stations_csv,
        out_dir=a.out_dir,
        start=datetime.strptime(a.start, "%Y-%m-%d"),
        end=datetime.strptime(a.end, "%Y-%m-%d"),
        bounds=tuple(a.bounds) if a.bounds else None,
        overwrite=a.overwrite,
        workers=a.workers,
    )


if __name__ == "__main__":
    main()
