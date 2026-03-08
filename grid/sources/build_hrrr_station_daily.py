"""
Build per-station daily Parquets from hourly HRRR GRIB2 files.

Reads concatenated 9-message HRRR GRIB2 files via eccodes, samples at
station locations using a BallTree (haversine), computes daily aggregates
(mean, tmax, tmin), and writes one Parquet per station.

Usage
-----
    python -m grid.sources.build_hrrr_station_daily \
        --grib-root /mnt/mco_nas1/shared/hrrr_hourly \
        --stations-csv artifacts/madis_pnw.csv \
        --out-dir /nas/dads/mvp/hrrr_daily \
        --start 2018-01-01 --end 2024-12-31 \
        --bounds -125.0 42.0 -104.0 49.0 \
        --workers 4

Output contract (per station file)
-----------------------------------
Index: daily DatetimeIndex (UTC day, naive timestamps)
Columns:
  tmp_hrrr     : daily mean 2 m temperature [degC]
  tmax_hrrr    : daily max 2 m temperature [degC]
  tmin_hrrr    : daily min 2 m temperature [degC]
  dpt_hrrr     : daily mean 2 m dewpoint [degC]
  ea_hrrr      : daily mean actual vapor pressure [kPa]
  pres_hrrr    : daily mean surface pressure [kPa]
  ugrd_hrrr    : daily mean 10 m u-wind [m/s]
  vgrd_hrrr    : daily mean 10 m v-wind [m/s]
  wind_hrrr    : daily mean 10 m wind speed [m/s]
  wdir_hrrr    : daily vector-mean wind direction [deg]
  dswrf_hrrr   : daily mean downward shortwave [W/m2]
  spfh_hrrr    : daily mean 2 m specific humidity [kg/kg]
  tcdc_hrrr    : daily mean total cloud cover [%]
  hpbl_hrrr    : daily mean PBL height [m]
  n_hours      : valid hour count
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from datetime import datetime, timedelta

import eccodes
import numpy as np
import pandas as pd

from grid.sources.download_hrrr_archive import TARGET_FIELDS
from grid.station_extract import build_haversine_tree, load_station_csv, query_nearest

# ── constants ────────────────────────────────────────────────────────────────

# Message order in the 9var GRIB matches TARGET_FIELDS in download_hrrr_archive.py:
#   0: TMP, 1: DPT, 2: UGRD, 3: VGRD, 4: DSWRF, 5: PRES, 6: TCDC, 7: HPBL, 8: SPFH
_MSG_NAMES = [var for var, _ in TARGET_FIELDS]
_MSG_INDEX = {name: i for i, name in enumerate(_MSG_NAMES)}
_N_MSGS = len(TARGET_FIELDS)

_SFX = "_hrrr"

# Version boundaries (same as hrrr_common.py)
_VERSION_BOUNDARIES = [
    ("v1", datetime(2014, 11, 15), datetime(2016, 8, 22)),
    ("v2", datetime(2016, 8, 23), datetime(2018, 7, 11)),
    ("v3", datetime(2018, 7, 12), datetime(2020, 12, 1)),
    ("v4", datetime(2020, 12, 2), datetime(2099, 12, 31)),
]


def _version_for_date(d: datetime) -> str:
    for ver, start, end in _VERSION_BOUNDARIES:
        if start <= d <= end:
            return ver
    return "v4"


def _grib_path(grib_root: str, day: datetime, hour: int) -> str | None:
    ver = _version_for_date(day)
    p = os.path.join(
        grib_root,
        ver,
        f"{day:%Y}",
        f"{day:%Y%m%d}",
        f"hrrr.t{hour:02d}z.9var.grib2",
    )
    return p if os.path.exists(p) else None


# ── grid decoding ────────────────────────────────────────────────────────────


def _decode_hour(path: str, n_msgs: int = _N_MSGS) -> np.ndarray | None:
    """Decode all messages from a 9var GRIB file.

    Returns shape (n_msgs, n_points) or None on failure.
    """
    arrays: list[np.ndarray] = []
    with open(path, "rb") as f:
        for _ in range(n_msgs):
            msgid = eccodes.codes_grib_new_from_file(f)
            if msgid is None:
                break
            vals = eccodes.codes_get_array(msgid, "values")
            arrays.append(vals)
            eccodes.codes_release(msgid)
    if len(arrays) != n_msgs:
        return None
    return np.array(arrays, dtype=np.float64)


def _get_grid_coords(grib_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract 1-D lat/lon arrays from the first message of a GRIB file."""
    with open(grib_path, "rb") as f:
        msgid = eccodes.codes_grib_new_from_file(f)
        lats = eccodes.codes_get_array(msgid, "latitudes")
        lons = eccodes.codes_get_array(msgid, "longitudes")
        eccodes.codes_release(msgid)
    return lats, lons


# ── daily aggregation ────────────────────────────────────────────────────────


def _ea_from_dpt(dpt_c: np.ndarray) -> np.ndarray:
    """Tetens formula: dewpoint (degC) -> actual vapour pressure (kPa)."""
    return 0.6108 * np.exp(17.27 * dpt_c / (dpt_c + 237.3))


def _process_day(
    grib_root: str,
    day: datetime,
    grid_indices: np.ndarray,
    n_stations: int,
) -> dict[str, np.ndarray] | None:
    """Read 24 hourly GRIBs for one day, sample at stations, aggregate daily."""
    hourly: list[np.ndarray] = []  # each: (n_msgs, n_stations)

    for hour in range(24):
        path = _grib_path(grib_root, day, hour)
        if path is None:
            continue
        data = _decode_hour(path)
        if data is None:
            continue
        # Sample at station grid indices: (n_msgs, n_stations)
        hourly.append(data[:, grid_indices])

    if not hourly:
        return None

    # (n_hours, n_msgs, n_stations)
    stack = np.array(hourly, dtype=np.float64)

    # Unit conversions
    # TMP (idx 0) and DPT (idx 1): K -> degC
    stack[:, _MSG_INDEX["TMP"], :] -= 273.15
    stack[:, _MSG_INDEX["DPT"], :] -= 273.15
    # PRES (idx 5): Pa -> kPa
    stack[:, _MSG_INDEX["PRES"], :] /= 1000.0

    daily: dict[str, np.ndarray] = {}

    # Mean aggregation
    for elem, col in [
        ("TMP", "tmp"),
        ("DPT", "dpt"),
        ("PRES", "pres"),
        ("UGRD", "ugrd"),
        ("VGRD", "vgrd"),
        ("SPFH", "spfh"),
        ("DSWRF", "dswrf"),
        ("TCDC", "tcdc"),
        ("HPBL", "hpbl"),
    ]:
        if elem in _MSG_INDEX:
            daily[col + _SFX] = np.nanmean(stack[:, _MSG_INDEX[elem], :], axis=0)

    # tmax / tmin
    daily["tmax" + _SFX] = np.nanmax(stack[:, _MSG_INDEX["TMP"], :], axis=0)
    daily["tmin" + _SFX] = np.nanmin(stack[:, _MSG_INDEX["TMP"], :], axis=0)

    # ea: mean of hourly ea (Tetens on each hour's DPT, then average)
    hourly_ea = _ea_from_dpt(stack[:, _MSG_INDEX["DPT"], :])
    daily["ea" + _SFX] = np.nanmean(hourly_ea, axis=0)

    # wind speed: from hourly u, v
    u_hourly = stack[:, _MSG_INDEX["UGRD"], :]
    v_hourly = stack[:, _MSG_INDEX["VGRD"], :]
    spd_hourly = np.sqrt(u_hourly**2 + v_hourly**2)
    daily["wind" + _SFX] = np.nanmean(spd_hourly, axis=0)

    # wind direction: vector-average via mean U, V
    mu = np.nanmean(u_hourly, axis=0)
    mv = np.nanmean(v_hourly, axis=0)
    daily["wdir" + _SFX] = np.degrees(np.arctan2(-mu, -mv)) % 360.0

    daily["n_hours"] = np.full(n_stations, len(hourly), dtype=np.int16)
    return daily


# ── multiprocessing ──────────────────────────────────────────────────────────

_W_GRIB_ROOT: str = ""
_W_GRID_INDICES: np.ndarray | None = None
_W_N_STATIONS: int = 0


def _init_worker(grib_root: str, grid_indices: np.ndarray, n_stations: int) -> None:
    global _W_GRIB_ROOT, _W_GRID_INDICES, _W_N_STATIONS
    _W_GRIB_ROOT = grib_root
    _W_GRID_INDICES = grid_indices
    _W_N_STATIONS = n_stations


def _worker_process_day(day: datetime) -> tuple[datetime, dict[str, np.ndarray] | None]:
    return day, _process_day(_W_GRIB_ROOT, day, _W_GRID_INDICES, _W_N_STATIONS)


# ── main ─────────────────────────────────────────────────────────────────────


def build_hrrr_station_daily(
    grib_root: str,
    stations_csv: str,
    out_dir: str,
    start: datetime,
    end: datetime,
    bounds: tuple[float, float, float, float] | None = None,
    overwrite: bool = False,
    workers: int = 1,
) -> int:
    """Build per-station daily Parquets from hourly HRRR GRIB2 archive."""
    stations = load_station_csv(stations_csv, bounds)
    n = len(stations)
    print(f"Stations loaded: {n}", flush=True)
    os.makedirs(out_dir, exist_ok=True)

    # Find a reference GRIB to build the BallTree from
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

    # Build BallTree from HRRR grid coordinates
    grid_lats, grid_lons = _get_grid_coords(ref_path)
    print(
        f"HRRR grid: {len(grid_lats)} points, "
        f"lat [{grid_lats.min():.2f}, {grid_lats.max():.2f}], "
        f"lon [{grid_lons.min():.2f}, {grid_lons.max():.2f}]",
        flush=True,
    )

    tree = build_haversine_tree(grid_lats, grid_lons)

    # Convert station lons to 0-360 to match HRRR convention
    sta_lats = stations["latitude"].values
    sta_lons = stations["longitude"].values
    sta_lons_360 = sta_lons % 360.0

    grid_indices = query_nearest(tree, sta_lats, sta_lons_360)
    print(f"Nearest grid points found for {len(grid_indices)} stations", flush=True)

    # Process days
    n_days = (end - start).days + 1
    all_days = [start + timedelta(days=i) for i in range(n_days)]
    days_list: list[pd.Timestamp] = []
    day_arrays: list[dict[str, np.ndarray]] = []

    fids = stations["fid"].values
    workers = max(1, workers)
    print(f"Processing {n_days} days with {workers} worker(s)...", flush=True)

    if workers == 1:
        for di, day in enumerate(all_days):
            daily = _process_day(grib_root, day, grid_indices, n)
            if daily is not None:
                days_list.append(pd.Timestamp(day))
                day_arrays.append(daily)
            if (di + 1) % 10 == 0 or di == n_days - 1:
                print(f"  {di + 1}/{n_days} days ({day:%Y-%m-%d})", flush=True)
    else:
        done = 0
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
                if done % 10 == 0 or done == n_days:
                    print(f"  {done}/{n_days} days", flush=True)

    if not day_arrays:
        raise RuntimeError("No valid days processed.")

    # Sort by day (imap_unordered may shuffle)
    order = sorted(range(len(days_list)), key=lambda i: days_list[i])
    days_list = [days_list[i] for i in order]
    day_arrays = [day_arrays[i] for i in order]

    # Build big DataFrame, split by station, write
    columns = list(day_arrays[0].keys())
    n_actual = len(days_list)

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


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hourly HRRR GRIB2 -> per-station daily Parquets."
    )
    p.add_argument(
        "--grib-root",
        required=True,
        help="Root of HRRR hourly GRIB archive.",
    )
    p.add_argument(
        "--stations-csv",
        required=True,
        help="Station inventory CSV (fid, latitude, longitude).",
    )
    p.add_argument("--out-dir", required=True, help="Output directory for Parquets.")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD).")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD).")
    p.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("W", "S", "E", "N"),
        help="Lon/lat bounding box to clip stations.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers for day processing (default: 4).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    build_hrrr_station_daily(
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
