"""
Build per-station daily Parquets from hourly RTMA/URMA GRIB2 files.

Reads hourly GRIB2 files from the archive, samples at station locations
(projected to the GRIB's Lambert Conformal grid), computes daily aggregates
(mean, tmax, tmin), and writes one Parquet per station.

Usage
-----
    uv run python -m grid.sources.build_station_daily \
        --model urma \
        --grib-root /mnt/mco_nas1/shared/rtma_hourly \
        --stations-csv /nas/dads/met/stations/madis_02JULY2025_mgrs.csv \
        --out-dir /mnt/mco_nas1/shared/rtma_daily/urma \
        --start 2024-01-01 --end 2024-12-31 \
        --bounds -125.0 42.0 -104.0 49.0 \
        --workers 4

Output contract (per station file)
-----------------------------------
Index: daily DatetimeIndex (UTC day, naive timestamps)
Columns (suffix ``_{model}`` where model in {rtma, urma}):
  - tmp_{model}    : daily mean 2 m temperature [degC]
  - tmax_{model}   : daily max 2 m temperature [degC]
  - tmin_{model}   : daily min 2 m temperature [degC]
  - dpt_{model}    : daily mean 2 m dewpoint [degC]
  - ea_{model}     : daily mean actual vapor pressure [kPa] (Tetens from hourly DPT)
  - pres_{model}   : daily mean surface pressure [kPa]
  - ugrd_{model}   : daily mean 10 m u-wind [m/s]
  - vgrd_{model}   : daily mean 10 m v-wind [m/s]
  - wind_{model}   : daily mean 10 m wind speed [m/s]
  - wdir_{model}   : daily vector-mean wind direction [deg] (from mean U, V)
  - gust_{model}   : daily max wind gust [m/s]
  - spfh_{model}   : daily mean 2 m specific humidity [kg/kg]
  - tcdc_{model}   : daily mean total cloud cover [%]
  - n_hours        : number of valid hourly files for that day
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from grid.station_extract import load_station_csv, project_to_lcc_pixels

# ------------------------------------------------------------------ constants

# GRIB_ELEMENT names to extract (NCEP convention, confirmed via rasterio tags).
EXTRACT_ELEMENTS = [
    "TMP",
    "DPT",
    "PRES",
    "UGRD",
    "VGRD",
    "SPFH",
    "WIND",
    "GUST",
    "TCDC",
]

# GRIB_ELEMENT -> base column name (before _{model} suffix)
_E2C = {
    "TMP": "tmp",
    "DPT": "dpt",
    "PRES": "pres",
    "UGRD": "ugrd",
    "VGRD": "vgrd",
    "SPFH": "spfh",
    "WIND": "wind",
    "GUST": "gust",
    "TCDC": "tcdc",
}

# ---- worker globals (set once per process via _init_worker) ----
_W_GRIB_ROOT: str = ""
_W_MODEL: str = ""
_W_ROWS_ADJ: np.ndarray | None = None
_W_COLS_ADJ: np.ndarray | None = None
_W_VALID: np.ndarray | None = None
_W_WINDOW: Window | None = None
_W_N_STATIONS: int = 0


# ----------------------------------------------------------- helper functions


def _find_grib(grib_root: str, model: str, day_str: str, hour: int) -> str | None:
    """Locate the GRIB2 file for *model*, day (YYYYMMDD), and UTC hour."""
    day_dir = os.path.join(grib_root, model, day_str[:4], day_str)
    base = f"{model}2p5.t{hour:02d}z.2dvaranl_ndfd"
    for suffix in (".grb2_wexp", ".grb2_ext", ".grb2"):
        p = os.path.join(day_dir, base + suffix)
        if os.path.exists(p):
            return p
    return None


def _build_band_map(grib_path: str) -> dict[str, int]:
    """Return ``{GRIB_ELEMENT: band_index}`` for the elements we need."""
    band_map: dict[str, int] = {}
    with rasterio.open(grib_path) as src:
        for bidx in range(1, src.count + 1):
            elem = src.tags(bidx).get("GRIB_ELEMENT", "")
            if elem in EXTRACT_ELEMENTS:
                band_map[elem] = bidx
    missing = set(EXTRACT_ELEMENTS) - set(band_map)
    if missing:
        print(f"  Warning: GRIB missing elements {sorted(missing)}", flush=True)
    return band_map


def _ea_from_dpt(dpt_c: np.ndarray) -> np.ndarray:
    """Tetens formula: dewpoint (degC) -> actual vapour pressure (kPa)."""
    return 0.6108 * np.exp(17.27 * dpt_c / (dpt_c + 237.3))


# ----------------------------------------------------------- day processing


def _process_day(
    grib_root: str,
    model: str,
    day: datetime,
    rows_adj: np.ndarray,
    cols_adj: np.ndarray,
    valid: np.ndarray,
    window: Window,
    n_stations: int,
) -> dict[str, np.ndarray] | None:
    """Read 24 hourly GRIBs for *day*, extract station pixels, aggregate.

    Band maps are built per-file because some hours carry extra bands
    (e.g. URMA 08z has TMAX, 20z has TMIN) which shift band positions.
    """
    day_str = day.strftime("%Y%m%d")

    # Canonical element order — consistent across all hours.
    target_elements = sorted(EXTRACT_ELEMENTS)
    eidx = {e: i for i, e in enumerate(target_elements)}
    n_elem = len(target_elements)

    hourly: list[np.ndarray] = []  # each: (n_elem, n_stations)

    for hour in range(24):
        grib_path = _find_grib(grib_root, model, day_str, hour)
        if grib_path is None:
            continue

        with rasterio.open(grib_path) as src:
            # Build band map for THIS file (band positions vary by hour).
            file_band_map: dict[str, int] = {}
            for bidx in range(1, src.count + 1):
                elem = src.tags(bidx).get("GRIB_ELEMENT", "")
                if elem in EXTRACT_ELEMENTS:
                    file_band_map[elem] = bidx

            present = [e for e in target_elements if e in file_band_map]
            if not present:
                continue
            band_indices = [file_band_map[e] for e in present]
            data = src.read(indexes=band_indices, window=window)

        arr = np.full((n_elem, n_stations), np.nan)
        for k, elem in enumerate(present):
            arr[eidx[elem], valid] = data[k, rows_adj[valid], cols_adj[valid]]
        hourly.append(arr)

    if not hourly:
        return None

    # (n_hours, n_elem, n_stations)
    stack = np.array(hourly, dtype=np.float64)

    # Unit conversions (in-place).
    # TMP and DPT are already in degC (GDAL converts from K).
    # PRES is in Pa -> kPa.
    if "PRES" in eidx:
        stack[:, eidx["PRES"], :] /= 1000.0

    sfx = f"_{model}"
    daily: dict[str, np.ndarray] = {}

    # --- mean aggregation ---
    for elem in ("TMP", "DPT", "PRES", "UGRD", "VGRD", "SPFH", "WIND", "TCDC"):
        if elem in eidx:
            daily[_E2C[elem] + sfx] = np.nanmean(stack[:, eidx[elem], :], axis=0)

    # --- tmax / tmin ---
    if "TMP" in eidx:
        daily["tmax" + sfx] = np.nanmax(stack[:, eidx["TMP"], :], axis=0)
        daily["tmin" + sfx] = np.nanmin(stack[:, eidx["TMP"], :], axis=0)

    # --- daily max gust ---
    if "GUST" in eidx:
        daily["gust" + sfx] = np.nanmax(stack[:, eidx["GUST"], :], axis=0)

    # --- ea: mean of hourly ea (Tetens on each hour, then average) ---
    if "DPT" in eidx:
        hourly_ea = _ea_from_dpt(stack[:, eidx["DPT"], :])
        daily["ea" + sfx] = np.nanmean(hourly_ea, axis=0)

    # --- wind direction: vector-average via mean U, V ---
    if "UGRD" in eidx and "VGRD" in eidx:
        mu = np.nanmean(stack[:, eidx["UGRD"], :], axis=0)
        mv = np.nanmean(stack[:, eidx["VGRD"], :], axis=0)
        daily["wdir" + sfx] = np.degrees(np.arctan2(-mu, -mv)) % 360.0

    daily["n_hours"] = np.full(n_stations, len(hourly), dtype=np.int16)
    return daily


# ----------------------------------------------------------- multiprocessing


def _init_worker(
    grib_root: str,
    model: str,
    rows_adj: np.ndarray,
    cols_adj: np.ndarray,
    valid: np.ndarray,
    window: Window,
    n_stations: int,
) -> None:
    """Set per-worker globals (shared via fork COW on Linux)."""
    global _W_GRIB_ROOT, _W_MODEL, _W_ROWS_ADJ, _W_COLS_ADJ
    global _W_VALID, _W_WINDOW, _W_N_STATIONS
    _W_GRIB_ROOT = grib_root
    _W_MODEL = model
    _W_ROWS_ADJ = rows_adj
    _W_COLS_ADJ = cols_adj
    _W_VALID = valid
    _W_WINDOW = window
    _W_N_STATIONS = n_stations


def _worker_process_day(
    day: datetime,
) -> tuple[datetime, dict[str, np.ndarray] | None]:
    """Thin wrapper that reads from worker globals."""
    daily = _process_day(
        _W_GRIB_ROOT,
        _W_MODEL,
        day,
        _W_ROWS_ADJ,
        _W_COLS_ADJ,
        _W_VALID,
        _W_WINDOW,
        _W_N_STATIONS,
    )
    return day, daily


# ----------------------------------------------------------- main entry point


def build_station_daily(
    model: str,
    grib_root: str,
    stations_csv: str,
    out_dir: str,
    start: datetime,
    end: datetime,
    bounds: tuple[float, float, float, float] | None = None,
    overwrite: bool = False,
    workers: int = 1,
) -> int:
    """Build per-station daily Parquets from hourly GRIB2 archives."""
    stations = load_station_csv(stations_csv, bounds)
    n = len(stations)
    print(f"Stations loaded: {n}", flush=True)

    os.makedirs(out_dir, exist_ok=True)

    # --- reference GRIB for CRS + band map ---
    ref_path = None
    d = start
    while d <= end:
        for h in range(24):
            ref_path = _find_grib(grib_root, model, d.strftime("%Y%m%d"), h)
            if ref_path:
                break
        if ref_path:
            break
        d += timedelta(days=1)
    if not ref_path:
        raise FileNotFoundError(f"No GRIBs found for {model} in [{start}, {end}]")

    band_map = _build_band_map(ref_path)
    print(f"Band map (reference): {band_map}", flush=True)

    rows, cols, valid, grid_h, grid_w = project_to_lcc_pixels(
        ref_path, stations["longitude"].values, stations["latitude"].values
    )
    n_valid = int(valid.sum())
    print(f"Stations in grid: {n_valid}/{n}", flush=True)
    if n_valid == 0:
        raise RuntimeError("No stations fall inside the GRIB grid.")

    # --- read window covering all valid stations (+ buffer) ---
    vr, vc = rows[valid], cols[valid]
    buf = 2
    r0 = max(0, int(vr.min()) - buf)
    c0 = max(0, int(vc.min()) - buf)
    r1 = min(grid_h, int(vr.max()) + buf + 1)
    c1 = min(grid_w, int(vc.max()) + buf + 1)
    window = Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0)
    pct = window.height * window.width / (grid_h * grid_w) * 100
    print(
        f"Read window: rows [{r0}:{r1}], cols [{c0}:{c1}]  "
        f"({window.height}x{window.width}, {pct:.1f}% of grid)",
        flush=True,
    )

    # Adjust pixel coords to window-relative
    rows_adj = rows - r0
    cols_adj = cols - c0

    # --- process days ---
    fids = stations["fid"].values
    valid_idx = np.where(valid)[0]
    valid_fids = fids[valid_idx]

    n_days = (end - start).days + 1
    all_days = [start + timedelta(days=i) for i in range(n_days)]
    days_list: list[pd.Timestamp] = []
    day_arrays: list[dict[str, np.ndarray]] = []

    workers = max(1, workers)
    print(f"Processing {n_days} days with {workers} worker(s)...", flush=True)

    if workers == 1:
        # Serial path — simpler, no fork overhead.
        for di, day in enumerate(all_days):
            daily = _process_day(
                grib_root, model, day, rows_adj, cols_adj, valid, window, n
            )
            if daily is not None:
                days_list.append(pd.Timestamp(day))
                day_arrays.append({k: v[valid_idx] for k, v in daily.items()})
            if (di + 1) % 10 == 0 or di == n_days - 1:
                print(f"  {di + 1}/{n_days} days ({day:%Y-%m-%d})", flush=True)
    else:
        # Parallel path — overlap NAS I/O across workers.
        done = 0
        with mp.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(grib_root, model, rows_adj, cols_adj, valid, window, n),
        ) as pool:
            for day, daily in pool.imap_unordered(
                _worker_process_day, all_days, chunksize=4
            ):
                done += 1
                if daily is not None:
                    days_list.append(pd.Timestamp(day))
                    day_arrays.append({k: v[valid_idx] for k, v in daily.items()})
                if done % 10 == 0 or done == n_days:
                    print(f"  {done}/{n_days} days", flush=True)

    if not day_arrays:
        raise RuntimeError("No valid days processed.")

    # --- build big DataFrame, split by station, write ---
    # Sort by day since imap_unordered may have shuffled order.
    order = sorted(range(len(days_list)), key=lambda i: days_list[i])
    days_list = [days_list[i] for i in order]
    day_arrays = [day_arrays[i] for i in order]

    columns = list(day_arrays[0].keys())
    n_actual = len(days_list)
    nv = len(valid_fids)

    big: dict[str, np.ndarray] = {
        c: np.concatenate([d[c] for d in day_arrays]) for c in columns
    }
    big["fid"] = np.tile(valid_fids, n_actual)
    big["day"] = np.repeat(days_list, nv)

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


# ------------------------------------------------------------------ CLI


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hourly GRIB2 -> per-station daily Parquets."
    )
    p.add_argument("--model", required=True, choices=["urma", "rtma"])
    p.add_argument("--grib-root", required=True, help="Root of hourly GRIB archive.")
    p.add_argument("--stations-csv", required=True, help="Station inventory CSV.")
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
        "--overwrite", action="store_true", help="Overwrite existing output files."
    )
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    build_station_daily(
        model=a.model,
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
