"""
Extract NOAA CDR Surface Reflectance at station locations.

Reads daily CDR netCDF files (global 0.05° grid) and extracts 5 features
at ~9,000 PNW station lat/lons.  Writes per-station daily Parquets with
the same layout as RTMA/URMA station daily files.

Features extracted:
  i1_cdr        – 640 nm BRDF-corrected reflectance (×0.0001)
  i2_cdr        – 860 nm BRDF-corrected reflectance (×0.0001)
  bt15_cdr      – 10.76 µm brightness temperature (×0.1 → K)
  cloud_state_cdr – QA bits 0–1 ordinal (0=clear … 3=confident cloudy)
  szen_cdr      – solar zenith angle (×0.01 → degrees)

Output: one Parquet per station in --out-dir, indexed by date.
Supports resume (skips dates already extracted) and multiprocessing.
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

try:
    import netCDF4
except ImportError:
    netCDF4 = None


# CDR variable names → (output column, scale factor)
# netCDF4 auto-applies scale_factor on read, so no manual scaling needed.
_VARS = {
    "BRDF_corrected_I1_SurfRefl_CMG": "i1_cdr",
    "BRDF_corrected_I2_SurfRefl_CMG": "i2_cdr",
    "BT_CH15": "bt15_cdr",
    "SZEN": "szen_cdr",
}
_COL_NAMES = list(_VARS.values())
_QA_VAR = "QA"
_QA_COL = "cloud_state_cdr"
_ALL_COLS = _COL_NAMES + [_QA_COL]
_FILL = -9999


def _parse_date_from_filename(path: str) -> pd.Timestamp | None:
    """Extract date from CDR filename like ...YYYYMMDD..."""
    base = os.path.basename(path)
    for part in base.split("_"):
        if len(part) == 8 and part.isdigit():
            try:
                return pd.Timestamp(part)
            except ValueError:
                continue
    return None


def _compute_pixel_indices(
    lats: np.ndarray, lons: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert lat/lon arrays to CDR grid row/col indices.

    CDR grid: 3600 rows × 7200 cols, 0.05° resolution.
    Row 0 = 90°N, Col 0 = 180°W.
    """
    lat_idx = ((90.0 - lats) / 0.05).astype(np.intp)
    lon_idx = ((lons + 180.0) / 0.05).astype(np.intp)
    lat_idx = np.clip(lat_idx, 0, 3599)
    lon_idx = np.clip(lon_idx, 0, 7199)
    return lat_idx, lon_idx


def _process_one_nc(args: tuple) -> tuple[str, np.ndarray] | None:
    """Process a single NC file. Returns (date_str, data_array (n_stations, 5)) or None."""
    nc_path, lat_idx, lon_idx, n_stations = args

    dt = _parse_date_from_filename(nc_path)
    if dt is None:
        return None

    try:
        ds = netCDF4.Dataset(nc_path, "r")
    except Exception:
        return None

    # Output: (n_stations, 5) — 4 numeric vars + cloud_state
    out = np.full((n_stations, 5), np.nan, dtype="float32")

    for col_idx, (var_name, _col_name) in enumerate(_VARS.items()):
        if var_name not in ds.variables:
            continue
        var = ds.variables[var_name]
        slab = np.asarray(var[0], dtype="float32").squeeze()
        raw = slab[lat_idx, lon_idx]
        # netCDF4 auto-applies scale_factor; only mask fill/out-of-range
        mask = np.isnan(raw)
        if hasattr(var, "valid_range"):
            vr = var.valid_range
            # valid_range is in packed (pre-scale) units; compare scaled values
            lo = float(vr[0]) * float(getattr(var, "scale_factor", 1.0))
            hi = float(vr[1]) * float(getattr(var, "scale_factor", 1.0))
            mask |= (raw < lo) | (raw > hi)
        raw[mask] = np.nan
        out[:, col_idx] = raw

    if _QA_VAR in ds.variables:
        qa_slab = np.asarray(ds.variables[_QA_VAR][0], dtype="int32").squeeze()
        qa_arr = qa_slab[lat_idx, lon_idx]
        cs = (qa_arr & 0x3).astype("float32")
        cs[qa_arr == _FILL] = np.nan
        out[:, 4] = cs

    ds.close()
    return dt.strftime("%Y-%m-%d"), out


def extract_cdr_stations(
    nc_dir: str,
    out_dir: str,
    stations_csv: str,
    start_date: str = "2018-01-01",
    end_date: str = "2024-12-31",
    num_workers: int = 4,
    flush_every: int = 200,
) -> None:
    if netCDF4 is None:
        raise ImportError("netCDF4 is required: uv add netCDF4")

    os.makedirs(out_dir, exist_ok=True)

    # Load stations
    stations = pd.read_csv(stations_csv)
    id_col = "fid" if "fid" in stations.columns else "station_id"
    fids = stations[id_col].astype(str).values
    lats = stations["latitude"].values.astype("float64")
    lons = stations["longitude"].values.astype("float64")
    n_stations = len(fids)

    lat_idx, lon_idx = _compute_pixel_indices(lats, lons)
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

    # Find all dates already fully extracted (present in ALL stations)
    if existing_dates:
        all_sets = list(existing_dates.values())
        done_dates = set.intersection(*all_sets) if all_sets else set()
    else:
        done_dates = set()
    print(f"Already extracted dates: {len(done_dates)}", flush=True)

    # Filter NC files to date range, skip already-done dates
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    nc_files = []
    for f in sorted(os.listdir(nc_dir)):
        if not f.endswith(".nc"):
            continue
        dt = _parse_date_from_filename(f)
        if dt is None or dt < start or dt > end:
            continue
        if dt.strftime("%Y-%m-%d") in done_dates:
            continue
        nc_files.append(os.path.join(nc_dir, f))

    print(f"NC files to process: {len(nc_files)}", flush=True)
    if not nc_files:
        print("Nothing to do.", flush=True)
        return

    # Per-station accumulators: fid_idx -> list of (date_str, row_array)
    accum: dict[int, list[tuple[str, np.ndarray]]] = {i: [] for i in range(n_stations)}

    t0 = time.time()
    processed = 0

    # Build worker args
    worker_args = [(p, lat_idx, lon_idx, n_stations) for p in nc_files]

    def _flush():
        """Write accumulated rows to per-station parquets (append-merge)."""
        for i in range(n_stations):
            if not accum[i]:
                continue
            fid = fids[i]
            dates = [r[0] for r in accum[i]]
            vals = np.stack([r[1] for r in accum[i]])
            new_df = pd.DataFrame(vals, index=pd.to_datetime(dates), columns=_ALL_COLS)
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

    with Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(_process_one_nc, worker_args, chunksize=4):
            if result is None:
                continue
            date_str, data = result

            # Distribute to per-station accumulators
            for i in range(n_stations):
                accum[i].append((date_str, data[i]))

            processed += 1
            if processed % flush_every == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed
                eta = (len(nc_files) - processed) / rate if rate > 0 else 0
                print(
                    f"  [{processed}/{len(nc_files)}]  "
                    f"{rate:.1f} files/s  ETA {eta / 60:.1f} min  (flushing...)",
                    flush=True,
                )
                _flush()

    # Final flush
    _flush()

    elapsed = time.time() - t0
    print(
        f"Done: {processed} days → {out_dir}  ({elapsed / 60:.1f} min)",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract CDR at station locations.")
    p.add_argument("--nc-dir", required=True, help="Directory of CDR daily .nc files.")
    p.add_argument(
        "--out-dir", required=True, help="Output dir for per-station Parquets."
    )
    p.add_argument("--stations-csv", required=True, help="Station inventory CSV.")
    p.add_argument("--start-date", default="2018-01-01")
    p.add_argument("--end-date", default="2024-12-31")
    p.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    extract_cdr_stations(
        nc_dir=a.nc_dir,
        out_dir=a.out_dir,
        stations_csv=a.stations_csv,
        start_date=a.start_date,
        end_date=a.end_date,
        num_workers=a.workers,
    )
