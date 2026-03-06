"""Extract QC'd daily parquets from raw MADIS mesonet netCDF archives.

Single-pass pipeline: raw gzip netCDF → three-layer QC → daily snappy parquet.

QC layers
---------
1. DD flag filter  — accept V/S/C/G only
2. QCR bitmask     — reject temporal/spatial/buddy failures
3. Physical bounds — agweather-qaqc tighter limits
"""

from __future__ import annotations

import argparse
import gzip
import os
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=xr.SerializationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")

# ---------------------------------------------------------------------------
# QC constants
# ---------------------------------------------------------------------------

ACCEPTABLE_DD = {"V", "S", "C", "G"}

# Bits: master(1) | validity(2) | temporal(16) | stat_spatial(32) | buddy(64)
# Excludes bit 4 (internal consistency = 8) — Td>T handled separately.
QCR_REJECT_BITS = 0b01110011  # = 115

BOUNDS = {
    "temperature": (223.15, 333.15),  # -50 to 60 °C in K
    "dewpoint": (205.15, 305.15),  # -68 to 32 °C in K
    "relHumidity": (2.0, 110.0),  # %
    "windSpeed": (0.0, 35.0),  # m/s
    "windDir": (0.0, 360.0),  # degrees
}

# Variables to extract from each netCDF file
ID_VARS = [
    "stationId",
    "latitude",
    "longitude",
    "elevation",
    "dataProvider",
    "observationTime",
]
MET_VARS = [
    "temperature",
    "dewpoint",
    "relHumidity",
    "windSpeed",
    "windDir",
    "precipAccum",
    "solarRadiation",
]
DD_VARS = [
    f"{v}DD" for v in ["temperature", "dewpoint", "relHumidity", "windSpeed", "windDir"]
]
QCR_VARS = [
    f"{v}QCR"
    for v in ["temperature", "dewpoint", "relHumidity", "windSpeed", "windDir"]
]


# ---------------------------------------------------------------------------
# netCDF I/O
# ---------------------------------------------------------------------------


def open_nc(f: str) -> xr.Dataset | None:
    """Gzip-open a MADIS netCDF; scipy first, netcdf4 fallback."""
    temp_nc_file = None
    try:
        with gzip.open(f) as fp:
            ds = xr.open_dataset(fp, engine="scipy", cache=False)
    except OverflowError as oe:
        print(f"  OverflowError in {f}: {oe}")
        return None
    except Exception:
        try:
            fd, temp_nc_file = tempfile.mkstemp(suffix=".nc")
            os.close(fd)
            with gzip.open(f, "rb") as f_in, open(temp_nc_file, "wb") as f_out:
                f_out.write(f_in.read())
            ds = xr.open_dataset(temp_nc_file, engine="netcdf4")
        except Exception as e2:
            print(f"  netcdf4 fallback failed for {f}: {e2}")
            return None
        finally:
            if temp_nc_file and os.path.exists(temp_nc_file):
                os.remove(temp_nc_file)
    return ds


# ---------------------------------------------------------------------------
# Extraction + QC
# ---------------------------------------------------------------------------


def _extract_hourly(
    ds: xr.Dataset, bounds: tuple[float, float, float, float]
) -> pd.DataFrame | None:
    """Extract one hourly netCDF dataset into a DataFrame with ID, met, DD, and QCR columns."""
    if "recNum" not in ds.sizes or ds.sizes["recNum"] == 0:
        return None

    n = ds.sizes["recNum"]
    data: dict[str, np.ndarray | list] = {}

    for var in ID_VARS + MET_VARS + DD_VARS + QCR_VARS:
        if var not in ds:
            if var in MET_VARS:
                data[var] = np.full(n, np.nan)
            elif var in DD_VARS or var in QCR_VARS:
                data[var] = np.full(n, np.nan)
            else:
                data[var] = [""] * n
            continue

        if var == "observationTime":
            data[var] = pd.to_datetime(ds[var].values, errors="coerce")
        elif var in ("stationId", "dataProvider") or var in DD_VARS:
            raw = ds[var].values
            if raw.dtype.kind in ("S", "U", "O"):
                data[var] = [
                    v.decode().strip() if isinstance(v, bytes) else str(v).strip()
                    for v in raw
                ]
            else:
                data[var] = [str(v).strip() for v in raw]
        elif var in QCR_VARS:
            data[var] = ds[var].values.astype(np.float64)
        else:
            data[var] = ds[var].values.astype(np.float64)

    df = pd.DataFrame(data)

    # Spatial filter
    w, s, e, n_lat = bounds
    mask = (
        (df["latitude"] >= s)
        & (df["latitude"] < n_lat)
        & (df["longitude"] >= w)
        & (df["longitude"] < e)
    )
    df = df.loc[mask]

    return df if len(df) > 0 else None


def _apply_qc(df: pd.DataFrame, qcr_mask: int = QCR_REJECT_BITS) -> pd.DataFrame:
    """Three-layer QC: DD flags, QCR bitmask, physical bounds.

    NaNs individual variable values on failure; sets per-row ``qc_passed`` bool
    (True if at least one met var survived all three layers).
    """
    met_with_dd = ["temperature", "dewpoint", "relHumidity", "windSpeed", "windDir"]

    for var in met_with_dd:
        dd_col = f"{var}DD"
        qcr_col = f"{var}QCR"

        # Layer 1: DD flag filter
        if dd_col in df.columns:
            bad_dd = ~df[dd_col].isin(ACCEPTABLE_DD)
            df.loc[bad_dd, var] = np.nan

        # Layer 2: QCR bitmask filter
        if qcr_col in df.columns:
            qcr = pd.to_numeric(df[qcr_col], errors="coerce").fillna(0).astype(np.int64)
            bad_qcr = (qcr & qcr_mask) > 0
            df.loc[bad_qcr, var] = np.nan

        # Layer 3: physical bounds
        if var in BOUNDS:
            lo, hi = BOUNDS[var]
            val = pd.to_numeric(df[var], errors="coerce")
            oob = (val < lo) | (val > hi)
            df.loc[oob, var] = np.nan

    # qc_passed = at least one of the core met vars is finite
    core = ["temperature", "dewpoint", "relHumidity", "windSpeed", "windDir"]
    df["qc_passed"] = df[core].notna().any(axis=1)

    return df


# ---------------------------------------------------------------------------
# Day-level worker
# ---------------------------------------------------------------------------


def _process_day(
    day_str: str,
    src: str,
    dst: str,
    bounds: tuple[float, float, float, float],
    qcr_mask: int = QCR_REJECT_BITS,
) -> tuple[str, int | None, str | None]:
    """Process 24 hourly files for one day → 1 daily parquet.

    Returns (day_str, row_count_or_None, error_msg_or_None).
    """
    out_path = os.path.join(dst, f"{day_str}.parquet")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return day_str, None, None  # already done

    hours = [f"{day_str}_{h:02d}00.gz" for h in range(24)]
    frames = []

    for fname in hours:
        fpath = os.path.join(src, fname)
        if not os.path.exists(fpath):
            continue
        ds = open_nc(fpath)
        if ds is None:
            continue
        hdf = _extract_hourly(ds, bounds)
        ds.close()
        if hdf is not None:
            frames.append(hdf)

    if not frames:
        return day_str, 0, "no data"

    df = pd.concat(frames, ignore_index=True)
    df = _apply_qc(df, qcr_mask=qcr_mask)
    df = df.sort_values("stationId").reset_index(drop=True)

    os.makedirs(dst, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="snappy")

    return day_str, len(df), None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_date_strings(start: str, end: str) -> list[str]:
    """Inclusive date range → list of YYYYMMDD strings."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    out = []
    cur = s
    while cur <= e:
        out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract QC'd daily parquets from MADIS mesonet netCDF."
    )
    p.add_argument("--src", required=True, help="Root dir with YYYYMMDD_HHMM.gz files")
    p.add_argument("--dst", required=True, help="Output dir for daily parquets")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    p.add_argument(
        "--workers", type=int, default=20, help="Parallel workers (default 20)"
    )
    p.add_argument(
        "--bounds",
        default="-125,24,-66,53",
        help="west,south,east,north (default CONUS: -125,24,-66,53)",
    )
    p.add_argument(
        "--qcr-mask",
        type=int,
        default=QCR_REJECT_BITS,
        help=f"QCR reject bitmask (default {QCR_REJECT_BITS}; use 2 for validity-only)",
    )
    a = p.parse_args()

    bounds = tuple(float(x) for x in a.bounds.split(","))
    assert len(bounds) == 4, f"bounds must have 4 values, got {len(bounds)}"

    days = _generate_date_strings(a.start, a.end)
    print(f"MADIS daily extraction: {len(days)} days, {a.workers} workers")
    print(f"  src: {a.src}")
    print(f"  dst: {a.dst}")
    print(f"  bounds: {bounds}")
    print(f"  qcr_mask: {a.qcr_mask} (0b{a.qcr_mask:08b})")

    os.makedirs(a.dst, exist_ok=True)

    done = 0
    errors = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {
            pool.submit(_process_day, d, a.src, a.dst, bounds, a.qcr_mask): d
            for d in days
        }

        for fut in as_completed(futures):
            day_str = futures[fut]
            try:
                _, nrows, err = fut.result()
            except Exception as exc:
                print(f"  EXCEPTION {day_str}: {exc}")
                errors += 1
                done += 1
                continue

            if nrows is None:
                skipped += 1
            elif err:
                errors += 1
            done += 1

            if done % 50 == 0:
                print(
                    f"  progress: {done}/{len(days)} done, {skipped} skipped, {errors} errors"
                )

    print(
        f"Finished: {done} days processed, {skipped} skipped (exist), {errors} errors"
    )


if __name__ == "__main__":
    main()
