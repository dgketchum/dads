"""
Download HRRR skin temperature at wind stations and compute MOST factors.

Three modes:

  --z0-only         Fetch SFCR:surface from one recent HRRR file, sample at
                    824 accepted wind stations, write z0 JSON artifact.

  (default)         For each day in HRRR POR, fetch 24 hourly TMP:surface
                    messages via byte-range, decode in-memory with eccodes,
                    sample at station grid indices, write daily parquets.

  --compute-factors Read daily HRRR T_skin parquets + RTMA station dailies +
                    z0 + crosswalk, compute daily-mean MOST correction factors.

Usage
-----
  uv run python -m grid.sources.download_hrrr_stability --z0-only
  uv run python -m grid.sources.download_hrrr_stability --workers 12
  uv run python -m grid.sources.download_hrrr_stability --compute-factors
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import eccodes
import numpy as np
import pandas as pd

from grid.sources.hrrr_common import (
    ARCHIVE_START,
    fetch_byte_range,
    fetch_idx,
    grib_url,
    parse_idx_for_var,
)
from grid.sources.hrrr_stability import daily_mean_most_factor
from grid.station_extract import build_haversine_tree, query_nearest

ACCEPTED_FIDS_PATH = "artifacts/rtma_wind_accepted_fids.json"
CROSSWALK_PATH = "artifacts/synoptic_wind_height_crosswalk.csv"
Z0_ARTIFACT = "artifacts/hrrr_z0_at_wind_stations.json"
STATIONS_CSV = "artifacts/madis_pnw.csv"
OUT_DIR = "/nas/dads/mvp/hrrr_stability"
RTMA_DAILY_DIR = "/nas/dads/mvp/rtma_daily_pnw_2018_2024"

# ── signal handling ──────────────────────────────────────────────────────────

_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n  Shutdown requested — finishing current batch ...")
    sys.stdout.flush()


# ── GRIB decode with eccodes ─────────────────────────────────────────────────


def _decode_grib_message(
    data: bytes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Decode a single GRIB message from bytes.  Returns (values, lats, lons) or None."""
    msgid = eccodes.codes_new_from_message(data)
    try:
        values = eccodes.codes_get_array(msgid, "values")
        lats = eccodes.codes_get_array(msgid, "latitudes")
        lons = eccodes.codes_get_array(msgid, "longitudes")
        return values.astype(np.float32), lats, lons
    finally:
        eccodes.codes_release(msgid)


# ── station loading ──────────────────────────────────────────────────────────


def _load_accepted_stations() -> pd.DataFrame:
    """Load accepted wind station locations.  Returns DataFrame with fid, lat, lon."""
    with open(ACCEPTED_FIDS_PATH) as f:
        accepted = set(json.load(f))

    stations = pd.read_csv(STATIONS_CSV)
    id_col = "station_id" if "station_id" in stations.columns else "fid"
    stations[id_col] = stations[id_col].astype(str)
    stations = stations[stations[id_col].isin(accepted)].copy()
    stations = stations.rename(columns={id_col: "fid"})
    return stations[["fid", "latitude", "longitude"]].reset_index(drop=True)


# ── Mode 1: z0 sample ───────────────────────────────────────────────────────


def _run_z0_sample() -> None:
    """Fetch SFCR:surface from one HRRR file, sample at wind stations."""
    stations = _load_accepted_stations()
    print(f"Accepted wind stations: {len(stations)}")

    # Use a recent date for z0 (static field)
    d = date(2024, 1, 1)
    hour = 0

    print(f"Fetching IDX for {d} {hour:02d}Z ...")
    idx_text = fetch_idx(d, hour)
    if idx_text is None:
        print("ERROR: could not fetch IDX file")
        sys.exit(1)

    rng = parse_idx_for_var(idx_text, "SFCR", "surface")
    if rng is None:
        print("ERROR: SFCR:surface not found in IDX")
        sys.exit(1)

    print(f"Fetching SFCR byte range {rng[0]}-{rng[1]} ...")
    data = fetch_byte_range(grib_url(d, hour), rng[0], rng[1])
    if data is None:
        print("ERROR: could not fetch SFCR data")
        sys.exit(1)

    result = _decode_grib_message(data)
    if result is None:
        print("ERROR: could not decode SFCR GRIB message")
        sys.exit(1)

    values, lats, lons = result
    print(
        f"GRIB grid: {len(values)} points, lon range [{lons.min():.1f}, {lons.max():.1f}]"
    )

    tree = build_haversine_tree(lats, lons)

    sta_lons_360 = stations["longitude"].values % 360.0
    indices = query_nearest(tree, stations["latitude"].values, sta_lons_360)

    z0_dict = {}
    for i, fid in enumerate(stations["fid"]):
        z0_dict[fid] = round(float(values[indices[i]]), 4)

    os.makedirs(os.path.dirname(Z0_ARTIFACT) or ".", exist_ok=True)
    with open(Z0_ARTIFACT, "w") as f:
        json.dump(z0_dict, f, indent=2)
    print(f"Written: {Z0_ARTIFACT} ({len(z0_dict)} stations)")

    # Summary stats
    z0_vals = np.array(list(z0_dict.values()))
    print(
        f"z0 stats: min={z0_vals.min():.4f}  median={np.median(z0_vals):.4f}  "
        f"max={z0_vals.max():.4f}  mean={z0_vals.mean():.4f}"
    )


# ── Mode 2: download hourly T_skin ──────────────────────────────────────────


def _fetch_hour_tskin(
    d: date,
    hour: int,
    url: str,
    station_indices: np.ndarray,
    fids: list[str],
    tree_ref: list,
) -> list[dict] | None:
    """Fetch TMP:surface for one hour, sample at stations.

    Returns list of dicts or None on failure.
    tree_ref is a mutable list holding [tree, lats, lons] — built on first call.
    """
    idx_text = fetch_idx(d, hour)
    if idx_text is None:
        return None

    rng = parse_idx_for_var(idx_text, "TMP", "surface")
    if rng is None:
        return None

    data = fetch_byte_range(url, rng[0], rng[1])
    if data is None:
        return None

    result = _decode_grib_message(data)
    if result is None:
        return None

    values, lats, lons = result

    # Build tree on first call
    if tree_ref[0] is None:
        tree_ref[0] = build_haversine_tree(lats, lons)
        tree_ref[1] = lats
        tree_ref[2] = lons

    rows = []
    for i, fid in enumerate(fids):
        rows.append(
            {
                "fid": fid,
                "hour": hour,
                "t_skin_k": float(values[station_indices[i]]),
            }
        )
    return rows


def _run_download(args: argparse.Namespace) -> None:
    """Download hourly TMP:surface for all days, write daily parquets."""
    stations = _load_accepted_stations()
    fids = stations["fid"].tolist()
    sta_lons_360 = stations["longitude"].values % 360.0
    sta_lats = stations["latitude"].values

    yesterday = date.today() - timedelta(days=1)
    start = date.fromisoformat(args.start_date) if args.start_date else ARCHIVE_START
    end = date.fromisoformat(args.end_date) if args.end_date else yesterday

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "manifest.parquet")
    manifest = _load_manifest(manifest_path)
    # Deduplicate manifest: keep latest entry per date
    if not manifest.empty:
        manifest = manifest.sort_values("processed_at").drop_duplicates(
            "date", keep="last"
        )
    done_dates = set(manifest[manifest["status"] == "done"]["date"].values)

    # Build day list
    days = []
    d = start
    while d <= end:
        if d.isoformat() not in done_dates:
            days.append(d)
        d += timedelta(days=1)

    print(f"Stations: {len(fids)}")
    print(f"Date range: {start} → {end}")
    print(f"Days to process: {len(days)} (skipping {len(done_dates)} already done)")
    sys.stdout.flush()

    # BallTree built lazily from first GRIB
    tree_ref: list = [None, None, None]
    station_indices: np.ndarray | None = None

    new_manifest_rows: list[dict] = []
    t_start = time.time()
    days_done = 0

    for day_i, day in enumerate(days):
        if _shutdown:
            break

        day_str = day.isoformat()
        out_path = os.path.join(OUT_DIR, f"{day:%Y%m%d}.parquet")

        # Fetch all 24 hours in parallel
        all_hour_rows: list[dict] = []
        n_hours_ok = 0

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for h in range(24):
                url = grib_url(day, h)
                fut = pool.submit(
                    _fetch_hour_tskin,
                    day,
                    h,
                    url,
                    station_indices
                    if station_indices is not None
                    else np.arange(len(fids)),
                    fids,
                    tree_ref,
                )
                futures[fut] = h

            for fut in as_completed(futures):
                rows = fut.result()
                if rows is not None:
                    all_hour_rows.extend(rows)
                    n_hours_ok += 1

        # After first day, compute station indices from the BallTree
        if station_indices is None and tree_ref[0] is not None:
            station_indices = query_nearest(tree_ref[0], sta_lats, sta_lons_360)
            # Re-fetch this day's data with correct indices if we used dummy ones
            all_hour_rows = []
            n_hours_ok = 0
            for h in range(24):
                rows = _fetch_hour_tskin(
                    day,
                    h,
                    grib_url(day, h),
                    station_indices,
                    fids,
                    tree_ref,
                )
                if rows is not None:
                    all_hour_rows.extend(rows)
                    n_hours_ok += 1

        if n_hours_ok == 0:
            status = "missing"
        elif n_hours_ok == 24:
            status = "done"
        else:
            status = "partial"

        if n_hours_ok > 0:
            df = pd.DataFrame(all_hour_rows)
            df["hour"] = df["hour"].astype("int8")
            df["t_skin_k"] = df["t_skin_k"].astype("float32")
            df.to_parquet(out_path, index=False)

        new_manifest_rows.append(
            {
                "date": day_str,
                "n_hours": n_hours_ok,
                "status": status,
                "processed_at": datetime.now().isoformat(),
            }
        )

        days_done += 1
        elapsed = time.time() - t_start
        rate = days_done / elapsed * 3600 if elapsed > 0 else 0
        print(
            f"  {day}  {n_hours_ok}/24 hours  "
            f"[{days_done}/{len(days)} days, {rate:.0f} days/hr, "
            f"{elapsed / 3600:.1f}h elapsed]"
        )
        sys.stdout.flush()

        # Flush manifest every 10 days
        if (day_i + 1) % 10 == 0 and new_manifest_rows:
            manifest = pd.concat(
                [manifest, pd.DataFrame(new_manifest_rows)], ignore_index=True
            )
            new_manifest_rows.clear()
            _save_manifest(manifest, manifest_path)

    # Final manifest flush
    if new_manifest_rows:
        manifest = pd.concat(
            [manifest, pd.DataFrame(new_manifest_rows)], ignore_index=True
        )
    _save_manifest(manifest, manifest_path)

    elapsed_h = (time.time() - t_start) / 3600
    print(f"\nDone: {days_done} days in {elapsed_h:.1f}h")
    if _shutdown:
        print("  (interrupted — resume by re-running)")


# ── Mode 3: compute MOST factors ────────────────────────────────────────────


def _read_station_daily(path: str) -> pd.DataFrame:
    """Read a per-station RTMA daily parquet, normalising the index to naive dates."""
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        for c in ("time", "date", "day", "dt"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
                df = df.set_index(c)
                break
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].copy()
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    return df


def _run_compute_factors() -> None:
    """Read daily HRRR T_skin + RTMA dailies + z0, compute MOST factors.

    Outer loop over HRRR daily files (each read once), inner loop over
    stations within each file.  RTMA dailies are pre-loaded into a dict
    so the total I/O is O(stations + days) instead of O(stations × days).
    """
    # Load z0
    if not os.path.exists(Z0_ARTIFACT):
        print(f"ERROR: z0 artifact not found: {Z0_ARTIFACT}")
        print("Run with --z0-only first.")
        sys.exit(1)
    with open(Z0_ARTIFACT) as f:
        z0_dict = json.load(f)
    print(f"z0 values: {len(z0_dict)} stations")

    # Load wind height crosswalk
    wind_ht: dict[str, float] = {}
    if os.path.exists(CROSSWALK_PATH):
        xw = pd.read_csv(CROSSWALK_PATH, usecols=["stationId", "wind_sensor_ht"])
        wind_ht = dict(zip(xw["stationId"], xw["wind_sensor_ht"]))
    print(f"Wind height crosswalk: {len(wind_ht)} stations")

    # Load accepted fids
    with open(ACCEPTED_FIDS_PATH) as f:
        accepted_fids = json.load(f)

    # Pre-load RTMA dailies for all accepted stations (one read per station)
    print("Loading RTMA dailies ...")
    rtma_cache: dict[str, pd.DataFrame] = {}
    for fid in accepted_fids:
        if z0_dict.get(fid) is None:
            continue
        rtma_path = os.path.join(RTMA_DAILY_DIR, f"{fid}.parquet")
        if not os.path.exists(rtma_path):
            continue
        rtma = _read_station_daily(rtma_path)
        if "tmp_rtma" in rtma.columns and "wind_rtma" in rtma.columns:
            rtma_cache[fid] = rtma
    print(f"RTMA dailies loaded: {len(rtma_cache)} stations")

    # List daily HRRR parquets
    hrrr_files = sorted(
        f
        for f in os.listdir(OUT_DIR)
        if f.endswith(".parquet")
        and f != "manifest.parquet"
        and f != "most_factors.parquet"
    )
    print(f"HRRR daily files: {len(hrrr_files)}")

    all_rows: list[dict] = []
    n_fallback = 0

    for file_i, hrrr_file in enumerate(hrrr_files):
        day_str = hrrr_file.replace(".parquet", "")
        day = pd.Timestamp(f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:8]}")

        # Read the entire daily HRRR parquet once
        hrrr_path = os.path.join(OUT_DIR, hrrr_file)
        hrrr_df = pd.read_parquet(hrrr_path)
        hrrr_df["fid"] = hrrr_df["fid"].astype(str)

        # Pivot to fid → hour → t_skin_k for fast lookup
        for fid, grp in hrrr_df.groupby("fid"):
            if fid not in rtma_cache:
                continue
            rtma = rtma_cache[fid]
            if day not in rtma.index:
                continue

            t_2m_c = rtma.loc[day, "tmp_rtma"]
            u_10m = rtma.loc[day, "wind_rtma"]
            if pd.isna(t_2m_c) or pd.isna(u_10m):
                continue

            hourly_tskin = np.full(24, np.nan)
            for h, t in zip(grp["hour"].values, grp["t_skin_k"].values):
                hourly_tskin[int(h)] = t

            factor, n_valid = daily_mean_most_factor(
                hourly_tskin,
                float(t_2m_c) + 273.15,
                float(u_10m),
                z0_dict[fid],
                wind_ht.get(fid, 10.0),
            )
            if n_valid < 4:
                n_fallback += 1

            all_rows.append(
                {
                    "fid": fid,
                    "day": day,
                    "most_factor": np.float32(factor),
                    "n_valid_hours": np.int8(n_valid),
                }
            )

        if (file_i + 1) % 500 == 0:
            print(f"  {file_i + 1}/{len(hrrr_files)} days, {len(all_rows)} factor rows")
            sys.stdout.flush()

    print(f"Processed {len(hrrr_files)} days, {len(all_rows)} total factor rows")
    print(f"Stations with RTMA data: {len(rtma_cache)}")
    print(f"Fallback to FAO-56 neutral: {n_fallback} station-days")

    if all_rows:
        out = pd.DataFrame(all_rows)
        out_path = os.path.join(OUT_DIR, "most_factors.parquet")
        out.to_parquet(out_path, index=False)
        print(f"Written: {out_path}")


# ── manifest helpers ─────────────────────────────────────────────────────────


def _load_manifest(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["date", "n_hours", "status", "processed_at"])


def _save_manifest(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HRRR skin temp download + MOST factors.")
    p.add_argument(
        "--z0-only", action="store_true", help="Sample z0 from one HRRR file."
    )
    p.add_argument(
        "--compute-factors", action="store_true", help="Compute daily MOST factors."
    )
    p.add_argument("--workers", type=int, default=12, help="Download threads per day.")
    p.add_argument(
        "--start-date", default=None, help="Override start date (YYYY-MM-DD)."
    )
    p.add_argument("--end-date", default=None, help="Override end date (YYYY-MM-DD).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if args.z0_only:
        _run_z0_sample()
    elif args.compute_factors:
        _run_compute_factors()
    else:
        _run_download(args)


if __name__ == "__main__":
    main()
