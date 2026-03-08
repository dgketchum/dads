"""Download GOES-R ABI MCMIP-C for a date range and export clipped COG GeoTIFFs.

Substitute for GridSat-B1 (stalled at 2025-08).  Downloads 3-hourly CONUS
multi-band clean-IR/WV/visible composites from NOAA's public S3 buckets,
reprojects from geostationary to EPSG:4326 at 0.07° resolution (matching
GridSat), clips to URMA+1° extent, and writes per-channel COGs.

For dates after 2022-09-01, merges GOES-16 (east) and GOES-18 (west) using
a longitude split at -105°W.

Uses a process pool to avoid HDF5/NetCDF4 thread-safety issues, a parquet
manifest for resume support, and signal handling for graceful shutdown.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

from extract.rs.goes_abi.goes_abi_common import (
    BAND_MAP,
    BUCKETS,
    CLIP_BOUNDS,
    HOURS_3H,
    SAT_LON_SPLIT,
    TARGET_RES_DEG,
    make_s3_client,
    pick_nearest_scan,
    s3_prefix,
    satellites_for_date,
)

# ── output grid geometry ─────────────────────────────────────────────────────

_LON_MIN = CLIP_BOUNDS["lon_min"]
_LON_MAX = CLIP_BOUNDS["lon_max"]
_LAT_MIN = CLIP_BOUNDS["lat_min"]
_LAT_MAX = CLIP_BOUNDS["lat_max"]

_DST_WIDTH = int(round((_LON_MAX - _LON_MIN) / TARGET_RES_DEG))
_DST_HEIGHT = int(round((_LAT_MAX - _LAT_MIN) / TARGET_RES_DEG))

_DST_TRANSFORM = from_bounds(
    _LON_MIN, _LAT_MIN, _LON_MAX, _LAT_MAX, _DST_WIDTH, _DST_HEIGHT
)

# channel list in write order
_CHANNELS = list(BAND_MAP.keys())


# ── reprojection ──────────────────────────────────────────────────────────────


def _crs_from_mcmip(ds: xr.Dataset) -> CRS:
    """Build a pyproj CRS from MCMIP-C ``goes_imager_projection`` metadata."""
    proj = ds["goes_imager_projection"]
    return CRS.from_cf(
        {
            "grid_mapping_name": "geostationary",
            "longitude_of_projection_origin": float(
                proj.attrs["longitude_of_projection_origin"]
            ),
            "perspective_point_height": float(proj.attrs["perspective_point_height"]),
            "semi_major_axis": float(proj.attrs["semi_major_axis"]),
            "semi_minor_axis": float(proj.attrs["semi_minor_axis"]),
            "sweep_angle_axis": str(proj.attrs["sweep_angle_axis"]),
        }
    )


def _src_transform(ds: xr.Dataset) -> rasterio.transform.Affine:
    """Build source affine transform from MCMIP-C x/y scanning-angle coords.

    The x and y variables are in radians and must be scaled by the satellite
    height to produce metres in the geostationary projection.
    """
    h = float(ds["goes_imager_projection"].attrs["perspective_point_height"])
    x = ds["x"].values * h
    y = ds["y"].values * h

    dx = abs(float(x[1] - x[0]))
    dy = abs(float(y[1] - y[0]))

    # x increases left-to-right; y decreases top-to-bottom
    x_min = float(x.min()) - dx / 2
    y_max = float(y.max()) + dy / 2

    return rasterio.transform.from_origin(x_min, y_max, dx, dy)


def _reproject_band(
    src_data: np.ndarray,
    src_crs: CRS,
    src_tf: rasterio.transform.Affine,
) -> np.ndarray:
    """Reproject a single 2-D band from geostationary to EPSG:4326 on the target grid."""
    dst = np.full((_DST_HEIGHT, _DST_WIDTH), np.nan, dtype=np.float32)
    reproject(
        source=src_data.astype(np.float32),
        destination=dst,
        src_crs=src_crs,
        src_transform=src_tf,
        dst_crs="EPSG:4326",
        dst_transform=_DST_TRANSFORM,
        resampling=Resampling.average,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def _merge_east_west(east: np.ndarray, west: np.ndarray) -> np.ndarray:
    """Merge GOES-16 (east) and GOES-18 (west) grids at SAT_LON_SPLIT.

    Both arrays must be on the same target grid.  West pixels are used where
    longitude < SAT_LON_SPLIT, east pixels elsewhere.
    """
    # Build a longitude vector for the target grid
    lons = np.linspace(
        _LON_MIN + TARGET_RES_DEG / 2,
        _LON_MAX - TARGET_RES_DEG / 2,
        _DST_WIDTH,
    )
    west_mask = lons < SAT_LON_SPLIT  # shape (W,)

    merged = east.copy()
    merged[:, west_mask] = west[:, west_mask]

    # Fill any remaining NaN in merged from the other source
    east_nan = np.isnan(merged)
    if east_nan.any():
        merged[east_nan] = west[east_nan]

    return merged


# ── calibration ───────────────────────────────────────────────────────────────

_QMAP_CACHE: dict | None = None


def _load_qmap(qmap_path: str) -> dict | None:
    """Load quantile-mapping calibration artifact if it exists."""
    global _QMAP_CACHE
    if _QMAP_CACHE is not None:
        return _QMAP_CACHE
    if not os.path.exists(qmap_path):
        return None
    with open(qmap_path) as f:
        _QMAP_CACHE = json.load(f)
    return _QMAP_CACHE


def _apply_qmap(data: np.ndarray, channel: str, qmap: dict) -> np.ndarray:
    """Apply quantile mapping to a single band."""
    if channel not in qmap:
        return data
    entry = qmap[channel]
    goes_bp = np.array(entry["goes_breakpoints"], dtype=np.float32)
    gridsat_bp = np.array(entry["gridsat_breakpoints"], dtype=np.float32)
    valid = ~np.isnan(data)
    out = data.copy()
    out[valid] = np.interp(data[valid], goes_bp, gridsat_bp)
    return out


# ── worker ────────────────────────────────────────────────────────────────────


def download_and_process(
    date_str: str,
    hh: str,
    out_dir: str,
    tmp_dir: str,
    calibrate: bool = True,
    qmap_path: str = "",
) -> dict:
    """Download one 3-hourly slot, reproject, and write COGs.

    Args use strings/primitives for pickling across process boundary.
    Returns a manifest row dict.
    """
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    ds_label = d.strftime("%Y%m%d")
    target_dt = datetime(d.year, d.month, d.day, int(hh))

    base_row = {
        "date": date_str,
        "hour": hh,
        "satellite": "",
        "scan_time": "",
        "status": "",
        "n_channels": 0,
        "size_bytes": 0,
        "downloaded_at": datetime.now().isoformat(),
    }

    # Skip if all 3 COGs exist
    if all(
        os.path.exists(os.path.join(out_dir, f"goes_abi_{ds_label}_{hh}00_{ch}.tif"))
        for ch in _CHANNELS
    ):
        return {**base_row, "status": "exists", "n_channels": len(_CHANNELS)}

    s3 = make_s3_client()
    sats = satellites_for_date(d)

    # Load calibration map if requested
    qmap = None
    if calibrate and qmap_path:
        qmap = _load_qmap(qmap_path)

    # Download and reproject each satellite
    sat_bands: dict[str, dict[str, np.ndarray]] = {}
    scan_times: list[str] = []

    for sat in sats:
        bucket = BUCKETS[sat]
        prefix = s3_prefix(sat, target_dt)
        key = pick_nearest_scan(s3, bucket, prefix, target_dt)
        if key is None:
            continue

        local_nc = os.path.join(tmp_dir, f"p{os.getpid()}_{sat}_{ds_label}_{hh}.nc")
        try:
            s3.download_file(bucket, key, local_nc)
        except Exception:
            continue

        try:
            ds = xr.open_dataset(local_nc, engine="netcdf4")
            crs = _crs_from_mcmip(ds)
            src_tf = _src_transform(ds)

            bands: dict[str, np.ndarray] = {}
            for ch_name, abi_var in BAND_MAP.items():
                if abi_var not in ds.data_vars:
                    continue
                raw = ds[abi_var].values
                if raw.ndim == 3:
                    raw = raw[0]
                bands[ch_name] = _reproject_band(raw, crs, src_tf)

            # Record scan time from filename
            scan_start = key.split("_s")[-1].split("_")[0] if "_s" in key else ""
            scan_times.append(f"{sat}:{scan_start}")

            sat_bands[sat] = bands
            ds.close()
        finally:
            if os.path.exists(local_nc):
                os.remove(local_nc)

    if not sat_bands:
        return {**base_row, "status": "missing"}

    # Merge or use single satellite
    sat_keys = list(sat_bands.keys())
    if len(sat_keys) == 2 and "goes16" in sat_bands and "goes18" in sat_bands:
        merged_bands: dict[str, np.ndarray] = {}
        for ch in _CHANNELS:
            east = sat_bands["goes16"].get(ch)
            west = sat_bands["goes18"].get(ch)
            if east is not None and west is not None:
                merged_bands[ch] = _merge_east_west(east, west)
            elif east is not None:
                merged_bands[ch] = east
            elif west is not None:
                merged_bands[ch] = west
        sat_label = "goes16+goes18"
    else:
        merged_bands = sat_bands[sat_keys[0]]
        sat_label = sat_keys[0]

    # Apply calibration and write COGs
    total_bytes = 0
    n_written = 0

    for ch, data in merged_bands.items():
        if qmap is not None:
            data = _apply_qmap(data, ch, qmap)

        out_name = f"goes_abi_{ds_label}_{hh}00_{ch}.tif"
        out_path = os.path.join(out_dir, out_name)

        with rasterio.open(
            out_path,
            "w",
            driver="COG",
            height=_DST_HEIGHT,
            width=_DST_WIDTH,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=_DST_TRANSFORM,
        ) as dst:
            dst.write(data, 1)

        total_bytes += os.path.getsize(out_path)
        n_written += 1

    return {
        **base_row,
        "satellite": sat_label,
        "scan_time": ";".join(scan_times),
        "status": "done" if n_written == len(_CHANNELS) else "partial",
        "n_channels": n_written,
        "size_bytes": total_bytes,
    }


# ── manifest ──────────────────────────────────────────────────────────────────

_MANIFEST_COLS = [
    "date",
    "hour",
    "satellite",
    "scan_time",
    "status",
    "n_channels",
    "size_bytes",
    "downloaded_at",
]


def _load_manifest(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame(columns=_MANIFEST_COLS)


def _save_manifest(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)


def _done_keys(manifest: pd.DataFrame) -> set[tuple[str, str]]:
    """Set of (date, hour) already completed or missing."""
    if manifest.empty:
        return set()
    done = manifest[manifest["status"].isin(("done", "exists", "missing"))]
    return set(zip(done["date"], done["hour"]))


# ── signal handling ───────────────────────────────────────────────────────────

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n  Shutdown requested — finishing current batch ...", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GOES-R ABI MCMIP-C clipped COGs"
    )
    parser.add_argument("--start", default="2025-09-01", help="Start date YYYY-MM-DD")
    parser.add_argument(
        "--end", default=None, help="End date YYYY-MM-DD (default: yesterday)"
    )
    parser.add_argument(
        "--out", default="/nas/dads/rs/goes_abi", help="Output root directory"
    )
    parser.add_argument(
        "--tmp", default="/tmp/goes_abi_nc", help="Temp dir for raw NetCDFs"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Parallel download processes"
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip quantile-map calibration (write raw GOES values)",
    )
    parser.add_argument(
        "--qmap",
        default="artifacts/goes_to_gridsat_qmap.json",
        help="Path to quantile-mapping calibration artifact",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    os.makedirs(args.tmp, exist_ok=True)

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end or yesterday, "%Y-%m-%d")
    calibrate = not args.no_calibrate

    manifest_path = os.path.join(args.out, "manifest.parquet")
    manifest = _load_manifest(manifest_path)
    done = _done_keys(manifest)
    new_rows: list[dict] = []

    total_downloaded = 0
    total_bytes = 0
    t_start = time.time()

    print(f"\n{'=' * 60}")
    print(f"  GOES ABI: {args.start} -> {args.end or yesterday}")
    print(f"  Workers: {args.workers}  Calibrate: {calibrate}")
    print(f"  Already done: {len(done)} slots")
    print(f"{'=' * 60}\n", flush=True)

    # Process one month at a time
    cur = start.replace(day=1)
    while cur <= end:
        if _shutdown_requested:
            break

        year_dir = os.path.join(args.out, cur.strftime("%Y"))
        os.makedirs(year_dir, exist_ok=True)

        tasks: list[tuple[str, str, str]] = []
        day = max(cur, start)
        if cur.month == 12:
            next_month = cur.replace(year=cur.year + 1, month=1)
        else:
            next_month = cur.replace(month=cur.month + 1)

        while day < next_month and day <= end:
            ds = day.strftime("%Y-%m-%d")
            for hh in HOURS_3H:
                if (ds, hh) not in done:
                    tasks.append((ds, hh, year_dir))
            day += timedelta(days=1)

        if not tasks:
            cur = next_month
            continue

        month_label = cur.strftime("%Y-%m")
        month_done = 0
        month_bytes = 0

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    download_and_process,
                    ds,
                    hh,
                    yd,
                    args.tmp,
                    calibrate,
                    args.qmap,
                ): (ds, hh)
                for ds, hh, yd in tasks
            }
            for future in as_completed(futures):
                if _shutdown_requested:
                    break
                ds, hh = futures[future]
                try:
                    row = future.result()
                    new_rows.append(row)
                    done.add((row["date"], row["hour"]))
                    if row["status"] in ("done", "partial"):
                        month_done += 1
                        month_bytes += row["size_bytes"]
                except Exception as e:
                    print(f"  ERROR {ds} {hh}Z: {e}", flush=True)

        total_downloaded += month_done
        total_bytes += month_bytes
        elapsed = time.time() - t_start
        print(
            f"  {month_label}: {month_done} slots  "
            f"{month_bytes / 1e6:.0f} MB  "
            f"[total: {total_downloaded} slots, {total_bytes / 1e9:.2f} GB, "
            f"{elapsed / 3600:.1f}h]",
            flush=True,
        )

        # Flush manifest each month
        if new_rows:
            manifest = pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True)
            new_rows.clear()
            _save_manifest(manifest, manifest_path)

        cur = next_month

    # Final manifest save
    if new_rows:
        manifest = pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True)
        _save_manifest(manifest, manifest_path)

    elapsed_h = (time.time() - t_start) / 3600
    print(
        f"\n  Done: {total_downloaded} slots, {total_bytes / 1e9:.2f} GB in {elapsed_h:.1f}h"
    )
    if _shutdown_requested:
        print("  (interrupted — resume by re-running the same command)")


if __name__ == "__main__":
    main()
