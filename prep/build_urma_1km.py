"""
Build daily 1 km COGs from hourly URMA/RTMA GRIB2 archives.

Reads 24 hourly GRIBs per day, aggregates to daily statistics, and
reprojects from the native Lambert Conformal grid to the 1 km EPSG:5070
PNW grid defined in prep.pnw_1km_grid.

Output bands (11, float32):
  tmp_c, tmax_c, tmin_c, dpt_c, ugrd, vgrd, gust, spfh, pres_kpa, tcdc_pct, ea_kpa

Usage:
    uv run python -m prep.build_urma_1km \
        --model urma \
        --grib-root /mnt/mco_nas1/shared/rtma_hourly \
        --start 2024-01-01 --end 2024-12-31 \
        --out /nas/dads/mvp/urma_1km_pnw_2024 \
        --workers 4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from datetime import datetime, timedelta

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window

from prep.pnw_1km_grid import (
    PNW_1KM_BOUNDS,
    PNW_1KM_CRS,
    PNW_1KM_SHAPE,
    PNW_1KM_TRANSFORM,
)

# ------------------------------------------------------------------ constants

EXTRACT_ELEMENTS = ["TMP", "DPT", "PRES", "UGRD", "VGRD", "SPFH", "GUST", "TCDC"]

OUTPUT_BANDS = (
    "tmp_c",
    "tmax_c",
    "tmin_c",
    "dpt_c",
    "ugrd",
    "vgrd",
    "gust",
    "spfh",
    "pres_kpa",
    "tcdc_pct",
    "ea_kpa",
)

# ---- worker globals (set via _init_worker for fork COW) ----
_W_GRIB_ROOT: str = ""
_W_MODEL: str = ""
_W_WINDOW: Window | None = None
_W_SRC_CRS_WKT: str = ""
_W_SRC_TRANSFORM: Affine | None = None
_W_WIN_H: int = 0
_W_WIN_W: int = 0
_W_OUT_DIR: str = ""


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


def _ea_from_dpt(dpt_c: np.ndarray) -> np.ndarray:
    """Tetens formula: dewpoint (degC) -> actual vapour pressure (kPa)."""
    return 0.6108 * np.exp(17.27 * dpt_c / (dpt_c + 237.3))


# ----------------------------------------------------------- day processing


def _process_day(
    grib_root: str,
    model: str,
    day: datetime,
    window: Window,
    src_crs_wkt: str,
    src_transform: Affine,
    win_h: int,
    win_w: int,
    out_dir: str,
) -> str | None:
    """Read 24 hourly GRIBs, aggregate to daily, reproject, write COG."""
    day_str = day.strftime("%Y%m%d")
    out_path = os.path.join(out_dir, f"{model.upper()}_1km_{day_str}.tif")
    if os.path.exists(out_path):
        return out_path

    target_elements = sorted(EXTRACT_ELEMENTS)
    eidx = {e: i for i, e in enumerate(target_elements)}
    n_elem = len(target_elements)

    hourly: list[np.ndarray] = []  # each: (n_elem, win_h, win_w)

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

        arr = np.full((n_elem, win_h, win_w), np.nan, dtype=np.float32)
        for k, elem in enumerate(present):
            arr[eidx[elem]] = data[k]
        hourly.append(arr)

    if not hourly:
        return None

    stack = np.array(hourly, dtype=np.float64)
    del hourly

    # Unit conversion: PRES Pa -> kPa
    if "PRES" in eidx:
        stack[:, eidx["PRES"], :, :] /= 1000.0

    # Aggregate to daily
    daily: dict[str, np.ndarray] = {}
    daily["tmp_c"] = np.nanmean(stack[:, eidx["TMP"]], axis=0).astype(np.float32)
    daily["tmax_c"] = np.nanmax(stack[:, eidx["TMP"]], axis=0).astype(np.float32)
    daily["tmin_c"] = np.nanmin(stack[:, eidx["TMP"]], axis=0).astype(np.float32)
    daily["dpt_c"] = np.nanmean(stack[:, eidx["DPT"]], axis=0).astype(np.float32)
    daily["ugrd"] = np.nanmean(stack[:, eidx["UGRD"]], axis=0).astype(np.float32)
    daily["vgrd"] = np.nanmean(stack[:, eidx["VGRD"]], axis=0).astype(np.float32)
    daily["gust"] = np.nanmax(stack[:, eidx["GUST"]], axis=0).astype(np.float32)
    daily["spfh"] = np.nanmean(stack[:, eidx["SPFH"]], axis=0).astype(np.float32)
    daily["pres_kpa"] = np.nanmean(stack[:, eidx["PRES"]], axis=0).astype(np.float32)
    daily["tcdc_pct"] = np.nanmean(stack[:, eidx["TCDC"]], axis=0).astype(np.float32)
    hourly_ea = _ea_from_dpt(stack[:, eidx["DPT"]])
    daily["ea_kpa"] = np.nanmean(hourly_ea, axis=0).astype(np.float32)
    del stack

    # Compute window transform from full-grid transform
    win_transform = rasterio.windows.transform(window, src_transform)
    src_crs = rasterio.crs.CRS.from_wkt(src_crs_wkt)

    H, W = PNW_1KM_SHAPE
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": len(OUTPUT_BANDS),
        "height": H,
        "width": W,
        "crs": PNW_1KM_CRS,
        "transform": PNW_1KM_TRANSFORM,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": float("nan"),
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        for i, band_name in enumerate(OUTPUT_BANDS, 1):
            src_arr = daily[band_name]
            buf = np.full((H, W), np.nan, dtype="float32")
            reproject(
                source=src_arr,
                destination=buf,
                src_transform=win_transform,
                src_crs=src_crs,
                dst_transform=PNW_1KM_TRANSFORM,
                dst_crs=PNW_1KM_CRS,
                resampling=Resampling.bilinear,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
            dst.write(buf, i)
            dst.set_band_description(i, band_name)

    return out_path


# ----------------------------------------------------------- multiprocessing


def _init_worker(
    grib_root: str,
    model: str,
    window: Window,
    src_crs_wkt: str,
    src_transform: Affine,
    win_h: int,
    win_w: int,
    out_dir: str,
) -> None:
    """Set per-worker globals (shared via fork COW on Linux)."""
    global _W_GRIB_ROOT, _W_MODEL, _W_WINDOW, _W_SRC_CRS_WKT
    global _W_SRC_TRANSFORM, _W_WIN_H, _W_WIN_W, _W_OUT_DIR
    _W_GRIB_ROOT = grib_root
    _W_MODEL = model
    _W_WINDOW = window
    _W_SRC_CRS_WKT = src_crs_wkt
    _W_SRC_TRANSFORM = src_transform
    _W_WIN_H = win_h
    _W_WIN_W = win_w
    _W_OUT_DIR = out_dir


def _worker_process_day(day: datetime) -> tuple[datetime, str | None]:
    """Thin wrapper that reads from worker globals."""
    return day, _process_day(
        _W_GRIB_ROOT,
        _W_MODEL,
        day,
        _W_WINDOW,
        _W_SRC_CRS_WKT,
        _W_SRC_TRANSFORM,
        _W_WIN_H,
        _W_WIN_W,
        _W_OUT_DIR,
    )


# ----------------------------------------------------------- main entry point


def build_urma_1km(
    model: str,
    grib_root: str,
    out_dir: str,
    start: datetime,
    end: datetime,
    workers: int = 1,
) -> int:
    """Build daily 1 km COGs from hourly GRIB2 archives."""
    os.makedirs(out_dir, exist_ok=True)

    # Find reference GRIB for CRS and grid geometry
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

    with rasterio.open(ref_path) as src:
        src_crs = src.crs
        src_crs_wkt = src_crs.to_wkt()
        src_transform = src.transform
        grid_h, grid_w = src.height, src.width

    print(f"GRIB grid: {grid_h}x{grid_w}, CRS: {src_crs}", flush=True)

    # Compute GRIB-space window covering the PNW 1km grid extent
    pnw_left, pnw_bottom, pnw_right, pnw_top = PNW_1KM_BOUNDS
    grib_left, grib_bottom, grib_right, grib_top = transform_bounds(
        PNW_1KM_CRS, src_crs, pnw_left, pnw_bottom, pnw_right, pnw_top
    )

    inv = ~src_transform
    c0f, r0f = inv * (grib_left, grib_top)
    c1f, r1f = inv * (grib_right, grib_bottom)

    buf = 5  # buffer for reprojection edge effects
    r0 = max(0, int(np.floor(min(r0f, r1f))) - buf)
    c0 = max(0, int(np.floor(min(c0f, c1f))) - buf)
    r1 = min(grid_h, int(np.ceil(max(r0f, r1f))) + buf)
    c1 = min(grid_w, int(np.ceil(max(c0f, c1f))) + buf)

    window = Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0)
    win_h, win_w = r1 - r0, c1 - c0
    pct = win_h * win_w / (grid_h * grid_w) * 100
    print(
        f"Read window: rows [{r0}:{r1}], cols [{c0}:{c1}]  "
        f"({win_h}x{win_w}, {pct:.1f}% of grid)",
        flush=True,
    )

    # Process days
    n_days = (end - start).days + 1
    all_days = [start + timedelta(days=i) for i in range(n_days)]
    workers = max(1, workers)
    print(f"Processing {n_days} days with {workers} worker(s)...", flush=True)

    done = 0
    written = 0

    if workers == 1:
        for day in all_days:
            result = _process_day(
                grib_root,
                model,
                day,
                window,
                src_crs_wkt,
                src_transform,
                win_h,
                win_w,
                out_dir,
            )
            done += 1
            if result is not None:
                written += 1
            if done % 10 == 0 or done == n_days:
                print(f"  {done}/{n_days} days, {written} written", flush=True)
    else:
        with mp.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(
                grib_root,
                model,
                window,
                src_crs_wkt,
                src_transform,
                win_h,
                win_w,
                out_dir,
            ),
        ) as pool:
            for day, result in pool.imap_unordered(
                _worker_process_day, all_days, chunksize=2
            ):
                done += 1
                if result is not None:
                    written += 1
                if done % 10 == 0 or done == n_days:
                    print(f"  {done}/{n_days} days, {written} written", flush=True)

    print(f"Finished: {written} COGs written to {out_dir}", flush=True)
    return written


# ------------------------------------------------------------------ CLI


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build daily 1 km COGs from hourly GRIB2 archives."
    )
    p.add_argument("--model", required=True, choices=["urma", "rtma"])
    p.add_argument("--grib-root", required=True, help="Root of hourly GRIB archive.")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD).")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD).")
    p.add_argument("--out", required=True, help="Output directory for 1 km COGs.")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers.")
    a = p.parse_args()

    build_urma_1km(
        model=a.model,
        grib_root=a.grib_root,
        out_dir=a.out,
        start=datetime.strptime(a.start, "%Y-%m-%d"),
        end=datetime.strptime(a.end, "%Y-%m-%d"),
        workers=a.workers,
    )


if __name__ == "__main__":
    main()
