"""
Build daily 1 km COGs from hourly HRRR GRIB2 archives.

Reads 24 hourly 9-variable GRIB2 files per day via eccodes, aggregates to
daily statistics using incremental accumulators, reprojects from the native
HRRR Lambert Conformal grid to the 1 km EPSG:5070 PNW grid.

Output bands (15, float32):
  tmp_hrrr, tmax_hrrr, tmin_hrrr, dpt_hrrr, ea_hrrr, pres_hrrr,
  ugrd_hrrr, vgrd_hrrr, wind_hrrr, wdir_hrrr, dswrf_hrrr, spfh_hrrr,
  tcdc_hrrr, hpbl_hrrr, n_hours

Usage:
    uv run python -m prep.build_hrrr_1km \\
        --grib-root /mnt/mco_nas1/shared/hrrr_hourly \\
        --start 2023-01-01 --end 2023-12-31 \\
        --out-dir /nas/dads/mvp/hrrr_1km_pnw \\
        --workers 4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from datetime import datetime, timedelta

import eccodes
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

from prep.pnw_1km_grid import (
    PNW_1KM_CRS,
    PNW_1KM_SHAPE,
    PNW_1KM_TRANSFORM,
)

# ── constants ─────────────────────────────────────────────────────────────────

# Message order matches TARGET_FIELDS in grid/sources/download_hrrr_archive.py:
#   0: TMP, 1: DPT, 2: UGRD, 3: VGRD, 4: DSWRF, 5: PRES, 6: TCDC, 7: HPBL, 8: SPFH
_MSG_NAMES = ["TMP", "DPT", "UGRD", "VGRD", "DSWRF", "PRES", "TCDC", "HPBL", "SPFH"]
_MSG_INDEX = {name: i for i, name in enumerate(_MSG_NAMES)}
_N_MSGS = len(_MSG_NAMES)

OUTPUT_BANDS = (
    "tmp_hrrr",
    "tmax_hrrr",
    "tmin_hrrr",
    "dpt_hrrr",
    "ea_hrrr",
    "pres_hrrr",
    "ugrd_hrrr",
    "vgrd_hrrr",
    "wind_hrrr",
    "wdir_hrrr",
    "dswrf_hrrr",
    "spfh_hrrr",
    "tcdc_hrrr",
    "hpbl_hrrr",
    "n_hours",
)

_VERSION_BOUNDARIES = [
    ("v1", datetime(2014, 11, 15), datetime(2016, 8, 22)),
    ("v2", datetime(2016, 8, 23), datetime(2018, 7, 11)),
    ("v3", datetime(2018, 7, 12), datetime(2020, 12, 1)),
    ("v4", datetime(2020, 12, 2), datetime(2099, 12, 31)),
]

# ── worker globals (set via _init_worker for fork COW) ────────────────────────

_W_GRIB_ROOT: str = ""
_W_CRS_WKT: str = ""
_W_SRC_TRANSFORM: Affine | None = None
_W_NI: int = 0
_W_NJ: int = 0
_W_J_SCAN_POS: bool = True
_W_OUT_DIR: str = ""
_W_OVERWRITE: bool = False


# ── path helpers ──────────────────────────────────────────────────────────────


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


# ── native grid metadata ──────────────────────────────────────────────────────


def get_hrrr_grid_info(
    grib_path: str,
) -> tuple[rasterio.crs.CRS, Affine, int, int, bool]:
    """Extract native CRS, north-up Affine transform, and dimensions.

    Returns
    -------
    crs : rasterio.crs.CRS
        Native HRRR Lambert Conformal CRS.
    transform : Affine
        North-up Affine transform (origin = top-left corner of NW pixel).
    Ni : int
        Grid columns (x direction).
    Nj : int
        Grid rows (y direction).
    j_scan_pos : bool
        True when j increases northward (row 0 of reshaped array is south).
    """
    with open(grib_path, "rb") as f:
        msgid = eccodes.codes_grib_new_from_file(f)
        Ni = int(eccodes.codes_get(msgid, "Ni"))
        Nj = int(eccodes.codes_get(msgid, "Nj"))
        dx = float(eccodes.codes_get(msgid, "DxInMetres"))
        dy = float(eccodes.codes_get(msgid, "DyInMetres"))
        La1 = float(eccodes.codes_get(msgid, "latitudeOfFirstGridPointInDegrees"))
        Lo1 = float(eccodes.codes_get(msgid, "longitudeOfFirstGridPointInDegrees"))
        Latin1 = float(eccodes.codes_get(msgid, "Latin1InDegrees"))
        Latin2 = float(eccodes.codes_get(msgid, "Latin2InDegrees"))
        LoV = float(eccodes.codes_get(msgid, "LoVInDegrees"))
        j_scan = int(eccodes.codes_get(msgid, "jScansPositively"))
        try:
            LaD = float(eccodes.codes_get(msgid, "LaDInDegrees"))
        except eccodes.CodesInternalError:
            LaD = Latin1
        try:
            radius = float(eccodes.codes_get(msgid, "radiusInMetres"))
        except eccodes.CodesInternalError:
            radius = 6371229.0
        eccodes.codes_release(msgid)

    lon_0 = LoV - 360.0 if LoV > 180.0 else LoV
    Lo1_signed = Lo1 - 360.0 if Lo1 > 180.0 else Lo1

    crs = rasterio.crs.CRS.from_proj4(
        f"+proj=lcc +lat_0={LaD} +lon_0={lon_0} "
        f"+lat_1={Latin1} +lat_2={Latin2} "
        f"+R={radius:.0f} +units=m +no_defs"
    )

    # Project the SW first-grid-point center to native LCC
    ll_to_xy = Transformer.from_crs("EPSG:4326", crs.to_wkt(), always_xy=True)
    x0, y0 = ll_to_xy.transform(Lo1_signed, La1)

    # Build north-up Affine: origin = top-left corner of NW pixel
    if j_scan:
        # jScansPositively=1: row 0 is south; after flip, NW corner is top
        y_top = y0 + (Nj - 0.5) * dy
    else:
        # row 0 is already north
        y_top = y0 + 0.5 * dy
    x_left = x0 - 0.5 * dx

    transform = Affine(dx, 0.0, x_left, 0.0, -dy, y_top)
    return crs, transform, Ni, Nj, bool(j_scan)


# ── hourly decode ─────────────────────────────────────────────────────────────


def _decode_hour(
    path: str,
    Ni: int,
    Nj: int,
    j_scan_pos: bool,
) -> np.ndarray | None:
    """Decode all messages from one 9-var HRRR GRIB2 file.

    Returns (n_msgs, Nj, Ni) float64, north-up, or None on failure.
    """
    arrays: list[np.ndarray] = []
    with open(path, "rb") as f:
        for _ in range(_N_MSGS):
            msgid = eccodes.codes_grib_new_from_file(f)
            if msgid is None:
                break
            vals = eccodes.codes_get_array(msgid, "values")
            arr_2d = vals.reshape(Nj, Ni)
            if j_scan_pos:
                arr_2d = arr_2d[::-1, :].copy()
            arrays.append(arr_2d.astype(np.float64))
            eccodes.codes_release(msgid)
    if len(arrays) != _N_MSGS:
        return None
    return np.array(arrays)


# ── daily aggregation helper ──────────────────────────────────────────────────


def _ea_from_dpt(dpt_c: np.ndarray) -> np.ndarray:
    """Tetens formula: dewpoint (degC) -> actual vapour pressure (kPa)."""
    return 0.6108 * np.exp(17.27 * dpt_c / (dpt_c + 237.3))


# ── day processing ────────────────────────────────────────────────────────────


def _process_day(
    grib_root: str,
    day: datetime,
    crs_wkt: str,
    src_transform: Affine,
    Ni: int,
    Nj: int,
    j_scan_pos: bool,
    out_dir: str,
    overwrite: bool,
) -> dict:
    """Read 24 hourly GRIBs, aggregate daily, reproject, write COG.

    Returns a manifest record dict.
    """
    t0 = time.monotonic()
    day_str = day.strftime("%Y%m%d")
    out_path = os.path.join(out_dir, f"HRRR_1km_{day_str}.tif")

    if not overwrite and os.path.exists(out_path):
        return {
            "date": day_str,
            "path": out_path,
            "hours_found": -1,
            "hours_missing": -1,
            "status": "skipped",
            "elapsed_s": 0.0,
        }

    # Incremental accumulators (avoid holding all 24 hourly arrays at once)
    sum_tmp: np.ndarray | None = None
    tmax_arr: np.ndarray | None = None
    tmin_arr: np.ndarray | None = None
    sum_dpt: np.ndarray | None = None
    sum_ea: np.ndarray | None = None
    sum_pres: np.ndarray | None = None
    sum_ugrd: np.ndarray | None = None
    sum_vgrd: np.ndarray | None = None
    sum_spd: np.ndarray | None = None
    sum_dswrf: np.ndarray | None = None
    sum_spfh: np.ndarray | None = None
    sum_tcdc: np.ndarray | None = None
    sum_hpbl: np.ndarray | None = None
    n_hours = 0

    for hour in range(24):
        path = _grib_path(grib_root, day, hour)
        if path is None:
            continue
        arr = _decode_hour(path, Ni, Nj, j_scan_pos)
        if arr is None:
            continue
        n_hours += 1

        # Unit conversions (in-place on float64 copy)
        arr[_MSG_INDEX["TMP"]] -= 273.15
        arr[_MSG_INDEX["DPT"]] -= 273.15
        arr[_MSG_INDEX["PRES"]] /= 1000.0

        tmp = arr[_MSG_INDEX["TMP"]]
        dpt = arr[_MSG_INDEX["DPT"]]
        u = arr[_MSG_INDEX["UGRD"]]
        v = arr[_MSG_INDEX["VGRD"]]

        if sum_tmp is None:
            sum_tmp = tmp.copy()
            tmax_arr = tmp.copy()
            tmin_arr = tmp.copy()
            sum_dpt = dpt.copy()
            sum_ea = _ea_from_dpt(dpt)
            sum_pres = arr[_MSG_INDEX["PRES"]].copy()
            sum_ugrd = u.copy()
            sum_vgrd = v.copy()
            sum_spd = np.sqrt(u**2 + v**2)
            sum_dswrf = arr[_MSG_INDEX["DSWRF"]].copy()
            sum_spfh = arr[_MSG_INDEX["SPFH"]].copy()
            sum_tcdc = arr[_MSG_INDEX["TCDC"]].copy()
            sum_hpbl = arr[_MSG_INDEX["HPBL"]].copy()
        else:
            sum_tmp += tmp
            np.maximum(tmax_arr, tmp, out=tmax_arr)
            np.minimum(tmin_arr, tmp, out=tmin_arr)
            sum_dpt += dpt
            sum_ea += _ea_from_dpt(dpt)
            sum_pres += arr[_MSG_INDEX["PRES"]]
            sum_ugrd += u
            sum_vgrd += v
            sum_spd += np.sqrt(u**2 + v**2)
            sum_dswrf += arr[_MSG_INDEX["DSWRF"]]
            sum_spfh += arr[_MSG_INDEX["SPFH"]]
            sum_tcdc += arr[_MSG_INDEX["TCDC"]]
            sum_hpbl += arr[_MSG_INDEX["HPBL"]]

    if n_hours == 0:
        return {
            "date": day_str,
            "path": None,
            "hours_found": 0,
            "hours_missing": 24,
            "status": "no_data",
            "elapsed_s": round(time.monotonic() - t0, 1),
        }

    inv_n = 1.0 / n_hours
    mu = sum_ugrd * inv_n
    mv = sum_vgrd * inv_n

    daily: dict[str, np.ndarray] = {
        "tmp_hrrr": (sum_tmp * inv_n).astype(np.float32),
        "tmax_hrrr": tmax_arr.astype(np.float32),
        "tmin_hrrr": tmin_arr.astype(np.float32),
        "dpt_hrrr": (sum_dpt * inv_n).astype(np.float32),
        "ea_hrrr": (sum_ea * inv_n).astype(np.float32),
        "pres_hrrr": (sum_pres * inv_n).astype(np.float32),
        "ugrd_hrrr": mu.astype(np.float32),
        "vgrd_hrrr": mv.astype(np.float32),
        "wind_hrrr": (sum_spd * inv_n).astype(np.float32),
        "wdir_hrrr": (np.degrees(np.arctan2(-mu, -mv)) % 360.0).astype(np.float32),
        "dswrf_hrrr": (sum_dswrf * inv_n).astype(np.float32),
        "spfh_hrrr": (sum_spfh * inv_n).astype(np.float32),
        "tcdc_hrrr": (sum_tcdc * inv_n).astype(np.float32),
        "hpbl_hrrr": (sum_hpbl * inv_n).astype(np.float32),
        "n_hours": np.full((Nj, Ni), float(n_hours), dtype=np.float32),
    }

    H, W = PNW_1KM_SHAPE
    src_crs = rasterio.crs.CRS.from_wkt(crs_wkt)
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
        for band_idx, band_name in enumerate(OUTPUT_BANDS, 1):
            buf = np.full((H, W), np.nan, dtype=np.float32)
            reproject(
                source=daily[band_name],
                destination=buf,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=PNW_1KM_TRANSFORM,
                dst_crs=PNW_1KM_CRS,
                resampling=Resampling.bilinear,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
            dst.write(buf, band_idx)
            dst.set_band_description(band_idx, band_name)

    status = "ok" if n_hours == 24 else "partial"
    return {
        "date": day_str,
        "path": out_path,
        "hours_found": n_hours,
        "hours_missing": 24 - n_hours,
        "status": status,
        "elapsed_s": round(time.monotonic() - t0, 1),
    }


# ── multiprocessing ───────────────────────────────────────────────────────────


def _init_worker(
    grib_root: str,
    crs_wkt: str,
    src_transform: Affine,
    Ni: int,
    Nj: int,
    j_scan_pos: bool,
    out_dir: str,
    overwrite: bool,
) -> None:
    """Populate per-worker globals (shared via fork COW on Linux)."""
    global _W_GRIB_ROOT, _W_CRS_WKT, _W_SRC_TRANSFORM, _W_NI, _W_NJ
    global _W_J_SCAN_POS, _W_OUT_DIR, _W_OVERWRITE
    _W_GRIB_ROOT = grib_root
    _W_CRS_WKT = crs_wkt
    _W_SRC_TRANSFORM = src_transform
    _W_NI = Ni
    _W_NJ = Nj
    _W_J_SCAN_POS = j_scan_pos
    _W_OUT_DIR = out_dir
    _W_OVERWRITE = overwrite


def _worker_process_day(day: datetime) -> dict:
    return _process_day(
        _W_GRIB_ROOT,
        day,
        _W_CRS_WKT,
        _W_SRC_TRANSFORM,
        _W_NI,
        _W_NJ,
        _W_J_SCAN_POS,
        _W_OUT_DIR,
        _W_OVERWRITE,
    )


# ── main entry point ──────────────────────────────────────────────────────────


def build_hrrr_1km(
    grib_root: str,
    out_dir: str,
    start: datetime,
    end: datetime,
    overwrite: bool = False,
    workers: int = 1,
) -> int:
    """Build daily 1 km HRRR COGs over a date range.

    Returns
    -------
    int
        Number of output files written.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Locate a reference GRIB to extract native grid geometry
    ref_path: str | None = None
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
        raise FileNotFoundError(
            f"No HRRR GRIBs found in [{start:%Y-%m-%d}, {end:%Y-%m-%d}]"
        )

    print(f"Reference GRIB: {ref_path}", flush=True)
    crs, src_transform, Ni, Nj, j_scan_pos = get_hrrr_grid_info(ref_path)
    crs_wkt = crs.to_wkt()
    print(
        f"HRRR grid: {Nj}x{Ni}, jScansPositively={j_scan_pos}, "
        f"CRS: {crs.to_string()[:60]}",
        flush=True,
    )

    n_days = (end - start).days + 1
    all_days = [start + timedelta(days=i) for i in range(n_days)]
    workers = max(1, workers)
    print(f"Processing {n_days} days with {workers} worker(s)...", flush=True)

    records: list[dict] = []
    written = 0

    if workers == 1:
        for di, day in enumerate(all_days):
            rec = _process_day(
                grib_root,
                day,
                crs_wkt,
                src_transform,
                Ni,
                Nj,
                j_scan_pos,
                out_dir,
                overwrite,
            )
            records.append(rec)
            if rec["status"] not in ("skipped", "no_data"):
                written += 1
            if (di + 1) % 10 == 0 or di == n_days - 1:
                print(
                    f"  {di + 1}/{n_days} days — {written} written "
                    f"({rec['date']} {rec['status']})",
                    flush=True,
                )
    else:
        done = 0
        with mp.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(
                grib_root,
                crs_wkt,
                src_transform,
                Ni,
                Nj,
                j_scan_pos,
                out_dir,
                overwrite,
            ),
        ) as pool:
            for rec in pool.imap_unordered(_worker_process_day, all_days, chunksize=2):
                done += 1
                records.append(rec)
                if rec["status"] not in ("skipped", "no_data"):
                    written += 1
                if done % 10 == 0 or done == n_days:
                    print(f"  {done}/{n_days} days — {written} written", flush=True)

    # Write manifest
    manifest_path = os.path.join(out_dir, "hrrr_1km_manifest.parquet")
    pd.DataFrame(records).sort_values("date").to_parquet(manifest_path, index=False)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Finished: {written} COGs written to {out_dir}", flush=True)
    return written


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build daily 1 km HRRR COGs from hourly GRIB2 archives."
    )
    p.add_argument(
        "--grib-root",
        required=True,
        help="Root of HRRR hourly GRIB archive (contains v1/v2/v3/v4 subdirs).",
    )
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD).")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD).")
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for 1 km COGs.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel worker count (default: 4).",
    )
    a = p.parse_args()

    build_hrrr_1km(
        grib_root=a.grib_root,
        out_dir=a.out_dir,
        start=datetime.strptime(a.start, "%Y-%m-%d"),
        end=datetime.strptime(a.end, "%Y-%m-%d"),
        overwrite=a.overwrite,
        workers=a.workers,
    )


if __name__ == "__main__":
    main()
