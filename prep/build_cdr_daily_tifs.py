"""
Preprocess VIIRS/AVHRR CDR daily netCDFs to PNW GeoTIFFs at native 0.05-deg resolution.

Reads daily CDR files and writes 5-band GeoTIFFs covering the PNW domain.

Bands: i1_cdr, i2_cdr, bt15_cdr, szen_cdr, cloud_state_cdr

Usage:
    uv run python prep/build_cdr_daily_tifs.py \
        --src /nas/dads/rs/cdr/nc \
        --dst /nas/dads/mvp/cdr_native_pnw \
        --start 2018-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np

try:
    import netCDF4
except ImportError:
    netCDF4 = None

import rasterio
from rasterio.transform import from_bounds

# PNW domain at 0.05-deg resolution
LAT_MIN, LAT_MAX = 42.0, 49.0
LON_MIN, LON_MAX = -125.0, -104.0
RES = 0.05

# CDR global grid: 3600 rows x 7200 cols, row 0 = 90 N, col 0 = 180 W
ROW_START = int((90.0 - LAT_MAX) / RES)
ROW_END = int((90.0 - LAT_MIN) / RES)
COL_START = int((LON_MIN + 180.0) / RES)
COL_END = int((LON_MAX + 180.0) / RES)
N_LAT = ROW_END - ROW_START
N_LON = COL_END - COL_START

VIIRS_START_YEAR = 2014

VIIRS_VARS = {
    "BRDF_corrected_I1_SurfRefl_CMG": "i1_cdr",
    "BRDF_corrected_I2_SurfRefl_CMG": "i2_cdr",
    "BT_CH15": "bt15_cdr",
    "SZEN": "szen_cdr",
}
AVHRR_VARS = {
    "SREFL_CH1": "i1_cdr",
    "SREFL_CH2": "i2_cdr",
    "BT_CH4": "bt15_cdr",
    "SZEN": "szen_cdr",
}
QA_VAR = "QA"
BAND_NAMES = ["i1_cdr", "i2_cdr", "bt15_cdr", "szen_cdr", "cloud_state_cdr"]
N_BANDS = len(BAND_NAMES)

_DATE_RE = re.compile(r"(\d{8})")


def _parse_date(filename: str) -> str | None:
    m = _DATE_RE.search(filename)
    return m.group(1) if m else None


def _process_one(nc_path: str, out_dir: str) -> str | None:
    basename = os.path.basename(nc_path)
    date_str = _parse_date(basename)
    if date_str is None:
        return None

    out_path = os.path.join(out_dir, f"CDR_005deg_{date_str}.tif")
    if os.path.exists(out_path):
        return out_path

    year = int(date_str[:4])
    var_map = VIIRS_VARS if year >= VIIRS_START_YEAR else AVHRR_VARS

    ds = netCDF4.Dataset(nc_path, "r")
    try:
        out_data = np.full((N_BANDS, N_LAT, N_LON), np.nan, dtype="float32")

        for nc_var, out_name in var_map.items():
            if nc_var not in ds.variables:
                continue
            band_idx = BAND_NAMES.index(out_name)
            arr = ds.variables[nc_var][:]
            if isinstance(arr, np.ma.MaskedArray):
                arr = arr.filled(np.nan)
            if arr.ndim == 3:
                arr = arr[0]
            subset = arr[ROW_START:ROW_END, COL_START:COL_END].astype("float32")
            subset[subset < -9000] = np.nan
            out_data[band_idx] = subset

        if QA_VAR in ds.variables:
            qa = ds.variables[QA_VAR][:]
            if isinstance(qa, np.ma.MaskedArray):
                qa = qa.filled(-1)
            if qa.ndim == 3:
                qa = qa[0]
            qa_subset = qa[ROW_START:ROW_END, COL_START:COL_END]
            cloud_state = (qa_subset & 0x03).astype("float32")
            cloud_state[qa_subset < 0] = np.nan
            out_data[BAND_NAMES.index("cloud_state_cdr")] = cloud_state
    finally:
        ds.close()

    transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, N_LON, N_LAT)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=N_LAT,
        width=N_LON,
        count=N_BANDS,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        compress="zstd",
    ) as dst:
        for i, name in enumerate(BAND_NAMES):
            dst.write(out_data[i], i + 1)
            dst.set_band_description(i + 1, name)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="CDR netCDF to PNW GeoTIFF")
    parser.add_argument("--src", default="/nas/dads/rs/cdr/nc")
    parser.add_argument("--dst", default="/nas/dads/mvp/cdr_native_pnw")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    start = args.start.replace("-", "")
    end = args.end.replace("-", "")

    nc_files = sorted(
        f
        for f in (os.path.join(args.src, n) for n in os.listdir(args.src))
        if f.endswith(".nc")
    )
    filtered = []
    for f in nc_files:
        d = _parse_date(os.path.basename(f))
        if d and start <= d <= end:
            filtered.append(f)

    print(f"Found {len(filtered)} CDR netCDFs in [{args.start}, {args.end}]")

    if args.workers > 1:
        from multiprocessing import Pool

        with Pool(args.workers) as pool:
            results = pool.starmap(_process_one, [(f, args.dst) for f in filtered])
    else:
        results = []
        for i, f in enumerate(filtered):
            results.append(_process_one(f, args.dst))
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(filtered)}")

    n_ok = sum(1 for r in results if r is not None)
    print(f"Wrote {n_ok} GeoTIFFs to {args.dst}")


if __name__ == "__main__":
    main()
