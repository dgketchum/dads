"""
Resample raw RTMA daily COGs (EPSG:4326, Int32) to the 1 km EPSG:5070 PNW grid.

Each output TIF has 8 decoded float32 bands:
  tmp_c, dpt_c, ugrd, vgrd, pres_kpa, tcdc_pct, prcp_mm, ea_kpa

Usage:
    uv run python -m prep.build_rtma_1km \
        --tif-root /data/ssd1/rtma/tif \
        --start 2024-01-01 --end 2024-12-31 \
        --out /nas/dads/mvp/rtma_1km_pnw_2024
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

from prep.pnw_1km_grid import PNW_1KM_CRS, PNW_1KM_SHAPE, PNW_1KM_TRANSFORM
from process.gridded.rtma_patch_sampler import _decode_rtma_raw_patch

DECODED_BANDS = (
    "tmp_c",
    "dpt_c",
    "ugrd",
    "vgrd",
    "pres_kpa",
    "tcdc_pct",
    "prcp_mm",
    "ea_kpa",
)


def resample_one(tif_path: str, out_path: str) -> bool:
    """Decode a raw RTMA COG and reproject to the 1 km PNW grid.

    Returns True on success, False if the source file is missing.
    """
    if not os.path.exists(tif_path):
        return False

    H, W = PNW_1KM_SHAPE

    with rasterio.open(tif_path) as src:
        raw = src.read()  # (B, H_src, W_src) int32
        decoded = _decode_rtma_raw_patch(src, raw)
        src_crs = src.crs
        src_tf = src.transform

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": len(DECODED_BANDS),
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
        for i, band_name in enumerate(DECODED_BANDS, 1):
            src_arr = decoded[band_name]
            buf = np.full((H, W), np.nan, dtype="float32")
            reproject(
                source=src_arr,
                destination=buf,
                src_transform=src_tf,
                src_crs=src_crs,
                dst_transform=PNW_1KM_TRANSFORM,
                dst_crs=PNW_1KM_CRS,
                resampling=Resampling.bilinear,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
            dst.write(buf, i)
            dst.set_band_description(i, band_name)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Resample RTMA COGs to 1 km PNW grid.")
    p.add_argument("--tif-root", required=True, help="Directory with RTMA_YYYYMMDD.tif")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--out", required=True, help="Output directory for 1 km TIFs")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    days = pd.date_range(a.start, a.end, freq="D")
    done, skipped = 0, 0

    for day in days:
        ymd = day.strftime("%Y%m%d")
        src_path = os.path.join(a.tif_root, f"RTMA_{ymd}.tif")
        dst_path = os.path.join(a.out, f"RTMA_1km_{ymd}.tif")

        if os.path.exists(dst_path):
            done += 1
            continue

        ok = resample_one(src_path, dst_path)
        if ok:
            done += 1
        else:
            skipped += 1

        if done % 25 == 0 or day == days[-1]:
            print(f"  {done}/{len(days)} done, {skipped} skipped", flush=True)

    print(f"Finished: {done} written, {skipped} skipped in {a.out}")


if __name__ == "__main__":
    main()
