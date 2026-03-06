"""
Build RTMA-aligned climatological Landsat composite grid for the PNW subset.

Computes per-pixel nanmean across all available years (1987-2025) for each
of 5 seasonal periods × 7 spectral bands, then reprojects from EPSG:5070 1 km
to the RTMA EPSG:4326 PNW grid.

Output
------
- landsat_pnw_rtma.tif — 35-band float32 GeoTIFF
  Band order: P0_B2, P0_B3, P0_B4, P0_B5, P0_B6, P0_B7, P0_B10, P1_B2, ..., P4_B10
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from terrain.grid import RTMA_CRS, _pnw_rtma_transform_and_shape
from prep.pnw_1km_grid import PNW_1KM_CRS, PNW_1KM_SHAPE, PNW_1KM_TRANSFORM

LANDSAT_DIR = "/nas/dads/landsat_composites"
YEARS = list(range(1987, 2026))
PERIODS = 5
BANDS = 7
BAND_NAMES = ("B2", "B3", "B4", "B5", "B6", "B7", "B10")
TOTAL_BANDS = PERIODS * BANDS  # 35

# Descaling: reflectance bands (0-5) / 10000, thermal (6) / 100
_REFLECTANCE_SCALE = 10000.0
_THERMAL_SCALE = 100.0


def _descale_band(arr: np.ndarray, band_idx: int) -> np.ndarray:
    """Descale int32 values: reflectance → [0,1], thermal → Kelvin."""
    out = arr.astype("float32")
    # Zero is nodata
    out[arr == 0] = np.nan
    if band_idx < 6:
        out /= _REFLECTANCE_SCALE
        out = np.clip(out, 0.0, None, out=out)
        out[np.isnan(arr.astype("float32"))] = np.nan  # restore NaNs after clip
    else:
        out /= _THERMAL_SCALE
    return out


def build_landsat(landsat_dir: str, out_tif: str) -> str:
    """Build 35-band climatological Landsat composite on RTMA PNW grid."""
    tf, H, W = _pnw_rtma_transform_and_shape()
    print(f"Target RTMA grid: {W}×{H}")

    # Get source grid geometry from any composite file
    ref_path = os.path.join(landsat_dir, f"{YEARS[0]}_p0.tif")
    with rasterio.open(ref_path) as ref:
        src_crs = ref.crs
        src_tf = ref.transform
        src_h, src_w = ref.height, ref.width
    print(f"Source grid: {src_w}×{src_h}, CRS: {src_crs}")

    os.makedirs(os.path.dirname(out_tif) or ".", exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": TOTAL_BANDS,
        "height": H,
        "width": W,
        "crs": RTMA_CRS,
        "transform": tf,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": float("nan"),
    }

    with rasterio.open(out_tif, "w", **profile) as dst:
        out_band = 0
        for p in range(PERIODS):
            # Collect available files for this period
            paths = []
            for yr in YEARS:
                fp = os.path.join(landsat_dir, f"{yr}_p{p}.tif")
                if os.path.exists(fp):
                    paths.append(fp)
            print(f"  Period {p}: {len(paths)} files")

            for b in range(BANDS):
                # Stack this band across all years → compute nanmean
                stack = np.empty((len(paths), src_h, src_w), dtype="float32")
                for i, fp in enumerate(paths):
                    with rasterio.open(fp) as src:
                        raw = src.read(b + 1)  # int32 (H, W)
                    stack[i] = _descale_band(raw, b)

                clim = np.nanmean(stack, axis=0)  # (src_h, src_w) float32
                del stack

                # Fill any remaining NaN with 0 before reprojection
                clim = np.nan_to_num(clim, nan=0.0)

                # Reproject to RTMA grid
                buf = np.full((H, W), np.nan, dtype="float32")
                reproject(
                    source=clim,
                    destination=buf,
                    src_transform=src_tf,
                    src_crs=src_crs,
                    dst_transform=tf,
                    dst_crs=RTMA_CRS,
                    resampling=Resampling.bilinear,
                    src_nodata=0.0,
                    dst_nodata=np.nan,
                )
                del clim

                out_band += 1
                desc = f"p{p}_{BAND_NAMES[b].lower()}"
                dst.write(buf, out_band)
                dst.set_band_description(out_band, desc)

            print(f"  Period {p} done ({out_band}/{TOTAL_BANDS} bands written)")

    print(f"Landsat composite written: {out_tif} ({TOTAL_BANDS} bands, {H}×{W})")
    return out_tif


def build_landsat_1km(landsat_dir: str, out_tif: str) -> str:
    """Build 35-band climatological Landsat composite on the 1 km PNW grid.

    Source composites are already EPSG:5070 1 km, so this is mostly a clip +
    reproject (handles any minor CRS/extent differences).
    """
    H, W = PNW_1KM_SHAPE
    tf = PNW_1KM_TRANSFORM
    dst_crs = PNW_1KM_CRS
    print(f"Target 1 km grid: {W}×{H}")

    ref_path = os.path.join(landsat_dir, f"{YEARS[0]}_p0.tif")
    with rasterio.open(ref_path) as ref:
        src_crs = ref.crs
        src_tf = ref.transform
        src_h, src_w = ref.height, ref.width
    print(f"Source grid: {src_w}×{src_h}, CRS: {src_crs}")

    os.makedirs(os.path.dirname(out_tif) or ".", exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": TOTAL_BANDS,
        "height": H,
        "width": W,
        "crs": dst_crs,
        "transform": tf,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": float("nan"),
    }

    with rasterio.open(out_tif, "w", **profile) as dst:
        out_band = 0
        for p in range(PERIODS):
            paths = []
            for yr in YEARS:
                fp = os.path.join(landsat_dir, f"{yr}_p{p}.tif")
                if os.path.exists(fp):
                    paths.append(fp)
            print(f"  Period {p}: {len(paths)} files")

            for b in range(BANDS):
                stack = np.empty((len(paths), src_h, src_w), dtype="float32")
                for i, fp in enumerate(paths):
                    with rasterio.open(fp) as src:
                        raw = src.read(b + 1)
                    stack[i] = _descale_band(raw, b)

                clim = np.nanmean(stack, axis=0)
                del stack
                clim = np.nan_to_num(clim, nan=0.0)

                buf = np.full((H, W), np.nan, dtype="float32")
                reproject(
                    source=clim,
                    destination=buf,
                    src_transform=src_tf,
                    src_crs=src_crs,
                    dst_transform=tf,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=0.0,
                    dst_nodata=np.nan,
                )
                del clim

                out_band += 1
                desc = f"p{p}_{BAND_NAMES[b].lower()}"
                dst.write(buf, out_band)
                dst.set_band_description(out_band, desc)

            print(f"  Period {p} done ({out_band}/{TOTAL_BANDS} bands written)")

    print(f"Landsat 1 km composite written: {out_tif} ({TOTAL_BANDS} bands, {H}×{W})")
    return out_tif


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build climatological Landsat composite for PNW."
    )
    p.add_argument(
        "--landsat-dir",
        default=LANDSAT_DIR,
        help="Directory of {year}_p{period}.tif composites",
    )
    p.add_argument(
        "--out-dir",
        default="/nas/dads/mvp",
        help="Output directory",
    )
    p.add_argument(
        "--grid",
        choices=["rtma", "1km"],
        default="rtma",
        help="Target grid: 'rtma' (EPSG:4326 ~2.5 km) or '1km' (EPSG:5070 1 km)",
    )
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.grid == "1km":
        build_landsat_1km(a.landsat_dir, os.path.join(a.out_dir, "landsat_pnw_1km.tif"))
    else:
        build_landsat(a.landsat_dir, os.path.join(a.out_dir, "landsat_pnw_rtma.tif"))
    print("Done.")


if __name__ == "__main__":
    main()
