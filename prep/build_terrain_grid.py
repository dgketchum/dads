"""
Build RTMA-aligned terrain and RSUN grids for the PNW subset.

Outputs
-------
- terrain_pnw_rtma.tif  — 6-band static terrain (elevation, slope, aspect_sin,
  aspect_cos, tpi_4, tpi_10) on the RTMA EPSG:4326 grid clipped to PNW.
- rsun_pnw_rtma.tif — 365-band clear-sky GHI (one band per DOY) reprojected from
  EPSG:5070 1 km to the same RTMA grid.

Both outputs share the exact RTMA grid geometry so the training dataset can index
them with the same (row, col) offsets used for COG patches.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject


# ---------------------------------------------------------------------------
# PNW bounds (cube-aligned) and RTMA grid constants
# ---------------------------------------------------------------------------
PNW_WEST, PNW_SOUTH, PNW_EAST, PNW_NORTH = -125.0, 42.0, -104.0, 49.0
RTMA_RES = 0.022457882102988037  # degrees per pixel
RTMA_CRS = CRS.from_epsg(4326)

# RTMA full-grid origin from /data/ssd1/rtma/tif/RTMA_20240101.tif
_RTMA_ORIGIN_X = -139.01429021749595
_RTMA_ORIGIN_Y = 61.13035508433344

TERRAIN_BAND_NAMES = (
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "tpi_4",
    "tpi_10",
)


def _pnw_rtma_transform_and_shape() -> tuple[Affine, int, int]:
    """Compute the RTMA sub-grid transform and (H, W) for PNW bounds."""
    col0 = round((PNW_WEST - _RTMA_ORIGIN_X) / RTMA_RES)
    row0 = round((_RTMA_ORIGIN_Y - PNW_NORTH) / RTMA_RES)
    col1 = round((PNW_EAST - _RTMA_ORIGIN_X) / RTMA_RES)
    row1 = round((_RTMA_ORIGIN_Y - PNW_SOUTH) / RTMA_RES)
    W = col1 - col0
    H = row1 - row0
    tf = Affine(
        RTMA_RES,
        0.0,
        _RTMA_ORIGIN_X + col0 * RTMA_RES,
        0.0,
        -RTMA_RES,
        _RTMA_ORIGIN_Y - row0 * RTMA_RES,
    )
    return tf, H, W


# ---------------------------------------------------------------------------
# Horn's slope / aspect  (pure numpy, no scipy)
# ---------------------------------------------------------------------------


def _horns_slope_aspect(elev: np.ndarray, cell_size: float):
    """Return (slope_deg, aspect_rad) via Horn's 3×3 method."""
    pad = np.pad(elev, 1, mode="edge")
    z1, z2, z3 = pad[:-2, :-2], pad[:-2, 1:-1], pad[:-2, 2:]
    z4, z6 = pad[1:-1, :-2], pad[1:-1, 2:]
    z7, z8, z9 = pad[2:, :-2], pad[2:, 1:-1], pad[2:, 2:]

    dzdx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8 * cell_size)
    dzdy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8 * cell_size)

    slope_deg = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
    aspect_rad = np.arctan2(-dzdx, dzdy)  # 0 = north, CW
    return slope_deg.astype("float32"), aspect_rad.astype("float32")


# ---------------------------------------------------------------------------
# TPI via cumulative-sum box filter (no scipy)
# ---------------------------------------------------------------------------


def _box_mean(arr: np.ndarray, radius: int) -> np.ndarray:
    """Fast NaN-aware box-mean using cumulative sums."""
    valid = np.where(np.isfinite(arr), 1.0, 0.0)
    filled = np.where(np.isfinite(arr), arr, 0.0).astype("float64")

    d = 2 * radius + 1
    pad_h = radius
    pad_w = radius
    filled_p = np.pad(filled, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    valid_p = np.pad(valid, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    def _cumsum_2d(a):
        cs = np.cumsum(np.cumsum(a, axis=0), axis=1)
        # Prepend zero row and column for standard integral-image indexing
        cs = np.pad(cs, ((1, 0), (1, 0)), mode="constant")
        H, W = arr.shape
        return (
            cs[d : d + H, d : d + W]
            - cs[:H, d : d + W]
            - cs[d : d + H, :W]
            + cs[:H, :W]
        )

    s = _cumsum_2d(filled_p)
    c = _cumsum_2d(valid_p)
    mean = np.where(c > 0, s / c, np.nan)
    return mean.astype("float32")


def _compute_tpi(elev: np.ndarray, radius: int) -> np.ndarray:
    return (elev - _box_mean(elev, radius)).astype("float32")


# ---------------------------------------------------------------------------
# Build terrain GeoTIFF
# ---------------------------------------------------------------------------


def build_terrain(dem_dir: str, out_tif: str) -> str:
    """Mosaic DEM tiles, resample to RTMA grid, compute terrain features."""
    tf, H, W = _pnw_rtma_transform_and_shape()
    print(f"Target grid: {W}×{H}, transform: {tf}")

    # 1) Build VRT mosaic of DEM tiles covering PNW
    tiles = sorted(
        os.path.join(dem_dir, f) for f in os.listdir(dem_dir) if f.endswith(".tif")
    )
    print(f"Found {len(tiles)} DEM tiles")

    with tempfile.NamedTemporaryFile(suffix=".vrt", delete=False) as tmp:
        vrt_path = tmp.name
    try:
        subprocess.run(
            [
                "gdalbuildvrt",
                "-te",
                str(PNW_WEST - 0.5),
                str(PNW_SOUTH - 0.5),
                str(PNW_EAST + 0.5),
                str(PNW_NORTH + 0.5),
                vrt_path,
            ]
            + tiles,
            check=True,
            capture_output=True,
        )
        print(f"VRT built: {vrt_path}")

        # 2) Resample to RTMA grid (average aggregation)
        dst_elev = np.full((H, W), np.nan, dtype="float32")
        with rasterio.open(vrt_path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_elev,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=tf,
                dst_crs=RTMA_CRS,
                resampling=Resampling.average,
                dst_nodata=np.nan,
            )
        print(f"DEM resampled: valid pixels = {np.isfinite(dst_elev).sum()}")
    finally:
        os.unlink(vrt_path)

    # 3) Compute terrain features
    # Cell size in metres (~2.5 km at ~45° lat)
    mid_lat = (PNW_SOUTH + PNW_NORTH) / 2.0
    cell_m = RTMA_RES * 111_320 * np.cos(np.radians(mid_lat))

    slope_deg, aspect_rad = _horns_slope_aspect(dst_elev, cell_m)
    aspect_sin = np.sin(aspect_rad).astype("float32")
    aspect_cos = np.cos(aspect_rad).astype("float32")
    tpi_4 = _compute_tpi(dst_elev, 4)
    tpi_10 = _compute_tpi(dst_elev, 10)

    # NaN propagation
    nan_mask = ~np.isfinite(dst_elev)
    for arr in [slope_deg, aspect_sin, aspect_cos, tpi_4, tpi_10]:
        arr[nan_mask] = np.nan

    # 4) Write 6-band GeoTIFF
    bands = [dst_elev, slope_deg, aspect_sin, aspect_cos, tpi_4, tpi_10]
    os.makedirs(os.path.dirname(out_tif) or ".", exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": len(bands),
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
        for i, (arr, name) in enumerate(zip(bands, TERRAIN_BAND_NAMES), 1):
            dst.write(arr, i)
            dst.set_band_description(i, name)
    print(f"Terrain written: {out_tif}  ({len(bands)} bands, {H}×{W})")
    return out_tif


# ---------------------------------------------------------------------------
# Build RSUN GeoTIFF (365 bands, DOY-indexed)
# ---------------------------------------------------------------------------


def build_rsun(rsun_dir: str, out_tif: str) -> str:
    """Reproject 365 RSUN DOY grids from EPSG:5070 1 km to RTMA grid."""
    tf, H, W = _pnw_rtma_transform_and_shape()

    os.makedirs(os.path.dirname(out_tif) or ".", exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 365,
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

    skipped = []
    with rasterio.open(out_tif, "w", **profile) as dst:
        for doy in range(1, 366):
            fname = f"rsun_doy_{doy:03d}.tif"
            path = os.path.join(rsun_dir, fname)
            buf = np.full((H, W), np.nan, dtype="float32")
            try:
                with rasterio.open(path) as src:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=buf,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=tf,
                        dst_crs=RTMA_CRS,
                        resampling=Resampling.bilinear,
                        src_nodata=src.nodata,
                        dst_nodata=np.nan,
                    )
            except Exception as exc:
                print(f"  WARNING: {fname} corrupt, filling with NaN: {exc}")
                skipped.append(doy)
            dst.write(buf, doy)
            dst.set_band_description(doy, f"rsun_doy_{doy:03d}")
            if doy % 50 == 0 or doy == 365:
                print(f"  rsun: {doy}/365", flush=True)
    if skipped:
        print(f"  WARNING: {len(skipped)} corrupt DOY(s) skipped: {skipped}")

    print(f"RSUN written: {out_tif}  (365 bands, {H}×{W})")
    return out_tif


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build RTMA-aligned terrain and RSUN grids for PNW."
    )
    p.add_argument(
        "--dem-dir",
        default="/data/ssd2/dads/dem/dem_30",
        help="Directory of MGRS DEM tiles (*.tif)",
    )
    p.add_argument(
        "--rsun-dir",
        default="/nas/dads/dem/rsun_1km",
        help="Directory of rsun_doy_NNN.tif files",
    )
    p.add_argument(
        "--out-dir",
        default="/nas/dads/mvp",
        help="Output directory for terrain and rsun grids",
    )
    p.add_argument(
        "--terrain-only",
        action="store_true",
        help="Build terrain grid only (skip RSUN)",
    )
    p.add_argument(
        "--rsun-only",
        action="store_true",
        help="Build RSUN grid only (skip terrain)",
    )
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if not a.rsun_only:
        build_terrain(a.dem_dir, os.path.join(a.out_dir, "terrain_pnw_rtma.tif"))
    if not a.terrain_only:
        build_rsun(a.rsun_dir, os.path.join(a.out_dir, "rsun_pnw_rtma.tif"))

    print("Done.")


if __name__ == "__main__":
    main()
