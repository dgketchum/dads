"""Feather MGRS tile-boundary seams in static 1 km PNW rasters.

Algorithm:
1. Build an MGRS tile-ID raster from DEM tile extents
2. Derive a boundary mask (4-connected neighbor has different tile ID)
3. Compute distance-to-boundary → linear blend weights (0 at seam, 1 at buffer edge)
4. For each band: feathered = alpha * original + (1 - alpha) * gaussian_blur(original)

Usage:
    uv run python -m prep.feather_raster_seams \
        --dem-dir /data/ssd2/dads/dem/dem_30 \
        --out-dir /nas/dads/mvp \
        --sigma 3 --buffer 5
"""

from __future__ import annotations

import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt, gaussian_filter

from prep.pnw_1km_grid import (
    PNW_1KM_CRS,
    PNW_1KM_SHAPE,
    PNW_1KM_TRANSFORM,
)

# Static rasters to feather (basename → band count read at runtime)
_RASTERS = [
    "terrain_pnw_1km.tif",
    "rsun_pnw_1km.tif",
    "landsat_pnw_1km.tif",
]


def _build_tile_id_raster(dem_dir: str) -> np.ndarray:
    """Rasterize DEM tile footprints as unique integer IDs onto the 1 km grid."""
    tiles = sorted(glob(os.path.join(dem_dir, "*.tif")))
    if not tiles:
        raise FileNotFoundError(f"No .tif files in {dem_dir}")

    transformer = Transformer.from_crs("EPSG:4326", PNW_1KM_CRS, always_xy=True)
    shapes = []
    for idx, tif in enumerate(tiles, start=1):
        with rasterio.open(tif) as src:
            b = src.bounds  # EPSG:4326
        # Project 4 corners → EPSG:5070, build polygon
        xs = [b.left, b.right, b.right, b.left, b.left]
        ys = [b.bottom, b.bottom, b.top, b.top, b.bottom]
        px, py = transformer.transform(xs, ys)
        ring = list(zip(px, py))
        geom = {"type": "Polygon", "coordinates": [ring]}
        shapes.append((geom, idx))

    tile_ids = rasterize(
        shapes,
        out_shape=PNW_1KM_SHAPE,
        transform=PNW_1KM_TRANSFORM,
        fill=0,
        dtype="int32",
    )
    print(
        f"  Rasterized {len(tiles)} DEM tiles, "
        f"{np.count_nonzero(tile_ids)} pixels covered"
    )
    return tile_ids


def _boundary_blend_weights(tile_ids: np.ndarray, buffer: int) -> np.ndarray:
    """Build blend weights: 0 at tile boundaries, 1 beyond buffer distance."""
    # 4-connected boundary: pixel where any neighbor has a different tile ID
    padded = np.pad(tile_ids, 1, mode="edge")
    boundary = np.zeros(tile_ids.shape, dtype=bool)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = padded[
            1 + dr : 1 + dr + tile_ids.shape[0], 1 + dc : 1 + dc + tile_ids.shape[1]
        ]
        boundary |= (tile_ids != shifted) & (tile_ids > 0) & (shifted > 0)

    dist = distance_transform_edt(~boundary)
    alpha = np.clip(dist / buffer, 0.0, 1.0).astype(np.float32)
    n_boundary = int(boundary.sum())
    print(f"  Boundary pixels: {n_boundary:,}, buffer: {buffer} px")
    return alpha


def _feather_band(band: np.ndarray, alpha: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian feathering in the boundary zone, preserving NaN."""
    nan_mask = np.isnan(band)
    filled = np.where(nan_mask, 0.0, band)
    blurred = gaussian_filter(filled, sigma=sigma)
    feathered = alpha * filled + (1.0 - alpha) * blurred
    feathered[nan_mask] = np.nan
    return feathered


def feather_rasters(
    dem_dir: str,
    out_dir: str,
    sigma: float = 3.0,
    buffer: int = 5,
) -> list[str]:
    """Feather tile-boundary seams in all static rasters. Returns output paths."""
    print("Building MGRS tile-ID raster...")
    tile_ids = _build_tile_id_raster(dem_dir)

    print("Computing boundary blend weights...")
    alpha = _boundary_blend_weights(tile_ids, buffer)

    raster_dir = Path(out_dir)
    outputs = []
    for name in _RASTERS:
        src_path = raster_dir / name
        dst_name = name.replace(".tif", "_feathered.tif")
        dst_path = raster_dir / dst_name

        if not src_path.exists():
            print(f"  SKIP {name} (not found)")
            continue

        print(f"Feathering {name}...")
        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
            nbands = src.count
            descriptions = src.descriptions

            profile.update(dtype="float32")
            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.descriptions = descriptions
                for b in range(1, nbands + 1):
                    data = src.read(b).astype(np.float32)
                    feathered = _feather_band(data, alpha, sigma)
                    dst.write(feathered, b)
                    if b % 50 == 0 or b == nbands:
                        print(f"    band {b}/{nbands}")

        print(f"  → {dst_path}")
        outputs.append(str(dst_path))

    return outputs


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feather MGRS tile-boundary seams")
    p.add_argument(
        "--dem-dir",
        default="/data/ssd2/dads/dem/dem_30",
        help="Directory of DEM tiles (for tile footprints)",
    )
    p.add_argument(
        "--out-dir",
        default="/nas/dads/mvp",
        help="Directory containing static rasters (output written here)",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Gaussian blur sigma in pixels (default: 3)",
    )
    p.add_argument(
        "--buffer",
        type=int,
        default=5,
        help="Linear ramp distance in pixels (default: 5)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    feather_rasters(args.dem_dir, args.out_dir, args.sigma, args.buffer)
