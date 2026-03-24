"""
Precompute the static station-to-grid influence structure for hetero DA inference.

For every pixel in the PNW 1 km grid, stores the indices and static edge
features of up to max_k nearest stations within radius_km.  The result is a
single .npz that inference loads once into RAM, replacing per-pixel BallTree
queries and Python edge-building loops.

Output arrays (shape n_pixels × max_k, C-order flat pixel index):
  nbr_idx          int32   station index into fids; -1 = no neighbor
  nbr_dist_km      float32 haversine distance in km
  nbr_bearing_sin  float32 sin of bearing FROM station TO pixel
  nbr_bearing_cos  float32 cos of bearing
  nbr_delta_elev   float32 terrain[station] − terrain[pixel] (metres)
  fids             object  (N_stations,) station fid strings

Metadata saved alongside as <out_path>_meta.json.

Usage:
    uv run python -m prep.build_hetero_inference_index \\
        --out-path /nas/dads/mvp/hetero_inference_index.npz
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol
from sklearn.neighbors import BallTree

from prep.paths import MVP_ROOT
from prep.pnw_1km_grid import PNW_1KM_CRS, PNW_1KM_SHAPE, PNW_1KM_TRANSFORM

_DEFAULT_TABLE = f"{MVP_ROOT}/station_day_hrrr_pnw.parquet"
_DEFAULT_TERRAIN = f"{MVP_ROOT}/terrain_pnw_1km.tif"
_DEFAULT_OUT = f"{MVP_ROOT}/hetero_inference_index.npz"


# ---------------------------------------------------------------------------
# Grid lat/lon
# ---------------------------------------------------------------------------


def _build_pnw_latlon_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return (H, W) lat and lon arrays in EPSG:4326."""
    H, W = PNW_1KM_SHAPE
    tf = PNW_1KM_TRANSFORM
    cols = np.arange(W, dtype="float64") + 0.5
    rows = np.arange(H, dtype="float64") + 0.5
    x5070 = tf.c + cols * tf.a
    y5070 = tf.f + rows * tf.e
    xx, yy = np.meshgrid(x5070, y5070)
    proj = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    lon, lat = proj.transform(xx.ravel(), yy.ravel())
    lat = lat.reshape(H, W).astype("float32")
    lon = lon.reshape(H, W).astype("float32")
    return lat, lon


# ---------------------------------------------------------------------------
# Station loading
# ---------------------------------------------------------------------------


def _load_unique_stations(
    table_path: str, terrain_tif: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return lats, lons, elev (float32), coords_rad (radians), fids."""
    df = pd.read_parquet(table_path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["fid"] = df["fid"].astype(str)
    station_df = (
        df.groupby("fid")[["latitude", "longitude", "elevation"]].first().reset_index()
    )

    fids = station_df["fid"].tolist()
    lats = station_df["latitude"].to_numpy(dtype="float32")
    lons = station_df["longitude"].to_numpy(dtype="float32")

    proj_to_5070 = Transformer.from_crs("EPSG:4326", str(PNW_1KM_CRS), always_xy=True)
    x5070, y5070 = proj_to_5070.transform(
        lons.astype("float64"), lats.astype("float64")
    )

    with rasterio.open(terrain_tif) as src:
        t_tf = src.transform
        t_data = src.read(1).astype("float32")
        t_H, t_W = t_data.shape
        t_crs = src.crs

    if str(t_crs) != str(PNW_1KM_CRS):
        proj_to_t = Transformer.from_crs(str(PNW_1KM_CRS), str(t_crs), always_xy=True)
        tx, ty = proj_to_t.transform(x5070, y5070)
    else:
        tx, ty = x5070, y5070

    t_rows, t_cols = rowcol(t_tf, tx, ty)
    t_rows = np.clip(np.asarray(t_rows, dtype=int), 0, t_H - 1)
    t_cols = np.clip(np.asarray(t_cols, dtype=int), 0, t_W - 1)
    sta_elev = t_data[t_rows, t_cols]

    coords_rad = np.radians(np.column_stack([lats, lons]))
    return lats, lons, sta_elev, coords_rad, fids


# ---------------------------------------------------------------------------
# Chunk builder
# ---------------------------------------------------------------------------


def _build_index_chunk(
    pix_lats_rad: np.ndarray,
    pix_lons_rad: np.ndarray,
    pix_elevs: np.ndarray,
    sta_coords_rad: np.ndarray,
    sta_elevs: np.ndarray,
    tree: BallTree,
    max_k: int,
    radius_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Query BallTree for one pixel chunk; return five (chunk, max_k) arrays."""
    pix_coords = np.column_stack([pix_lats_rad, pix_lons_rad])
    dists_rad, idx = tree.query(pix_coords, k=max_k)

    valid = dists_rad <= radius_rad
    dist_km = (dists_rad * 6371.0).astype("float32")

    # Bearing FROM station TO pixel, vectorized over (chunk, max_k)
    s_lat_r = sta_coords_rad[idx, 0]  # (chunk, max_k)
    s_lon_r = sta_coords_rad[idx, 1]
    p_lat_r = pix_lats_rad[:, np.newaxis]  # (chunk, 1) broadcasts to (chunk, max_k)
    p_lon_r = pix_lons_rad[:, np.newaxis]

    dlon = p_lon_r - s_lon_r
    bx = np.sin(dlon) * np.cos(p_lat_r)
    by = np.cos(s_lat_r) * np.sin(p_lat_r) - np.sin(s_lat_r) * np.cos(p_lat_r) * np.cos(
        dlon
    )
    brng = np.arctan2(bx, by)
    brng_sin = np.sin(brng).astype("float32")
    brng_cos = np.cos(brng).astype("float32")

    # Delta elevation: station_elev − pixel_elev
    sta_e = sta_elevs[idx]  # (chunk, max_k)
    pix_e = pix_elevs[:, np.newaxis]
    delta_elev = (sta_e - pix_e).astype("float32")

    # Sentinel values for out-of-radius entries
    idx_out = idx.astype("int32")
    idx_out[~valid] = -1
    dist_km[~valid] = 0.0
    brng_sin[~valid] = 0.0
    brng_cos[~valid] = 1.0
    delta_elev[~valid] = 0.0

    return idx_out, dist_km, brng_sin, brng_cos, delta_elev


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_hetero_inference_index(
    table_path: str = _DEFAULT_TABLE,
    terrain_tif: str = _DEFAULT_TERRAIN,
    radius_km: float = 150.0,
    max_k: int = 16,
    out_path: str = _DEFAULT_OUT,
    chunk_size: int = 50_000,
) -> None:
    t0 = time.time()
    H, W = PNW_1KM_SHAPE
    n_pixels = H * W
    radius_rad = radius_km / 6371.0

    print(f"Building PNW lat/lon grid ({H}×{W} = {n_pixels:,} pixels)…", flush=True)
    lat_grid, lon_grid = _build_pnw_latlon_grid()
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()

    print("Loading terrain raster for pixel elevations…", flush=True)
    with rasterio.open(terrain_tif) as src:
        pix_elev = src.read(1).astype("float32").ravel()

    print(f"Loading unique stations from {table_path}…", flush=True)
    _sta_lats, _sta_lons, sta_elevs, sta_coords_rad, fids = _load_unique_stations(
        table_path, terrain_tif
    )
    n_stations = len(fids)
    print(f"  {n_stations} unique stations", flush=True)

    tree = BallTree(sta_coords_rad, metric="haversine")

    # Pre-allocate output arrays
    nbr_idx = np.full((n_pixels, max_k), -1, dtype="int32")
    nbr_dist_km = np.zeros((n_pixels, max_k), dtype="float32")
    nbr_bearing_sin = np.zeros((n_pixels, max_k), dtype="float32")
    nbr_bearing_cos = np.ones((n_pixels, max_k), dtype="float32")
    nbr_delta_elev = np.zeros((n_pixels, max_k), dtype="float32")

    n_chunks = (n_pixels + chunk_size - 1) // chunk_size
    pix_lats_rad = np.radians(lat_flat)
    pix_lons_rad = np.radians(lon_flat)

    print(f"Processing {n_chunks} chunks of {chunk_size:,} pixels…", flush=True)
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_pixels)

        idx_c, dist_c, bs_c, bc_c, de_c = _build_index_chunk(
            pix_lats_rad[start:end],
            pix_lons_rad[start:end],
            pix_elev[start:end],
            sta_coords_rad,
            sta_elevs,
            tree,
            max_k,
            radius_rad,
        )
        nbr_idx[start:end] = idx_c
        nbr_dist_km[start:end] = dist_c
        nbr_bearing_sin[start:end] = bs_c
        nbr_bearing_cos[start:end] = bc_c
        nbr_delta_elev[start:end] = de_c

        if (i + 1) % 5 == 0 or i == n_chunks - 1:
            elapsed = time.time() - t0
            print(f"  chunk {i + 1}/{n_chunks}  {elapsed:.0f}s", flush=True)

    print(f"Saving index to {out_path}…", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        nbr_idx=nbr_idx,
        nbr_dist_km=nbr_dist_km,
        nbr_bearing_sin=nbr_bearing_sin,
        nbr_bearing_cos=nbr_bearing_cos,
        nbr_delta_elev=nbr_delta_elev,
        fids=np.array(fids, dtype=object),
    )

    meta = {
        "radius_km": radius_km,
        "max_k": max_k,
        "n_pixels": n_pixels,
        "n_stations": n_stations,
        "grid_H": H,
        "grid_W": W,
        "build_elapsed_s": round(time.time() - t0, 1),
    }
    meta_path = out_path.replace(".npz", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Elapsed: {time.time() - t0:.0f}s. Meta: {meta_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute hetero inference index.")
    p.add_argument("--station-table", default=_DEFAULT_TABLE)
    p.add_argument("--terrain-tif", default=_DEFAULT_TERRAIN)
    p.add_argument("--radius-km", type=float, default=150.0)
    p.add_argument("--max-k", type=int, default=16)
    p.add_argument("--out-path", default=_DEFAULT_OUT)
    p.add_argument("--chunk-size", type=int, default=50_000)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_hetero_inference_index(
        table_path=args.station_table,
        terrain_tif=args.terrain_tif,
        radius_km=args.radius_km,
        max_k=args.max_k,
        out_path=args.out_path,
        chunk_size=args.chunk_size,
    )
