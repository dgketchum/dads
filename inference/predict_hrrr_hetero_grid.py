"""
Tiled full-domain inference for the hetero HRRR DA GNN.

Loads a precomputed station-to-grid index (built by
prep/build_hetero_inference_index.py), then for each requested day:
  1. Loads the daily HRRR background raster
  2. Filters the station-day table to that day
  3. Tiles the PNW grid into 64×64 patches
  4. Builds a HeteroData graph per tile using precomputed edge arrays
  5. Runs the model forward pass on each tile
  6. Writes a multi-band output GeoTIFF (one band per target)

Usage:
    uv run python -m inference.predict_hrrr_hetero_grid \\
        --checkpoint /nas/dads/mvp/hrrr_hetero_smoke/<ckpt>.ckpt \\
        --inference-index /nas/dads/mvp/hetero_inference_index.npz \\
        --date 2024-06-15 \\
        --out-dir /nas/dads/mvp/hrrr_hetero_da_test \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import rasterio
import torch
from pyproj import Transformer
from torch_geometric.data import HeteroData

from models.hrrr_da.hetero_dataset import (
    _RasterCache,
    _apply_norm,
    _doy_features,
)
from models.hrrr_da.lit_hetero_gnn import LitHRRRHeteroGNN
from prep.paths import MVP_ROOT
from prep.pnw_1km_grid import PNW_1KM_CRS, PNW_1KM_SHAPE, PNW_1KM_TRANSFORM

_DEFAULT_TABLE = f"{MVP_ROOT}/station_day_hrrr_pnw.parquet"
_DEFAULT_TERRAIN = f"{MVP_ROOT}/terrain_pnw_1km.tif"
_DEFAULT_BG_DIR = f"{MVP_ROOT}/hrrr_1km_pnw"
_DEFAULT_BG_PATTERN = "HRRR_1km_{date}.tif"


# ---------------------------------------------------------------------------
# Static asset loading
# ---------------------------------------------------------------------------


def _build_latlon_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(H, W) lat/lon arrays in EPSG:4326 for the PNW 1 km grid."""
    H, W = shape
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


def load_inference_assets(
    checkpoint: str,
    norm_stats_path: str,
    index_path: str,
    terrain_tif: str,
    device: torch.device,
) -> dict:
    """Load model, norm stats, full index arrays into RAM, terrain, lat/lon grid.

    Returns a dict of all static assets. Called once per run.
    """
    print("Loading norm stats…", flush=True)
    with open(norm_stats_path) as f:
        ns = json.load(f)

    print("Loading checkpoint…", flush=True)
    model = LitHRRRHeteroGNN.load_from_checkpoint(checkpoint, map_location=device)
    model.eval()
    model.to(device)

    print(f"Loading inference index from {index_path}…", flush=True)
    npz = np.load(index_path, allow_pickle=True)
    fids_all: list[str] = npz["fids"].tolist()
    fid_to_global: dict[str, int] = {f: i for i, f in enumerate(fids_all)}

    assets = {
        "model": model,
        "norm_stats": ns,
        "fids_all": fids_all,
        "fid_to_global": fid_to_global,
        "nbr_idx": npz["nbr_idx"],  # (n_pixels, max_k) int32
        "nbr_dist_km": npz["nbr_dist_km"],  # (n_pixels, max_k) float32
        "nbr_bearing_sin": npz["nbr_bearing_sin"],
        "nbr_bearing_cos": npz["nbr_bearing_cos"],
        "nbr_delta_elev": npz["nbr_delta_elev"],
        "station_radius_km": float(ns["station_radius_km"]),
    }

    print("Loading terrain raster…", flush=True)
    with rasterio.open(terrain_tif) as src:
        assets["terrain_data"] = src.read().astype("float32")  # (B, H, W)
        assets["terrain_desc"] = list(src.descriptions)

    print("Building PNW lat/lon grid…", flush=True)
    assets["lat_grid"], assets["lon_grid"] = _build_latlon_grid(PNW_1KM_SHAPE)

    print("Assets loaded.", flush=True)
    return assets


# ---------------------------------------------------------------------------
# Per-day station matrix
# ---------------------------------------------------------------------------


def build_station_day_matrix(
    station_table: pd.DataFrame,
    day: pd.Timestamp,
    norm_stats: dict,
    fids_all: list[str],
    station_feature_cols: list[str],
    station_mask_cols: list[str],
    station_node_dim: int,
    target_names: list[str],
) -> tuple[np.ndarray, set[int]]:
    """Build (N_total_stations, station_node_dim) float32 matrix for one day.

    Rows absent for the day are zero-filled.  Returns the matrix and the set
    of global station indices that are present today.
    """
    day_norm = pd.Timestamp(day).normalize()
    day_df = station_table[station_table["day"] == day_norm].copy()
    day_df["fid"] = day_df["fid"].astype(str)

    n_total = len(fids_all)
    mat = np.zeros((n_total, station_node_dim), dtype="float32")
    present: set[int] = set()

    if day_df.empty:
        return mat, present

    for _, row in day_df.iterrows():
        fid = str(row["fid"])
        gi = next((i for i, f in enumerate(fids_all) if f == fid), None)
        if gi is None:
            continue
        present.add(gi)
        feat = []
        for col in station_feature_cols:
            val = row.get(col, np.nan)
            feat.append(float(val) if pd.notna(val) else 0.0)
        for name in target_names:
            if name in station_feature_cols:
                feat.append(1.0 if pd.notna(row.get(name, np.nan)) else 0.0)
        mat[gi] = feat[:station_node_dim]

    # Apply normalization to rows that are present
    if present:
        present_idx = np.array(sorted(present), dtype=int)
        all_cols = station_feature_cols + station_mask_cols
        mat[present_idx] = _apply_norm(
            mat[present_idx], all_cols, norm_stats["station_norm_stats"]
        )

    return mat, present


# ---------------------------------------------------------------------------
# Edge builders
# ---------------------------------------------------------------------------


def build_grid_grid_edges_vectorized(
    tile_H: int,
    tile_W: int,
    tile_lat: np.ndarray,
    tile_lon: np.ndarray,
    tile_elev: np.ndarray,
    tile_ugrd: np.ndarray,
    tile_vgrd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 4-connected grid-grid edges for a tile.  Fully vectorized.

    Edge feature order (7 values):
      [dist_km/5.0, brng_sin, brng_cos, delta_elev, 0.0, uw_cos, uw_sin]
    """
    idx = np.arange(tile_H * tile_W, dtype=np.int64).reshape(tile_H, tile_W)

    # Horizontal edges (left→right and right→left)
    h_src = np.concatenate([idx[:, :-1].ravel(), idx[:, 1:].ravel()])
    h_dst = np.concatenate([idx[:, 1:].ravel(), idx[:, :-1].ravel()])
    # Vertical edges (top→bottom and bottom→top)
    v_src = np.concatenate([idx[:-1, :].ravel(), idx[1:, :].ravel()])
    v_dst = np.concatenate([idx[1:, :].ravel(), idx[:-1, :].ravel()])

    src_all = np.concatenate([h_src, v_src])
    dst_all = np.concatenate([h_dst, v_dst])

    # Distance (≈1 km for all 4-connected pairs at 1 km grid)
    s_lat_r = np.radians(tile_lat.ravel()[src_all])
    s_lon_r = np.radians(tile_lon.ravel()[src_all])
    d_lat_r = np.radians(tile_lat.ravel()[dst_all])
    d_lon_r = np.radians(tile_lon.ravel()[dst_all])

    dlat = d_lat_r - s_lat_r
    dlon = d_lon_r - s_lon_r
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(s_lat_r) * np.cos(d_lat_r) * np.sin(dlon / 2) ** 2
    )
    dist_km = 6371.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    # Bearing src→dst
    bx = np.sin(dlon) * np.cos(d_lat_r)
    by = np.cos(s_lat_r) * np.sin(d_lat_r) - np.sin(s_lat_r) * np.cos(d_lat_r) * np.cos(
        dlon
    )
    brng = np.arctan2(bx, by)
    brng_sin = np.sin(brng).astype("float32")
    brng_cos = np.cos(brng).astype("float32")

    # Delta elevation: src − dst
    flat_elev = tile_elev.ravel()
    delta_elev = (flat_elev[src_all] - flat_elev[dst_all]).astype("float32")

    # Upwind features at destination node
    flat_ugrd = tile_ugrd.ravel()
    flat_vgrd = tile_vgrd.ravel()
    u_dst = flat_ugrd[dst_all]
    v_dst = flat_vgrd[dst_all]
    theta = np.arctan2(-u_dst, -v_dst)
    uw_cos = np.cos(theta - brng).astype("float32")
    uw_sin = np.sin(theta - brng).astype("float32")

    edge_index = np.stack([src_all, dst_all]).astype(np.int64)
    edge_attr = np.column_stack(
        [
            (dist_km / 5.0).astype("float32"),
            brng_sin,
            brng_cos,
            delta_elev,
            np.zeros(len(src_all), dtype="float32"),
            uw_cos,
            uw_sin,
        ]
    )
    return edge_index, edge_attr


def build_station_grid_edges_vectorized(
    tile_flat_idx: np.ndarray,
    nbr_idx: np.ndarray,
    nbr_dist_km: np.ndarray,
    nbr_bearing_sin: np.ndarray,
    nbr_bearing_cos: np.ndarray,
    nbr_delta_elev: np.ndarray,
    present_sta_indices: set[int],
    tile_ugrd: np.ndarray,
    tile_vgrd: np.ndarray,
    station_radius_km: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build station→grid edges for one tile from precomputed index.

    Returns:
      sg_edge_index  (2, E) int64  — row 0: local station idx, row 1: local grid idx
      sg_edge_attr   (E, 7) float32
      active_sta_global  (A,) int64 — global station indices present and connected
    """
    # Slice precomputed index for this tile's pixels
    tile_nbr_idx = nbr_idx[tile_flat_idx]  # (n_tile, max_k)
    tile_dist_km = nbr_dist_km[tile_flat_idx]  # (n_tile, max_k)
    tile_brng_sin = nbr_bearing_sin[tile_flat_idx]  # (n_tile, max_k)
    tile_brng_cos = nbr_bearing_cos[tile_flat_idx]  # (n_tile, max_k)
    tile_delta_elev = nbr_delta_elev[tile_flat_idx]  # (n_tile, max_k)

    n_tile, max_k = tile_nbr_idx.shape

    # Flatten to (n_tile * max_k,)
    flat_sta_global = tile_nbr_idx.ravel()
    flat_grid_local = np.repeat(np.arange(n_tile, dtype=np.int64), max_k)
    flat_dist = tile_dist_km.ravel()
    flat_bs = tile_brng_sin.ravel()
    flat_bc = tile_brng_cos.ravel()
    flat_de = tile_delta_elev.ravel()

    # Mask: valid index AND station present today
    valid = flat_sta_global >= 0
    valid &= np.array([gi in present_sta_indices for gi in flat_sta_global], dtype=bool)

    if not valid.any():
        empty = np.zeros((2, 0), dtype=np.int64)
        return empty, np.zeros((0, 7), dtype="float32"), np.array([], dtype=np.int64)

    flat_sta_global = flat_sta_global[valid]
    flat_grid_local = flat_grid_local[valid]
    flat_dist = flat_dist[valid]
    flat_bs = flat_bs[valid]
    flat_bc = flat_bc[valid]
    flat_de = flat_de[valid]

    # Remap global station indices → local (contiguous) station indices
    active_global = np.unique(flat_sta_global)
    global_to_local = {g: loc for loc, g in enumerate(active_global)}
    flat_sta_local = np.array(
        [global_to_local[g] for g in flat_sta_global], dtype=np.int64
    )

    # Dynamic wind-alignment features (upwind at destination grid node)
    flat_ugrd = tile_ugrd.ravel()[flat_grid_local]
    flat_vgrd = tile_vgrd.ravel()[flat_grid_local]
    brng = np.arctan2(flat_bs, flat_bc)
    theta = np.arctan2(-flat_ugrd.astype("float64"), -flat_vgrd.astype("float64"))
    uw_cos = np.cos(theta - brng).astype("float32")
    uw_sin = np.sin(theta - brng).astype("float32")

    sg_edge_index = np.stack([flat_sta_local, flat_grid_local])  # (2, E)
    sg_edge_attr = np.column_stack(
        [
            (flat_dist / station_radius_km).astype("float32"),
            flat_bs.astype("float32"),
            flat_bc.astype("float32"),
            flat_de.astype("float32"),
            np.zeros(len(flat_dist), dtype="float32"),
            uw_cos,
            uw_sin,
        ]
    )
    return sg_edge_index, sg_edge_attr, active_global.astype(np.int64)


# ---------------------------------------------------------------------------
# Per-day inference
# ---------------------------------------------------------------------------


def predict_day_hetero(
    day: pd.Timestamp,
    model: LitHRRRHeteroGNN,
    assets: dict,
    station_table: pd.DataFrame,
    background_dir: str,
    background_pattern: str,
    out_dir: str,
    device: torch.device,
    tile_size: int = 64,
) -> str | None:
    """Run full-domain inference for one day; return output path or None."""
    H, W = PNW_1KM_SHAPE
    date_str = day.strftime("%Y%m%d")
    ns = assets["norm_stats"]
    target_names: list[str] = ns["target_names"]
    grid_feature_names: list[str] = ns["grid_feature_names"]
    background_feature_names: list[str] = ns["background_feature_names"]
    terrain_feature_names: list[str] = ns["terrain_feature_names"]
    station_feature_cols: list[str] = ns["station_feature_cols"]
    station_mask_cols: list[str] = ns["station_mask_cols"]
    station_node_dim: int = int(ns["station_node_dim"])
    station_radius_km: float = assets["station_radius_km"]
    grid_norm: dict = ns["grid_norm_stats"]
    n_targets = len(target_names)

    # Background band indices (from background raster only)
    bg_band_map = {name: i for i, name in enumerate(background_feature_names)}
    ugrd_bg_idx = bg_band_map.get("ugrd_hrrr")
    vgrd_bg_idx = bg_band_map.get("vgrd_hrrr")

    # Terrain band index for elevation
    ter_band_map = {name: i for i, name in enumerate(terrain_feature_names)}
    elev_ter_idx = ter_band_map.get("elevation", 0)

    # Load background raster
    bg_path = os.path.join(background_dir, background_pattern.format(date=date_str))
    if not os.path.exists(bg_path):
        print(f"  SKIP {date_str}: no background raster at {bg_path}", flush=True)
        return None

    raster_cache = _RasterCache(max_items=2)
    bg = raster_cache.get(bg_path)
    bg_data = bg["data"]  # (n_bg_bands, H, W)

    terrain_data = assets["terrain_data"]  # (n_ter_bands, H, W)
    lat_grid = assets["lat_grid"]  # (H, W)
    lon_grid = assets["lon_grid"]  # (H, W)

    # Precompute DOY features (scalar, same for all pixels)
    doy_sin, doy_cos = _doy_features(day)

    # Build station-day matrix
    sta_mat, present_sta = build_station_day_matrix(
        station_table=station_table,
        day=day,
        norm_stats=ns,
        fids_all=assets["fids_all"],
        station_feature_cols=station_feature_cols,
        station_mask_cols=station_mask_cols,
        station_node_dim=station_node_dim,
        target_names=target_names,
    )

    # Output array
    out_arr = np.full((n_targets, H, W), np.nan, dtype="float32")

    n_rows = (H + tile_size - 1) // tile_size
    n_cols = (W + tile_size - 1) // tile_size
    tile_count = 0
    t0 = time.time()

    for tr in range(n_rows):
        for tc in range(n_cols):
            r0 = tr * tile_size
            c0 = tc * tile_size
            r1 = min(r0 + tile_size, H)
            c1 = min(c0 + tile_size, W)
            th = r1 - r0
            tw = c1 - c0
            n_tile = th * tw

            # Flat pixel indices (C-order into the full H×W grid)
            rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
            tile_flat_idx = (rr.ravel() * W + cc.ravel()).astype(np.int64)

            # Assemble grid node features
            # Background features
            bg_tile = (
                bg_data[:, r0:r1, c0:c1].reshape(len(background_feature_names), -1).T
            )
            # Terrain features
            ter_tile = (
                terrain_data[:, r0:r1, c0:c1].reshape(len(terrain_feature_names), -1).T
            )
            # DOY + lat/lon
            temp_tile = np.full((n_tile, 2), [doy_sin, doy_cos], dtype="float32")
            lat_tile = lat_grid[r0:r1, c0:c1].ravel()[:, np.newaxis]
            lon_tile = lon_grid[r0:r1, c0:c1].ravel()[:, np.newaxis]

            grid_x_raw = np.concatenate(
                [bg_tile, ter_tile, temp_tile, lat_tile, lon_tile], axis=1
            ).astype("float32")
            grid_x_raw = np.nan_to_num(grid_x_raw, nan=0.0)
            grid_x = _apply_norm(grid_x_raw, grid_feature_names, grid_norm)

            # Wind components for edge feature construction (raw, unnormalized)
            if ugrd_bg_idx is not None and vgrd_bg_idx is not None:
                tile_ugrd = bg_data[ugrd_bg_idx, r0:r1, c0:c1]
                tile_vgrd = bg_data[vgrd_bg_idx, r0:r1, c0:c1]
            else:
                tile_ugrd = np.zeros((th, tw), dtype="float32")
                tile_vgrd = np.zeros((th, tw), dtype="float32")

            tile_elev = terrain_data[elev_ter_idx, r0:r1, c0:c1]

            # Grid-grid edges
            gg_edge_index, gg_edge_attr = build_grid_grid_edges_vectorized(
                th,
                tw,
                lat_grid[r0:r1, c0:c1],
                lon_grid[r0:r1, c0:c1],
                tile_elev,
                tile_ugrd,
                tile_vgrd,
            )

            # Station-grid edges from precomputed index
            sg_edge_index, sg_edge_attr, active_global = (
                build_station_grid_edges_vectorized(
                    tile_flat_idx,
                    assets["nbr_idx"],
                    assets["nbr_dist_km"],
                    assets["nbr_bearing_sin"],
                    assets["nbr_bearing_cos"],
                    assets["nbr_delta_elev"],
                    present_sta,
                    tile_ugrd,
                    tile_vgrd,
                    station_radius_km,
                )
            )

            # Station node features for active stations only
            if len(active_global) > 0:
                local_sta_x = sta_mat[active_global]  # (A, station_node_dim)
            else:
                local_sta_x = np.zeros((0, station_node_dim), dtype="float32")

            # Build HeteroData
            data = HeteroData()
            data["grid"].x = torch.from_numpy(grid_x).to(device)
            data["station"].x = torch.from_numpy(local_sta_x).to(device)
            data["grid", "neighbors", "grid"].edge_index = torch.from_numpy(
                gg_edge_index
            ).to(device)
            data["grid", "neighbors", "grid"].edge_attr = torch.from_numpy(
                gg_edge_attr
            ).to(device)
            data["station", "influences", "grid"].edge_index = torch.from_numpy(
                sg_edge_index
            ).to(device)
            data["station", "influences", "grid"].edge_attr = torch.from_numpy(
                sg_edge_attr
            ).to(device)

            with torch.no_grad():
                pred = model(data)  # (n_tile, n_targets)

            pred_np = pred.cpu().numpy().reshape(th, tw, n_targets)
            for ti, _name in enumerate(target_names):
                out_arr[ti, r0:r1, c0:c1] = pred_np[:, :, ti]

            tile_count += 1

    dt = time.time() - t0
    print(f"  {date_str}: {tile_count} tiles in {dt:.1f}s", flush=True)

    # Write output GeoTIFF
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"HRRR_hetero_da_{date_str}.tif")
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": n_targets,
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
        for ti, name in enumerate(target_names):
            dst.write(out_arr[ti], ti + 1)
            dst.set_band_description(ti + 1, name)

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiled hetero DA grid inference.")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument(
        "--norm-stats",
        default=None,
        help="Path to norm_stats.json (default: <ckpt_dir>/norm_stats.json)",
    )
    p.add_argument(
        "--inference-index", required=True, help="Path to hetero_inference_index.npz"
    )
    p.add_argument("--station-table", default=_DEFAULT_TABLE)
    p.add_argument("--terrain-tif", default=_DEFAULT_TERRAIN)
    p.add_argument("--background-dir", default=_DEFAULT_BG_DIR)
    p.add_argument("--background-pattern", default=_DEFAULT_BG_PATTERN)
    p.add_argument("--out-dir", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", help="Single date YYYY-MM-DD")
    g.add_argument("--start", help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (inclusive)")
    p.add_argument("--tile-size", type=int, default=64)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    norm_stats_path = args.norm_stats or os.path.join(
        os.path.dirname(args.checkpoint), "norm_stats.json"
    )
    device = torch.device(args.device)

    assets = load_inference_assets(
        checkpoint=args.checkpoint,
        norm_stats_path=norm_stats_path,
        index_path=args.inference_index,
        terrain_tif=args.terrain_tif,
        device=device,
    )

    print("Loading station table…", flush=True)
    station_table = pd.read_parquet(args.station_table)
    if isinstance(station_table.index, pd.MultiIndex):
        station_table = station_table.reset_index()
    station_table["fid"] = station_table["fid"].astype(str)
    station_table["day"] = pd.to_datetime(station_table["day"]).dt.normalize()

    if args.date:
        dates = [pd.Timestamp(args.date)]
    else:
        start = pd.Timestamp(args.start)
        end = pd.Timestamp(args.end) if args.end else start
        dates = pd.date_range(start, end, freq="D").tolist()

    model = assets["model"]
    os.makedirs(args.out_dir, exist_ok=True)

    for day in dates:
        out = predict_day_hetero(
            day=day,
            model=model,
            assets=assets,
            station_table=station_table,
            background_dir=args.background_dir,
            background_pattern=args.background_pattern,
            out_dir=args.out_dir,
            device=device,
            tile_size=args.tile_size,
        )
        if out:
            print(f"  → {out}", flush=True)


if __name__ == "__main__":
    main()
