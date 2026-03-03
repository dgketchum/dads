"""
Grid inference for the tmax GNN — produce corrected tmax GeoTIFFs.

Loads the trained GNN checkpoint, assembles 29 node features at every
1 km grid pixel from URMA + terrain + rsun + landsat rasters, tiles
the grid into 256x256 patches, builds per-tile graphs with grid→station
edges, and runs the model to predict delta_tmax corrections.

Output: 2-band GeoTIFF per day (band 1 = delta_tmax, band 2 = corrected_tmax).

Usage:
    uv run python -m inference.predict_tmax_grid \
        --date 2024-07-15 \
        --checkpoint /nas/dads/mvp/e7_tmax_multiyear/ckpt-epoch=011-val_loss=0.7953.ckpt \
        --out-dir /nas/dads/mvp/tmax_corrected_pnw_2024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import rasterio
import torch
from pyproj import Transformer
from sklearn.neighbors import BallTree

from models.rtma_bias.lit_scalar_gnn import LitScalarGNN
from models.rtma_bias.patch_dataset import _date_to_period
from models.wind_bias.wind_dataset import (
    build_knn_map,
    build_static_edge_attrs,
)
from prep.pnw_1km_grid import (
    PNW_1KM_CRS,
    PNW_1KM_SHAPE,
    PNW_1KM_TRANSFORM,
)
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# URMA COG band mapping
# ---------------------------------------------------------------------------
# URMA 1km COG bands (0-indexed):
#   0: tmp_c, 1: tmax_c, 2: tmin_c, 3: dpt_c, 4: ugrd, 5: vgrd,
#   6: gust, 7: spfh, 8: pres_kpa, 9: tcdc_pct, 10: ea_kpa
#
# Training features (URMA_WEATHER_COLS, 11):
#   tmp_urma, tmax_urma, tmin_urma, pres_urma, ugrd_urma, vgrd_urma,
#   wind_urma, tcdc_urma, gust_urma, ea_urma, wdir_urma

URMA_BAND_MAP = {
    "tmp_urma": 0,
    "tmax_urma": 1,
    "tmin_urma": 2,
    "pres_urma": 8,
    "ugrd_urma": 4,
    "vgrd_urma": 5,
    # wind_urma: derived sqrt(ugrd² + vgrd²)
    "tcdc_urma": 9,
    "gust_urma": 6,
    "ea_urma": 10,
    # wdir_urma: derived
}

FEATURE_COLS = [
    "tmp_urma",
    "tmax_urma",
    "tmin_urma",
    "pres_urma",
    "ugrd_urma",
    "vgrd_urma",
    "wind_urma",
    "tcdc_urma",
    "gust_urma",
    "ea_urma",
    "wdir_urma",
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "tpi_4",
    "tpi_10",
    "rsun",
    "ls_b2",
    "ls_b3",
    "ls_b4",
    "ls_b5",
    "ls_b6",
    "ls_b7",
    "ls_b10",
    "doy_sin",
    "doy_cos",
    "latitude",
    "longitude",
]


# ---------------------------------------------------------------------------
# Static raster loading
# ---------------------------------------------------------------------------


def _load_raster(path: str, label: str) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        data = src.read().astype("float32")
        tf = np.array(src.transform, dtype="float64")
    print(f"{label} loaded: {data.shape} from {path}", flush=True)
    return data, tf


def _build_latlon_grid(
    shape: tuple[int, int], transform: rasterio.Affine
) -> tuple[np.ndarray, np.ndarray]:
    """Compute lat/lon (EPSG:4326) for every pixel center in the 5070 grid."""
    H, W = shape

    cols = np.arange(W, dtype="float64") + 0.5
    rows = np.arange(H, dtype="float64") + 0.5
    x5070 = transform.c + cols * transform.a  # (W,)
    y5070 = transform.f + rows * transform.e  # (H,)
    xx, yy = np.meshgrid(x5070, y5070)  # (H, W) each

    proj = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    lon, lat = proj.transform(xx.ravel(), yy.ravel())
    lat = lat.reshape(H, W).astype("float32")
    lon = lon.reshape(H, W).astype("float32")
    return lat, lon


# ---------------------------------------------------------------------------
# Station infrastructure
# ---------------------------------------------------------------------------


def _load_station_infrastructure(
    stations_csv: str,
    meta: dict,
    terrain_data: np.ndarray,
    terrain_tf: np.ndarray,
    rsun_data: np.ndarray,
    rsun_tf: np.ndarray,
    landsat_data: np.ndarray,
    landsat_tf: np.ndarray,
) -> dict:
    """Load stations, project coords, build BallTree, extract static features."""
    stations = pd.read_csv(stations_csv)
    id_col = "station_id" if "station_id" in stations.columns else "fid"
    stations[id_col] = stations[id_col].astype(str)

    # Filter to PNW region (rough bounding box in lat/lon)
    stations = stations[
        (stations["latitude"] >= 41.5)
        & (stations["latitude"] <= 49.5)
        & (stations["longitude"] >= -125.5)
        & (stations["longitude"] <= -103.5)
    ].copy()

    fids = stations[id_col].values
    lats = stations["latitude"].values
    lons = stations["longitude"].values
    elevs = stations["elevation"].values

    # Project to EPSG:5070
    proj = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x5070, y5070 = proj.transform(lons, lats)

    # Build BallTree on lat/lon (radians) for grid→station queries
    coords_rad = np.radians(np.column_stack([lats, lons]))
    tree = BallTree(coords_rad, metric="haversine")

    # Extract per-station terrain features
    a_t, _, c_t, _, e_t, f_t = terrain_tf[:6]
    sta_terrain = np.zeros((len(fids), 6), dtype="float32")
    for i in range(len(fids)):
        col = int((x5070[i] - c_t) / a_t)
        row = int((y5070[i] - f_t) / e_t)
        _, tH, tW = terrain_data.shape
        if 0 <= row < tH and 0 <= col < tW:
            sta_terrain[i] = terrain_data[:, row, col]

    # Build station-station knn_map and static_edges
    tmp_csv = "/tmp/_grid_inference_stations.csv"
    stations.to_csv(tmp_csv, index=False)
    k = meta.get("k", 16)
    max_radius_km = meta.get("max_radius_km", 150.0)
    knn_map = build_knn_map(tmp_csv, k=k, max_radius_km=max_radius_km)
    static_edges = build_static_edge_attrs(tmp_csv, knn_map)
    os.remove(tmp_csv)

    return {
        "fids": fids,
        "lats": lats,
        "lons": lons,
        "elevs": elevs,
        "x5070": x5070,
        "y5070": y5070,
        "coords_rad": coords_rad,
        "tree": tree,
        "terrain": sta_terrain,
        "knn_map": knn_map,
        "static_edges": static_edges,
    }


def _extract_station_features(
    sta: dict,
    urma_data: np.ndarray,
    urma_tf: np.ndarray,
    terrain_data: np.ndarray,
    terrain_tf: np.ndarray,
    rsun_data: np.ndarray,
    rsun_tf: np.ndarray,
    landsat_data: np.ndarray,
    landsat_tf: np.ndarray,
    day: pd.Timestamp,
) -> np.ndarray:
    """Extract 29 features for all stations from rasters (same as grid pixels)."""
    n = len(sta["fids"])
    x5070 = sta["x5070"]
    y5070 = sta["y5070"]

    # URMA extraction
    a_u, _, c_u, _, e_u, f_u = urma_tf[:6]
    cols_u = ((x5070 - c_u) / a_u).astype(int)
    rows_u = ((y5070 - f_u) / e_u).astype(int)
    _, uH, uW = urma_data.shape
    valid_u = (rows_u >= 0) & (rows_u < uH) & (cols_u >= 0) & (cols_u < uW)

    urma_feats = np.zeros((n, 11), dtype="float32")
    for i, (col_name, band_idx) in enumerate(URMA_BAND_MAP.items()):
        feat_idx = FEATURE_COLS.index(col_name)
        vals = np.where(
            valid_u,
            urma_data[band_idx, np.clip(rows_u, 0, uH - 1), np.clip(cols_u, 0, uW - 1)],
            0.0,
        )
        urma_feats[:, feat_idx] = vals

    # Derived: wind_urma, wdir_urma
    ugrd = urma_feats[:, 4]  # ugrd_urma
    vgrd = urma_feats[:, 5]  # vgrd_urma
    urma_feats[:, 6] = np.sqrt(ugrd**2 + vgrd**2)  # wind_urma
    wdir = (np.degrees(np.arctan2(-ugrd, -vgrd)) + 360) % 360
    urma_feats[:, 10] = wdir  # wdir_urma

    # Terrain (already extracted at init)
    terrain_feats = sta["terrain"]  # (n, 6)

    # RSUN
    doy_idx = min(day.dayofyear - 1, rsun_data.shape[0] - 1)
    a_r, _, c_r, _, e_r, f_r = rsun_tf[:6]
    cols_r = ((x5070 - c_r) / a_r).astype(int)
    rows_r = ((y5070 - f_r) / e_r).astype(int)
    _, rH, rW = rsun_data.shape
    valid_r = (rows_r >= 0) & (rows_r < rH) & (cols_r >= 0) & (cols_r < rW)
    rsun_feats = (
        np.where(
            valid_r,
            rsun_data[doy_idx, np.clip(rows_r, 0, rH - 1), np.clip(cols_r, 0, rW - 1)],
            0.0,
        )
        .reshape(-1, 1)
        .astype("float32")
    )

    # Landsat
    period = _date_to_period(day)
    b_start = period * 7
    a_l, _, c_l, _, e_l, f_l = landsat_tf[:6]
    cols_l = ((x5070 - c_l) / a_l).astype(int)
    rows_l = ((y5070 - f_l) / e_l).astype(int)
    _, lH, lW = landsat_data.shape
    valid_l = (rows_l >= 0) & (rows_l < lH) & (cols_l >= 0) & (cols_l < lW)
    landsat_feats = np.zeros((n, 7), dtype="float32")
    for b in range(7):
        landsat_feats[:, b] = np.where(
            valid_l,
            landsat_data[
                b_start + b,
                np.clip(rows_l, 0, lH - 1),
                np.clip(cols_l, 0, lW - 1),
            ],
            0.0,
        )

    # Temporal
    doy = day.dayofyear
    doy_sin = np.float32(np.sin(2 * np.pi * doy / 365.25))
    doy_cos = np.float32(np.cos(2 * np.pi * doy / 365.25))
    temporal_feats = np.full((n, 2), [doy_sin, doy_cos], dtype="float32")

    # Location
    loc_feats = np.column_stack(
        [
            sta["lats"].astype("float32"),
            sta["lons"].astype("float32"),
        ]
    )

    x = np.concatenate(
        [
            urma_feats,
            terrain_feats,
            rsun_feats,
            landsat_feats,
            temporal_feats,
            loc_feats,
        ],
        axis=1,
    )
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Tile + graph construction
# ---------------------------------------------------------------------------


def _build_tile_graph(
    pixel_features: np.ndarray,
    pixel_lats: np.ndarray,
    pixel_lons: np.ndarray,
    pixel_elevs: np.ndarray,
    pixel_ugrd: np.ndarray,
    pixel_vgrd: np.ndarray,
    station_features: np.ndarray,
    sta: dict,
    sta_indices: np.ndarray,
    edge_norm: dict,
    norm_stats: dict,
    feature_cols: list[str],
) -> dict:
    """Build a combined grid+station graph for one tile.

    Returns dict with x, edge_index, edge_attr, n_grid tensors.
    """
    n_grid = len(pixel_features)
    n_sta = len(sta_indices)
    n_total = n_grid + n_sta

    # Combined node features: [grid_pixels..., stations...]
    x = np.concatenate([pixel_features, station_features[sta_indices]], axis=0)

    # --- Edge construction ---
    src_list = []
    dst_list = []
    edge_feats = []

    sta_fids = sta["fids"][sta_indices]
    sta_fid_to_local = {fid: n_grid + j for j, fid in enumerate(sta_fids)}

    # Grid ← Station edges (stations send messages to grid pixels)
    # Each grid pixel connects to its k nearest stations
    pix_coords_rad = np.radians(np.column_stack([pixel_lats, pixel_lons]))
    sta_coords_rad = sta["coords_rad"][sta_indices]
    sta_tree = BallTree(sta_coords_rad, metric="haversine")
    k_query = min(16, n_sta)
    dists_rad, nbr_idx = sta_tree.query(pix_coords_rad, k=k_query)

    # Fully vectorized grid←station edge features
    # Flatten all k neighbors: (n_grid * k_query,)
    all_dist_km = (dists_rad * 6371.0).ravel()
    all_j_local = nbr_idx.ravel()
    all_pix_idx = np.repeat(np.arange(n_grid), k_query)

    # Filter by max radius
    keep = all_dist_km <= 150.0
    all_dist_km = all_dist_km[keep]
    all_j_local = all_j_local[keep]
    all_pix_idx = all_pix_idx[keep]

    if len(all_pix_idx) > 0:
        s_lat = sta_coords_rad[all_j_local, 0]
        s_lon = sta_coords_rad[all_j_local, 1]
        p_lat = pix_coords_rad[all_pix_idx, 0]
        p_lon = pix_coords_rad[all_pix_idx, 1]

        dlon = s_lon - p_lon
        bx = np.sin(dlon) * np.cos(s_lat)
        by = np.cos(p_lat) * np.sin(s_lat) - np.sin(p_lat) * np.cos(s_lat) * np.cos(
            dlon
        )
        brng_rad = np.arctan2(bx, by)

        dist_norm = (all_dist_km - edge_norm["dist_mean"]) / edge_norm["dist_std"]
        s_elev = sta["elevs"][sta_indices[all_j_local]]
        p_elev = pixel_elevs[all_pix_idx]
        d_elev = (s_elev - p_elev - edge_norm["delev_mean"]) / edge_norm["delev_std"]

        theta_from = np.arctan2(-pixel_ugrd[all_pix_idx], -pixel_vgrd[all_pix_idx])
        upwind_cos = np.cos(theta_from - brng_rad)
        upwind_sin = np.sin(theta_from - brng_rad)

        gs_src = (n_grid + all_j_local).tolist()
        gs_dst = all_pix_idx.tolist()
        gs_feats = np.column_stack(
            [
                dist_norm,
                np.sin(brng_rad),
                np.cos(brng_rad),
                d_elev,
                np.zeros(len(all_pix_idx), dtype="float32"),
                upwind_cos,
                upwind_sin,
            ]
        )
        src_list.extend(gs_src)
        dst_list.extend(gs_dst)
        edge_feats.extend(gs_feats.tolist())

    # Station ← Station edges (reuse training knn_map + static_edges)
    knn_map = sta["knn_map"]
    static_edges = sta["static_edges"]
    for j, fid_i in enumerate(sta_fids):
        local_i = n_grid + j
        nbrs = knn_map.get(str(fid_i), [])
        u_i = float(station_features[sta_indices[j], 4])  # ugrd_urma
        v_i = float(station_features[sta_indices[j], 5])  # vgrd_urma
        theta_from_i = np.arctan2(-u_i, -v_i)

        for fid_j in nbrs:
            if str(fid_j) not in sta_fid_to_local:
                continue
            local_j = sta_fid_to_local[str(fid_j)]

            ea = static_edges.get(str(fid_i), {}).get(str(fid_j))
            if ea is None:
                continue

            d_n = (ea["distance_km"] - edge_norm["dist_mean"]) / edge_norm["dist_std"]
            b_s = ea["bearing_sin"]
            b_c = ea["bearing_cos"]
            d_e = (ea["delta_elevation"] - edge_norm["delev_mean"]) / edge_norm[
                "delev_std"
            ]
            b_r = np.arctan2(b_s, b_c)
            uw_cos = float(np.cos(theta_from_i - b_r))
            uw_sin = float(np.sin(theta_from_i - b_r))

            src_list.append(local_j)
            dst_list.append(local_i)
            edge_feats.append([d_n, b_s, b_c, d_e, 0.0, uw_cos, uw_sin])

    # Build tensors
    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 7), dtype=torch.float32)

    # Apply z-score normalization to node features
    x_t = torch.from_numpy(x.astype("float32"))
    for i, col in enumerate(feature_cols):
        if col in norm_stats:
            x_t[:, i] = (x_t[:, i] - norm_stats[col]["mean"]) / norm_stats[col]["std"]

    return {
        "x": x_t,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "n_grid": n_grid,
        "num_nodes": n_total,
    }


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------


def predict_day(
    day: pd.Timestamp,
    model: LitScalarGNN,
    urma_dir: str,
    terrain_data: np.ndarray,
    terrain_tf: np.ndarray,
    rsun_data: np.ndarray,
    rsun_tf: np.ndarray,
    landsat_data: np.ndarray,
    landsat_tf: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    valid_mask: np.ndarray,
    sta: dict,
    meta: dict,
    norm_stats: dict,
    feature_cols: list[str],
    out_dir: str,
    device: torch.device,
    tile_size: int = 256,
) -> str | None:
    """Run grid inference for a single day."""
    H, W = PNW_1KM_SHAPE
    date_str = day.strftime("%Y%m%d")

    # Load URMA COG
    urma_path = os.path.join(urma_dir, f"URMA_1km_{date_str}.tif")
    if not os.path.exists(urma_path):
        print(f"  SKIP {date_str}: no URMA COG", flush=True)
        return None

    with rasterio.open(urma_path) as src:
        urma_data = src.read().astype("float32")
        urma_tf = np.array(src.transform, dtype="float64")

    # Extract station features from rasters (same source as grid)
    station_features = _extract_station_features(
        sta,
        urma_data,
        urma_tf,
        terrain_data,
        terrain_tf,
        rsun_data,
        rsun_tf,
        landsat_data,
        landsat_tf,
        day,
    )

    # Precompute per-pixel URMA weather (11 features) for full grid
    doy_idx = min(day.dayofyear - 1, rsun_data.shape[0] - 1)
    period = _date_to_period(day)
    b_start = period * 7
    doy = day.dayofyear
    doy_sin = np.float32(np.sin(2 * np.pi * doy / 365.25))
    doy_cos = np.float32(np.cos(2 * np.pi * doy / 365.25))

    edge_norm = meta["edge_norm"]

    # Output arrays
    delta_out = np.full((H, W), np.nan, dtype="float32")

    # Tile the grid
    n_rows = (H + tile_size - 1) // tile_size
    n_cols = (W + tile_size - 1) // tile_size
    tile_count = 0
    t_tile = time.time()

    for tr in range(n_rows):
        for tc in range(n_cols):
            r0 = tr * tile_size
            c0 = tc * tile_size
            r1 = min(r0 + tile_size, H)
            c1 = min(c0 + tile_size, W)

            tile_valid = valid_mask[r0:r1, c0:c1]
            valid_idx = np.argwhere(tile_valid)
            if len(valid_idx) == 0:
                continue

            rows = valid_idx[:, 0] + r0
            cols = valid_idx[:, 1] + c0
            n_pix = len(rows)

            # Assemble 29 features per pixel
            # URMA weather (11)
            urma_feats = np.zeros((n_pix, 11), dtype="float32")
            for col_name, band_idx in URMA_BAND_MAP.items():
                feat_idx = FEATURE_COLS.index(col_name)
                urma_feats[:, feat_idx] = urma_data[band_idx, rows, cols]
            ugrd = urma_feats[:, 4]
            vgrd = urma_feats[:, 5]
            urma_feats[:, 6] = np.sqrt(ugrd**2 + vgrd**2)  # wind
            urma_feats[:, 10] = (
                np.degrees(np.arctan2(-ugrd, -vgrd)) + 360
            ) % 360  # wdir

            # Terrain (6)
            terrain_feats = terrain_data[:, rows, cols].T  # (n_pix, 6)

            # RSUN (1)
            rsun_feats = rsun_data[doy_idx, rows, cols].reshape(-1, 1)

            # Landsat (7)
            landsat_feats = landsat_data[b_start : b_start + 7, rows, cols].T

            # Temporal (2)
            temporal_feats = np.full((n_pix, 2), [doy_sin, doy_cos], dtype="float32")

            # Location (2)
            loc_feats = np.column_stack(
                [
                    lat_grid[rows, cols],
                    lon_grid[rows, cols],
                ]
            )

            pixel_features = np.concatenate(
                [
                    urma_feats,
                    terrain_feats,
                    rsun_feats,
                    landsat_feats,
                    temporal_feats,
                    loc_feats,
                ],
                axis=1,
            )
            pixel_features = np.nan_to_num(
                pixel_features, nan=0.0, posinf=0.0, neginf=0.0
            )

            pixel_lats = lat_grid[rows, cols]
            pixel_lons = lon_grid[rows, cols]
            pixel_elevs = terrain_data[0, rows, cols]  # band 0 = elevation
            pixel_ugrd = ugrd
            pixel_vgrd = vgrd

            # Find relevant stations via tile center + generous radius
            tile_center_lat = np.mean(pixel_lats)
            tile_center_lon = np.mean(pixel_lons)
            # Tile diagonal ~= tile_size * 1km * sqrt(2) ≈ 362 km max, but
            # typically ~180 km. Add 150 km search radius.
            tile_rad = (200.0 + 150.0) / 6371.0
            center_rad = np.radians([[tile_center_lat, tile_center_lon]])
            candidate_idx = sta["tree"].query_radius(center_rad, r=tile_rad)[0]

            if len(candidate_idx) == 0:
                continue

            # Build graph
            graph = _build_tile_graph(
                pixel_features=pixel_features,
                pixel_lats=pixel_lats,
                pixel_lons=pixel_lons,
                pixel_elevs=pixel_elevs,
                pixel_ugrd=pixel_ugrd,
                pixel_vgrd=pixel_vgrd,
                station_features=station_features,
                sta=sta,
                sta_indices=candidate_idx,
                edge_norm=edge_norm,
                norm_stats=norm_stats,
                feature_cols=feature_cols,
            )

            # Forward pass
            data = Data(
                x=graph["x"].to(device),
                edge_index=graph["edge_index"].to(device),
                edge_attr=graph["edge_attr"].to(device),
                num_nodes=graph["num_nodes"],
            )

            with torch.no_grad():
                pred = model(data).squeeze(-1)  # (n_total,)
            delta = pred[: graph["n_grid"]].cpu().numpy()

            # Write into output array
            delta_out[rows, cols] = delta
            tile_count += 1

    dt_tile = time.time() - t_tile
    print(
        f"  {date_str}: {tile_count} tiles, {dt_tile:.1f}s",
        flush=True,
    )

    # Compute corrected tmax = tmax_urma + delta_tmax
    tmax_urma = urma_data[1]  # band 1 = tmax_c
    corrected = tmax_urma + delta_out

    # Write 2-band GeoTIFF
    out_path = os.path.join(out_dir, f"tmax_corrected_{date_str}.tif")
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 2,
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
        dst.write(delta_out, 1)
        dst.set_band_description(1, "delta_tmax")
        dst.write(corrected, 2)
        dst.set_band_description(2, "corrected_tmax")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid inference for tmax GNN.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", help="Single date (YYYY-MM-DD)")
    g.add_argument("--start-date", help="Start of date range (YYYY-MM-DD)")
    p.add_argument("--end-date", help="End of date range (YYYY-MM-DD)")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument(
        "--norm-stats",
        default=None,
        help="Path to norm_stats.json (default: <checkpoint_dir>/norm_stats.json)",
    )
    p.add_argument(
        "--urma-dir",
        default="/nas/dads/mvp/urma_1km_pnw_2024",
        help="URMA 1km COG directory",
    )
    p.add_argument(
        "--stations-csv",
        default="/nas/dads/met/stations/madis_02JULY2025_mgrs.csv",
    )
    p.add_argument(
        "--terrain-tif",
        default="/nas/dads/mvp/terrain_pnw_1km.tif",
    )
    p.add_argument("--rsun-tif", default="/nas/dads/mvp/rsun_pnw_1km.tif")
    p.add_argument("--landsat-tif", default="/nas/dads/mvp/landsat_pnw_1km.tif")
    p.add_argument("--out-dir", required=True, help="Output directory for GeoTIFFs")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tile-size", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve dates
    if args.date:
        dates = [pd.Timestamp(args.date)]
    else:
        start = pd.Timestamp(args.start_date)
        end = pd.Timestamp(args.end_date) if args.end_date else start
        dates = pd.date_range(start, end, freq="D")

    # Load checkpoint
    ckpt_dir = os.path.dirname(args.checkpoint)
    device = torch.device(args.device)

    print("Loading model checkpoint...", flush=True)
    lit_model = LitScalarGNN.load_from_checkpoint(args.checkpoint, map_location=device)
    lit_model.eval()
    lit_model.to(device)

    # Load norm_stats
    ns_path = args.norm_stats or os.path.join(ckpt_dir, "norm_stats.json")
    print(f"Loading norm stats from {ns_path}", flush=True)
    with open(ns_path) as f:
        ns_data = json.load(f)
    feature_cols = ns_data["feature_cols"]
    norm_stats = ns_data["norm_stats"]

    # Load meta.json from graph dir (for edge_norm)
    # Try to find meta.json in the checkpoint dir first
    meta_path = os.path.join(ckpt_dir, "meta.json")
    if not os.path.exists(meta_path):
        # Fall back to the graph dir used for training
        for candidate in [
            "/nas/dads/mvp/tmax_graphs_relaxed_pnw_2018_2024/meta.json",
            "/nas/dads/mvp/tmax_graphs_pnw_2024/meta.json",
        ]:
            if os.path.exists(candidate):
                meta_path = candidate
                break
    print(f"Loading meta from {meta_path}", flush=True)
    with open(meta_path) as f:
        meta = json.load(f)

    # Load static rasters
    print("Loading static rasters...", flush=True)
    t0 = time.time()
    terrain_data, terrain_tf = _load_raster(args.terrain_tif, "Terrain")
    rsun_data, rsun_tf = _load_raster(args.rsun_tif, "RSUN")
    landsat_data, landsat_tf = _load_raster(args.landsat_tif, "Landsat")
    print(f"  Static rasters loaded in {time.time() - t0:.1f}s", flush=True)

    # Build lat/lon grid
    print("Computing lat/lon grid...", flush=True)
    t0 = time.time()
    lat_grid, lon_grid = _build_latlon_grid(PNW_1KM_SHAPE, PNW_1KM_TRANSFORM)
    print(f"  Lat/lon grid computed in {time.time() - t0:.1f}s", flush=True)

    # Valid mask from terrain band 0 (elevation)
    valid_mask = ~np.isnan(terrain_data[0])
    n_valid = valid_mask.sum()
    n_total = valid_mask.size
    print(f"  Valid pixels: {n_valid:,} / {n_total:,} ({100 * n_valid / n_total:.1f}%)")

    # Station infrastructure
    print("Building station infrastructure...", flush=True)
    t0 = time.time()
    sta = _load_station_infrastructure(
        args.stations_csv,
        meta,
        terrain_data,
        terrain_tf,
        rsun_data,
        rsun_tf,
        landsat_data,
        landsat_tf,
    )
    print(
        f"  {len(sta['fids'])} stations, "
        f"BallTree + edges built in {time.time() - t0:.1f}s",
        flush=True,
    )

    # Process each day
    print(f"\nProcessing {len(dates)} day(s)...", flush=True)
    t_all = time.time()
    written = 0
    for i, day in enumerate(dates):
        out = predict_day(
            day=day,
            model=lit_model,
            urma_dir=args.urma_dir,
            terrain_data=terrain_data,
            terrain_tf=terrain_tf,
            rsun_data=rsun_data,
            rsun_tf=rsun_tf,
            landsat_data=landsat_data,
            landsat_tf=landsat_tf,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            valid_mask=valid_mask,
            sta=sta,
            meta=meta,
            norm_stats=norm_stats,
            feature_cols=feature_cols,
            out_dir=args.out_dir,
            device=device,
            tile_size=args.tile_size,
        )
        if out is not None:
            written += 1
        if (i + 1) % 10 == 0 or (i + 1) == len(dates):
            elapsed = time.time() - t_all
            rate = (i + 1) / elapsed * 60
            print(
                f"  [{i + 1}/{len(dates)}] {written} written, {rate:.1f} days/min",
                flush=True,
            )

    print(f"\nDone. {written} GeoTIFFs written to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
