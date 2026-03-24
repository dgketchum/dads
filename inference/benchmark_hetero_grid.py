"""
Inference speed benchmark for the hetero HRRR DA model.

Sweeps every Kth pixel of the PNW 1 km grid, builds a local graph for each
(identical to what full inference would do), runs the model forward pass, and
reports per-pixel timing.

Outputs:
  - per-pixel breakdown: graph-build time, forward-pass time
  - extrapolated wall-clock for full PNW and CONUS at 1 km
  - JSON summary written alongside the checkpoint

Usage:
    uv run python -m inference.benchmark_hetero_grid \\
        --checkpoint /nas/dads/mvp/hrrr_hetero_smoke/ckpt.ckpt \\
        --date 2024-06-15 \\
        --background-dir /nas/dads/mvp/hrrr_1km_pnw \\
        --station-table /nas/dads/mvp/station_day_hrrr_pnw.parquet \\
        --terrain-tif /nas/dads/mvp/terrain_pnw_1km.tif \\
        --stride 40
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from pyproj import Transformer
from rasterio.transform import rowcol, xy

from models.hrrr_da.hetero_dataset import (
    DEFAULT_STATION_FEATURE_CANDIDATES,
    _bearing_sin_cos,
    _doy_features,
    _haversine_km,
    _RasterCache,
)
from models.hrrr_da.lit_hetero_gnn import LitHRRRHeteroGNN
from prep.pnw_1km_grid import PNW_1KM_SHAPE

PNW_PIXELS = PNW_1KM_SHAPE[0] * PNW_1KM_SHAPE[1]
CONUS_PIXELS = 4_500 * 3_000


def _build_station_pool(
    station_table: str,
    day: pd.Timestamp,
) -> pd.DataFrame:
    df = pd.read_parquet(station_table)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["fid"] = df["fid"].astype(str)
    df["day"] = pd.to_datetime(df["day"]).dt.normalize()
    return df[df["day"] == day].reset_index(drop=True)


def _build_graph_for_pixel(
    row_idx: int,
    col_idx: int,
    bg_data: np.ndarray,
    bg_tf,
    bg_crs,
    bg_feat_names: list[str],
    terrain_data: np.ndarray | None,
    terrain_tf,
    terrain_crs,
    terrain_feat_names: list[str],
    station_pool: pd.DataFrame,
    station_feat_cols: list[str],
    day: pd.Timestamp,
    grid_radius: int,
    station_radius_km: float,
    max_stations: int,
    edge_dim: int,
):
    """Build a HeteroData graph centered on one grid pixel."""
    from torch_geometric.data import HeteroData

    Nj, Ni = bg_data.shape[1], bg_data.shape[2]
    r0 = max(0, row_idx - grid_radius)
    r1 = min(Nj, row_idx + grid_radius + 1)
    c0 = max(0, col_idx - grid_radius)
    c1 = min(Ni, col_idx + grid_radius + 1)

    rows = np.arange(r0, r1)
    cols = np.arange(c0, c1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr_flat = rr.ravel()
    cc_flat = cc.ravel()

    bg_feats = bg_data[:, rr_flat, cc_flat].T.astype("float32")

    x_proj, y_proj = xy(bg_tf, rr_flat, cc_flat, offset="center")
    x_proj = np.asarray(x_proj, dtype="float64")
    y_proj = np.asarray(y_proj, dtype="float64")
    to_ll = Transformer.from_crs(bg_crs, "EPSG:4326", always_xy=True)
    grid_lon, grid_lat = to_ll.transform(x_proj, y_proj)
    grid_lon = np.asarray(grid_lon, dtype="float32")
    grid_lat = np.asarray(grid_lat, dtype="float32")

    feats = [bg_feats]
    if terrain_data is not None:
        if str(terrain_crs) == str(bg_crs):
            tr, tc = rowcol(terrain_tf, x_proj, y_proj)
        else:
            to_terrain = Transformer.from_crs(bg_crs, terrain_crs, always_xy=True)
            tx, ty = to_terrain.transform(x_proj, y_proj)
            tr, tc = rowcol(terrain_tf, tx, ty)
        tr = np.clip(np.asarray(tr, dtype=int), 0, terrain_data.shape[1] - 1)
        tc = np.clip(np.asarray(tc, dtype=int), 0, terrain_data.shape[2] - 1)
        feats.append(terrain_data[:, tr, tc].T.astype("float32"))

    doy_sin, doy_cos = _doy_features(day)
    feats.append(np.full((len(rr_flat), 2), [doy_sin, doy_cos], dtype="float32"))
    feats.append(np.column_stack([grid_lat, grid_lon]).astype("float32"))
    grid_x = np.concatenate(feats, axis=1)

    # Center pixel projected coords for station search
    cx, cy = xy(bg_tf, np.array([row_idx]), np.array([col_idx]), offset="center")
    clat, clon = to_ll.transform(np.asarray(cx), np.asarray(cy))
    center_lat, center_lon = float(clat[0]), float(clon[0])

    # Station nodes
    if not station_pool.empty:
        d_km = _haversine_km(
            np.full(len(station_pool), center_lat),
            np.full(len(station_pool), center_lon),
            station_pool["latitude"].to_numpy(dtype=float),
            station_pool["longitude"].to_numpy(dtype=float),
        )
        nearby = station_pool[d_km <= station_radius_km].copy()
        nearby["__d"] = d_km[d_km <= station_radius_km]
        nearby = nearby.sort_values("__d").head(max_stations)
        station_lat = nearby["latitude"].to_numpy(dtype="float32")
        station_lon = nearby["longitude"].to_numpy(dtype="float32")
        station_elev = (
            nearby["elevation"].fillna(0.0).to_numpy(dtype="float32")
            if "elevation" in nearby
            else np.zeros(len(nearby), dtype="float32")
        )
        base = (
            nearby[[c for c in station_feat_cols if c in nearby.columns]]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype="float32")
        )
        station_x = base
    else:
        station_lat = np.zeros(0, dtype="float32")
        station_lon = np.zeros(0, dtype="float32")
        station_elev = np.zeros(0, dtype="float32")
        station_x = np.zeros((0, len(station_feat_cols)), dtype="float32")

    # Grid edges (4-connected)
    patch_h = r1 - r0
    patch_w = c1 - c0
    feat_map = {
        n: i
        for i, n in enumerate(
            bg_feat_names
            + (terrain_feat_names if terrain_data is not None else [])
            + ["doy_sin", "doy_cos", "latitude", "longitude"]
        )
    }
    elev_idx = feat_map.get("elevation")
    ugrd_idx = feat_map.get("ugrd_hrrr")
    vgrd_idx = feat_map.get("vgrd_hrrr")

    src_list, dst_list, gg_feats = [], [], []
    for r in range(patch_h):
        for c in range(patch_w):
            dst = r * patch_w + c
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr2, cc2 = r + dr, c + dc
                if not (0 <= rr2 < patch_h and 0 <= cc2 < patch_w):
                    continue
                src = rr2 * patch_w + cc2
                dist_km = float(
                    _haversine_km(
                        np.array([grid_lat[dst]]),
                        np.array([grid_lon[dst]]),
                        np.array([grid_lat[src]]),
                        np.array([grid_lon[src]]),
                    )[0]
                )
                br_sin, br_cos = _bearing_sin_cos(
                    np.array([grid_lat[src]]),
                    np.array([grid_lon[src]]),
                    np.array([grid_lat[dst]]),
                    np.array([grid_lon[dst]]),
                )
                delev = (
                    float(grid_x[src, elev_idx] - grid_x[dst, elev_idx])
                    if elev_idx is not None
                    else 0.0
                )
                if ugrd_idx is not None and vgrd_idx is not None:
                    u, v = float(grid_x[dst, ugrd_idx]), float(grid_x[dst, vgrd_idx])
                    theta = np.arctan2(-u, -v)
                    brng = np.arctan2(br_sin[0], br_cos[0])
                    uw_cos, uw_sin = (
                        float(np.cos(theta - brng)),
                        float(np.sin(theta - brng)),
                    )
                else:
                    uw_cos = uw_sin = 0.0
                src_list.append(src)
                dst_list.append(dst)
                gg_feats.append(
                    [
                        dist_km / 5.0,
                        float(br_sin[0]),
                        float(br_cos[0]),
                        delev,
                        0.0,
                        uw_cos,
                        uw_sin,
                    ]
                )

    # Station→grid edges
    sg_src, sg_dst, sg_feats = [], [], []
    for s in range(len(station_lat)):
        d_km = _haversine_km(
            np.full(len(grid_lat), station_lat[s]),
            np.full(len(grid_lon), station_lon[s]),
            grid_lat.astype(float),
            grid_lon.astype(float),
        )
        br_sin_a, br_cos_a = _bearing_sin_cos(
            np.full(len(grid_lat), station_lat[s]),
            np.full(len(grid_lon), station_lon[s]),
            grid_lat.astype(float),
            grid_lon.astype(float),
        )
        for g in range(len(grid_lat)):
            delev = (
                float(station_elev[s] - grid_x[g, elev_idx])
                if elev_idx is not None
                else 0.0
            )
            if ugrd_idx is not None and vgrd_idx is not None:
                u, v = float(grid_x[g, ugrd_idx]), float(grid_x[g, vgrd_idx])
                theta = np.arctan2(-u, -v)
                brng = np.arctan2(br_sin_a[g], br_cos_a[g])
                uw_cos, uw_sin = (
                    float(np.cos(theta - brng)),
                    float(np.sin(theta - brng)),
                )
            else:
                uw_cos = uw_sin = 0.0
            sg_src.append(s)
            sg_dst.append(g)
            sg_feats.append(
                [
                    float(d_km[g] / max(station_radius_km, 1.0)),
                    float(br_sin_a[g]),
                    float(br_cos_a[g]),
                    delev,
                    0.0,
                    uw_cos,
                    uw_sin,
                ]
            )

    data = HeteroData()
    data["grid"].x = torch.from_numpy(grid_x.astype("float32"))
    data["station"].x = torch.from_numpy(station_x)
    if src_list:
        data["grid", "neighbors", "grid"].edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long
        )
        data["grid", "neighbors", "grid"].edge_attr = torch.tensor(
            gg_feats, dtype=torch.float32
        )
    else:
        data["grid", "neighbors", "grid"].edge_index = torch.zeros(
            (2, 0), dtype=torch.long
        )
        data["grid", "neighbors", "grid"].edge_attr = torch.zeros(
            (0, edge_dim), dtype=torch.float32
        )
    if sg_src:
        data["station", "influences", "grid"].edge_index = torch.tensor(
            [sg_src, sg_dst], dtype=torch.long
        )
        data["station", "influences", "grid"].edge_attr = torch.tensor(
            sg_feats, dtype=torch.float32
        )
    else:
        data["station", "influences", "grid"].edge_index = torch.zeros(
            (2, 0), dtype=torch.long
        )
        data["station", "influences", "grid"].edge_attr = torch.zeros(
            (0, edge_dim), dtype=torch.float32
        )
    return data


def run_benchmark(
    checkpoint: str,
    date_str: str,
    background_dir: str,
    station_table: str,
    terrain_tif: str | None,
    stride: int,
    device: str,
    grid_radius: int = 2,
    station_radius_km: float = 150.0,
    max_stations: int = 16,
) -> dict:
    day = pd.Timestamp(date_str)
    background_pattern = "HRRR_1km_{date}.tif"
    bg_path = os.path.join(
        background_dir, background_pattern.format(date=day.strftime("%Y%m%d"))
    )
    if not os.path.exists(bg_path):
        raise FileNotFoundError(f"Background raster not found: {bg_path}")

    dev = torch.device(device)
    print(f"Loading checkpoint: {checkpoint}", flush=True)
    model = LitHRRRHeteroGNN.load_from_checkpoint(checkpoint, map_location=dev)
    model.eval()

    print("Loading rasters ...", flush=True)
    cache = _RasterCache(max_items=2)
    bg = cache.get(bg_path)
    bg_data = bg["data"]
    bg_tf = bg["transform"]
    bg_crs = bg["crs"]
    bg_feat_names = [d if d else f"bg_{i}" for i, d in enumerate(bg["descriptions"])]

    terrain_data = terrain_tf = terrain_crs = None
    terrain_feat_names: list[str] = []
    if terrain_tif and os.path.exists(terrain_tif):
        t = cache.get(terrain_tif)
        terrain_data = t["data"]
        terrain_tf = t["transform"]
        terrain_crs = t["crs"]
        terrain_feat_names = [
            d if d else f"terrain_{i}" for i, d in enumerate(t["descriptions"])
        ]

    print(f"Loading station pool for {date_str} ...", flush=True)
    station_pool = _build_station_pool(station_table, day)
    print(f"  {len(station_pool)} station rows for this day", flush=True)

    # Infer station feature cols from the pool
    station_feat_cols = [
        c for c in DEFAULT_STATION_FEATURE_CANDIDATES if c in station_pool.columns
    ]

    H, W = bg_data.shape[1], bg_data.shape[2]
    row_coords = np.arange(grid_radius, H - grid_radius, stride)
    col_coords = np.arange(grid_radius, W - grid_radius, stride)
    pixel_pairs = [(r, c) for r in row_coords for c in col_coords]
    n_samples = len(pixel_pairs)
    print(
        f"Benchmarking {n_samples} pixels (stride={stride}, grid={H}×{W}={H * W:,} total) ...",
        flush=True,
    )

    build_times: list[float] = []
    fwd_times: list[float] = []

    edge_dim = 7
    with torch.no_grad():
        for i, (r, c) in enumerate(pixel_pairs):
            t0 = time.perf_counter()
            data = _build_graph_for_pixel(
                r,
                c,
                bg_data,
                bg_tf,
                bg_crs,
                bg_feat_names,
                terrain_data,
                terrain_tf,
                terrain_crs,
                terrain_feat_names,
                station_pool,
                station_feat_cols,
                day,
                grid_radius,
                station_radius_km,
                max_stations,
                edge_dim,
            )
            t1 = time.perf_counter()
            data = data.to(dev)
            _ = model(data)
            t2 = time.perf_counter()

            build_times.append(t1 - t0)
            fwd_times.append(t2 - t1)

            if (i + 1) % 100 == 0 or i == n_samples - 1:
                mean_build = np.mean(build_times) * 1000
                mean_fwd = np.mean(fwd_times) * 1000
                mean_total = mean_build + mean_fwd
                print(
                    f"  {i + 1}/{n_samples}  build={mean_build:.2f}ms  fwd={mean_fwd:.2f}ms  total={mean_total:.2f}ms/px",
                    flush=True,
                )

    mean_build_ms = float(np.mean(build_times)) * 1000
    mean_fwd_ms = float(np.mean(fwd_times)) * 1000
    mean_total_ms = mean_build_ms + mean_fwd_ms

    pnw_total_px = PNW_PIXELS
    results = {
        "date": date_str,
        "stride": stride,
        "n_samples": n_samples,
        "mean_build_ms": round(mean_build_ms, 3),
        "mean_fwd_ms": round(mean_fwd_ms, 3),
        "mean_total_ms": round(mean_total_ms, 3),
        "device": device,
    }

    print("\n=== Extrapolation ===")
    for label, n_px in [("PNW 1km", pnw_total_px), ("CONUS 1km", CONUS_PIXELS)]:
        t_s = n_px * mean_total_ms / 1000.0
        results[f"{label.replace(' ', '_')}_est_min"] = round(t_s / 60, 1)
        print(
            f"  {label:12s}: {n_px:>12,} px × {mean_total_ms:.2f}ms = {t_s / 60:.1f} min"
        )

    print("\n=== Breakdown ===")
    print(
        f"  Graph build: {mean_build_ms:.2f}ms/px  ({mean_build_ms / mean_total_ms * 100:.0f}%)"
    )
    print(
        f"  Forward pass: {mean_fwd_ms:.2f}ms/px  ({mean_fwd_ms / mean_total_ms * 100:.0f}%)"
    )

    out_path = os.path.join(os.path.dirname(checkpoint), "inference_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark hetero HRRR DA grid inference speed."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--date", required=True, help="Date to benchmark (YYYY-MM-DD).")
    p.add_argument("--background-dir", default="/nas/dads/mvp/hrrr_1km_pnw")
    p.add_argument(
        "--station-table", default="/nas/dads/mvp/station_day_hrrr_pnw.parquet"
    )
    p.add_argument("--terrain-tif", default="/nas/dads/mvp/terrain_pnw_1km.tif")
    p.add_argument(
        "--stride", type=int, default=40, help="Sample every Nth pixel (default 40)."
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--grid-radius", type=int, default=2)
    p.add_argument("--station-radius-km", type=float, default=150.0)
    p.add_argument("--max-stations", type=int, default=16)
    a = p.parse_args()

    run_benchmark(
        checkpoint=a.checkpoint,
        date_str=a.date,
        background_dir=a.background_dir,
        station_table=a.station_table,
        terrain_tif=a.terrain_tif,
        stride=a.stride,
        device=a.device,
        grid_radius=a.grid_radius,
        station_radius_km=a.station_radius_km,
        max_stations=a.max_stations,
    )


if __name__ == "__main__":
    main()
