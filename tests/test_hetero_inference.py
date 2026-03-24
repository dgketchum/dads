"""End-to-end smoke tests for hetero HRRR DA inference pipeline.

Uses a synthetic 20×15 grid (EPSG:4326) with 3 stations and 1 day.
No NAS data required — everything is built from scratch in tmp_path.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from models.hrrr_da.hetero_dataset import HRRRHeteroTileDataset
from models.hrrr_da.lit_hetero_gnn import LitHRRRHeteroGNN
from prep.build_hetero_inference_index import build_hetero_inference_index
from inference.predict_hrrr_hetero_grid import (
    load_inference_assets,
    predict_day_hetero,
)


# ---------------------------------------------------------------------------
# Synthetic raster/table helpers
# ---------------------------------------------------------------------------

_H, _W = 20, 15
_LAT_S, _LAT_N = 45.0, 45.2
_LON_W, _LON_E = -120.2, -120.0

_TRANSFORM = from_bounds(_LON_W, _LAT_S, _LON_E, _LAT_N, _W, _H)
_CRS = CRS.from_epsg(4326)

_BG_BANDS = [
    "ugrd_hrrr",
    "vgrd_hrrr",
    "wind_hrrr",
    "tmp_hrrr",
    "dpt_hrrr",
    "pres_hrrr",
    "tcdc_hrrr",
    "ea_hrrr",
    "dswrf_hrrr",
    "hpbl_hrrr",
    "spfh_hrrr",
    "tmax_hrrr",
    "tmin_hrrr",
    "wdir_hrrr",
    "n_hours",
]
_TER_BANDS = ["elevation", "slope", "aspect_sin", "aspect_cos", "tpi_4", "tpi_10"]
_TARGET_NAMES = ["delta_tmax", "delta_tmin"]


def _write_tif(
    path, data: np.ndarray, descriptions: list[str], crs=_CRS, transform=_TRANSFORM
):
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("float32"))
        dst.descriptions = tuple(descriptions)


def _make_station_table(day: pd.Timestamp) -> pd.DataFrame:
    """3 stations within the synthetic domain."""
    rows = []
    lats = [45.05, 45.10, 45.15]
    lons = [-120.15, -120.10, -120.05]
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        base = {
            "fid": f"S{i}",
            "day": day,
            "latitude": lat,
            "longitude": lon,
            "elevation": 500.0 + i * 50,
            "ugrd_hrrr": 1.0,
            "vgrd_hrrr": -0.5,
            "wind_hrrr": 1.1,
            "tmp_hrrr": 15.0 + i,
            "dpt_hrrr": 8.0,
            "pres_hrrr": 90.0,
            "tcdc_hrrr": 20.0,
            "ea_hrrr": 1.0,
            "dswrf_hrrr": 200.0,
            "hpbl_hrrr": 1000.0,
            "spfh_hrrr": 0.005,
            "delta_tmax": float(i) * 0.5,
            "delta_tmin": float(-i) * 0.3,
            "slope": 0.1,
            "aspect_sin": 0.0,
            "aspect_cos": 1.0,
            "tpi_4": 0.0,
            "tpi_10": 0.0,
            "doy_sin": 0.1,
            "doy_cos": 0.9,
        }
        rows.append(base)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_env(tmp_path_factory):
    """Build all synthetic artefacts once for the module."""
    tmp = tmp_path_factory.mktemp("hetero_infer")

    day = pd.Timestamp("2024-06-15")

    # Background raster (15 bands)
    bg_data = np.random.rand(len(_BG_BANDS), _H, _W).astype("float32") * 2
    bg_path = tmp / f"HRRR_1km_{day.strftime('%Y%m%d')}.tif"
    _write_tif(bg_path, bg_data, _BG_BANDS)

    # Terrain raster (6 bands)
    ter_data = np.random.rand(len(_TER_BANDS), _H, _W).astype("float32") * 100
    ter_data[0] = 500.0  # elevation band
    ter_path = tmp / "terrain.tif"
    _write_tif(ter_path, ter_data, _TER_BANDS)

    # Station-day parquet
    df = _make_station_table(day)
    table_path = tmp / "station_day.parquet"
    df.to_parquet(table_path, index=False)

    return {
        "tmp": tmp,
        "day": day,
        "bg_path": bg_path,
        "ter_path": ter_path,
        "table_path": table_path,
        "bg_data": bg_data,
    }


# ---------------------------------------------------------------------------
# Test 1: index build
# ---------------------------------------------------------------------------


def test_build_inference_index(synthetic_env, monkeypatch):
    """build_hetero_inference_index should produce a valid .npz for a tiny grid."""
    tmp = synthetic_env["tmp"]
    table_path = synthetic_env["table_path"]
    ter_path = synthetic_env["ter_path"]
    out_path = str(tmp / "test_index.npz")

    # Monkeypatch grid constants to use our tiny 20×15 synthetic grid
    import prep.build_hetero_inference_index as bhi_mod
    import prep.pnw_1km_grid as grid_mod

    monkeypatch.setattr(grid_mod, "PNW_1KM_SHAPE", (_H, _W))
    monkeypatch.setattr(grid_mod, "PNW_1KM_TRANSFORM", _TRANSFORM)
    monkeypatch.setattr(grid_mod, "PNW_1KM_CRS", _CRS)
    monkeypatch.setattr(bhi_mod, "PNW_1KM_SHAPE", (_H, _W))
    monkeypatch.setattr(bhi_mod, "PNW_1KM_TRANSFORM", _TRANSFORM)
    monkeypatch.setattr(bhi_mod, "PNW_1KM_CRS", _CRS)

    build_hetero_inference_index(
        table_path=str(table_path),
        terrain_tif=str(ter_path),
        radius_km=50.0,
        max_k=3,
        out_path=out_path,
        chunk_size=100,
    )

    npz = np.load(out_path, allow_pickle=True)
    n_pixels = _H * _W
    assert npz["nbr_idx"].shape == (n_pixels, 3)
    assert npz["nbr_dist_km"].shape == (n_pixels, 3)
    assert npz["fids"].shape == (3,)  # 3 unique stations


# ---------------------------------------------------------------------------
# Test 2: full pipeline round-trip
# ---------------------------------------------------------------------------


def test_predict_day_hetero_roundtrip(synthetic_env, monkeypatch, tmp_path):
    """Full pipeline: build index → save norm_stats → run inference → validate output."""
    tmp = synthetic_env["tmp"]
    table_path = synthetic_env["table_path"]
    ter_path = synthetic_env["ter_path"]
    day = synthetic_env["day"]

    # Monkeypatch grid constants
    import prep.build_hetero_inference_index as bhi_mod
    import prep.pnw_1km_grid as grid_mod
    import inference.predict_hrrr_hetero_grid as infer_mod

    monkeypatch.setattr(grid_mod, "PNW_1KM_SHAPE", (_H, _W))
    monkeypatch.setattr(grid_mod, "PNW_1KM_TRANSFORM", _TRANSFORM)
    monkeypatch.setattr(grid_mod, "PNW_1KM_CRS", _CRS)
    monkeypatch.setattr(bhi_mod, "PNW_1KM_SHAPE", (_H, _W))
    monkeypatch.setattr(bhi_mod, "PNW_1KM_TRANSFORM", _TRANSFORM)
    monkeypatch.setattr(bhi_mod, "PNW_1KM_CRS", _CRS)
    monkeypatch.setattr(infer_mod, "PNW_1KM_SHAPE", (_H, _W))
    monkeypatch.setattr(infer_mod, "PNW_1KM_TRANSFORM", _TRANSFORM)
    monkeypatch.setattr(infer_mod, "PNW_1KM_CRS", _CRS)

    # Build inference index
    index_path = str(tmp / "index.npz")
    build_hetero_inference_index(
        table_path=str(table_path),
        terrain_tif=str(ter_path),
        radius_km=50.0,
        max_k=3,
        out_path=index_path,
        chunk_size=100,
    )

    # Build HRRRHeteroTileDataset to get norm stats + dims
    ds = HRRRHeteroTileDataset(
        table_path=str(table_path),
        background_dir=str(tmp),
        background_pattern="HRRR_1km_{date}.tif",
        terrain_tif=str(ter_path),
        target_names=_TARGET_NAMES,
        grid_radius_cells=1,
        station_radius_km=50.0,
        max_neighbor_stations=3,
    )

    # Save norm_stats.json
    norm_stats_out = {
        "grid_norm_stats": ds.grid_norm_stats,
        "station_norm_stats": ds.station_norm_stats,
        "background_feature_names": ds.background_feature_names,
        "terrain_feature_names": ds.terrain_feature_names,
        "grid_feature_names": ds.grid_feature_names,
        "station_feature_cols": ds.station_feature_cols,
        "station_mask_cols": ds.station_mask_cols,
        "target_names": ds.target_names,
        "station_radius_km": ds.station_radius_km,
        "grid_node_dim": ds.grid_node_dim,
        "station_node_dim": ds.station_node_dim,
        "edge_dim": ds.edge_dim,
    }
    norm_path = str(tmp / "norm_stats.json")
    with open(norm_path, "w") as f:
        json.dump(norm_stats_out, f)

    # Build a LitHRRRHeteroGNN with random weights and save checkpoint
    model = LitHRRRHeteroGNN(
        grid_node_dim=ds.grid_node_dim,
        station_node_dim=ds.station_node_dim,
        edge_dim=ds.edge_dim,
        hidden_dim=8,
        n_hops=1,
        target_names=_TARGET_NAMES,
    )
    import lightning

    ckpt_path = str(tmp / "test.ckpt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": lightning.__version__,
            "epoch": 0,
            "global_step": 0,
        },
        ckpt_path,
    )

    # load_inference_assets must work
    assets = load_inference_assets(
        checkpoint=ckpt_path,
        norm_stats_path=norm_path,
        index_path=index_path,
        terrain_tif=str(ter_path),
        device=torch.device("cpu"),
    )

    station_table = pd.read_parquet(str(table_path))
    station_table["fid"] = station_table["fid"].astype(str)
    station_table["day"] = pd.to_datetime(station_table["day"]).dt.normalize()

    out_dir = str(tmp_path / "out")
    out_path = predict_day_hetero(
        day=day,
        model=assets["model"],
        assets=assets,
        station_table=station_table,
        background_dir=str(tmp),
        background_pattern="HRRR_1km_{date}.tif",
        out_dir=out_dir,
        device=torch.device("cpu"),
        tile_size=8,
    )

    assert out_path is not None
    assert out_path.endswith(".tif")

    with rasterio.open(out_path) as src:
        assert src.count == len(_TARGET_NAMES)
        assert src.height == _H
        assert src.width == _W
        descs = list(src.descriptions)
        assert descs == _TARGET_NAMES
        data = src.read()

    assert data.shape == (len(_TARGET_NAMES), _H, _W)
    # No NaN expected — all pixels have background raster values
    assert not np.isnan(data).any(), "Output contains unexpected NaN values"
