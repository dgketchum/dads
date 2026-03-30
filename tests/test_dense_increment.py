"""Tests for the dense-increment dataset (Stage A pretraining)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.transform import from_bounds

from models.hrrr_da.dense_increment_dataset import (
    DenseIncrementDataset,
    collate_dense_increment,
)


def _write_tif(path, data, descriptions, crs="EPSG:5070"):
    """Write a synthetic GeoTIFF with the given bands and descriptions."""
    h, w = data.shape[1], data.shape[2]
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": data.shape[0],
        "dtype": "float32",
        "crs": crs,
        "transform": from_bounds(0, 0, w * 1000, h * 1000, w, h),
    }
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(data.astype("float32"))
        dst.descriptions = tuple(descriptions)


def _make_test_rasters(tmp_path, h=80, w=80):
    """Create minimal HRRR and URMA rasters plus a terrain static TIF."""
    bg_dir = tmp_path / "hrrr"
    bg_dir.mkdir()
    urma_dir = tmp_path / "urma"
    urma_dir.mkdir()

    hrrr_bands = ["tmax_hrrr", "tmin_hrrr"]
    urma_bands = ["tmax_c", "tmin_c"]

    rng = np.random.default_rng(42)
    for day_str in ["20200101", "20200102", "20200103"]:
        bg_data = rng.standard_normal((2, h, w)).astype("float32") + 280.0
        _write_tif(bg_dir / f"HRRR_1km_{day_str}.tif", bg_data, hrrr_bands)

        urma_data = bg_data + rng.standard_normal((2, h, w)).astype("float32") * 2.0
        _write_tif(urma_dir / f"URMA_1km_{day_str}.tif", urma_data, urma_bands)

    terrain_data = np.stack(
        [
            rng.uniform(100, 2000, (h, w)).astype("float32"),
            rng.uniform(0, 0.5, (h, w)).astype("float32"),
        ]
    )
    terrain_path = tmp_path / "terrain.tif"
    _write_tif(terrain_path, terrain_data, ["elevation", "slope"])

    return bg_dir, urma_dir, terrain_path


def test_dataset_basic(tmp_path):
    """Verify shapes and delta computation."""
    bg_dir, urma_dir, terrain_path = _make_test_rasters(tmp_path)
    train_days = {pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")}

    ds = DenseIncrementDataset(
        background_dir=str(bg_dir),
        background_pattern="HRRR_1km_{date}.tif",
        teacher_dir=str(urma_dir),
        teacher_pattern="URMA_1km_{date}.tif",
        static_tifs=[str(terrain_path)],
        train_days=train_days,
        increment_map={"tmax_c": "tmax_hrrr"},
        patch_size=64,
        tiles_per_day=2,
        base_seed=42,
    )

    assert len(ds) == 4  # 2 days × 2 tiles
    x, y, v = ds[0]
    assert x.shape == (ds.in_channels, 64, 64)
    assert y.shape == (1, 64, 64)
    assert v.shape == (1, 64, 64)
    assert v.dtype == torch.bool
    assert v.all()  # synthetic data has no NaN


def test_delta_correctness(tmp_path):
    """Verify y_dense == URMA − HRRR at the same tile location."""
    h, w = 80, 80
    bg_dir = tmp_path / "hrrr"
    bg_dir.mkdir()
    urma_dir = tmp_path / "urma"
    urma_dir.mkdir()

    hrrr_val = np.full((1, h, w), 10.0, dtype="float32")
    urma_val = np.full((1, h, w), 13.5, dtype="float32")
    _write_tif(bg_dir / "HRRR_1km_20200101.tif", hrrr_val, ["tmax_hrrr"])
    _write_tif(urma_dir / "URMA_1km_20200101.tif", urma_val, ["tmax_c"])

    ds = DenseIncrementDataset(
        background_dir=str(bg_dir),
        background_pattern="HRRR_1km_{date}.tif",
        teacher_dir=str(urma_dir),
        teacher_pattern="URMA_1km_{date}.tif",
        train_days={pd.Timestamp("2020-01-01")},
        increment_map={"tmax_c": "tmax_hrrr"},
        patch_size=64,
        tiles_per_day=1,
    )
    _, y, v = ds[0]
    assert v.all()
    assert torch.allclose(y, torch.full_like(y, 3.5), atol=1e-5)


def test_nan_handling(tmp_path):
    """Verify NaN in URMA produces y_valid=False."""
    h, w = 80, 80
    bg_dir = tmp_path / "hrrr"
    bg_dir.mkdir()
    urma_dir = tmp_path / "urma"
    urma_dir.mkdir()

    hrrr_val = np.full((1, h, w), 10.0, dtype="float32")
    urma_val = np.full((1, h, w), 13.5, dtype="float32")
    urma_val[0, 30:40, 30:40] = np.nan
    _write_tif(bg_dir / "HRRR_1km_20200101.tif", hrrr_val, ["tmax_hrrr"])
    _write_tif(urma_dir / "URMA_1km_20200101.tif", urma_val, ["tmax_c"])

    ds = DenseIncrementDataset(
        background_dir=str(bg_dir),
        background_pattern="HRRR_1km_{date}.tif",
        teacher_dir=str(urma_dir),
        teacher_pattern="URMA_1km_{date}.tif",
        train_days={pd.Timestamp("2020-01-01")},
        increment_map={"tmax_c": "tmax_hrrr"},
        patch_size=64,
        tiles_per_day=1,
    )
    _, y, v = ds[0]
    # Some pixels should be invalid
    assert not v.all()
    # Valid pixels should have the correct delta
    assert torch.allclose(y[v], torch.full((v.sum(),), 3.5), atol=1e-5)
    # Invalid pixels should be zero (NaN replaced)
    assert (y[~v] == 0.0).all()


def test_epoch_diversity(tmp_path):
    """Verify set_epoch changes tile origins for the same day."""
    bg_dir, urma_dir, terrain_path = _make_test_rasters(tmp_path)
    train_days = {pd.Timestamp("2020-01-01")}

    ds = DenseIncrementDataset(
        background_dir=str(bg_dir),
        background_pattern="HRRR_1km_{date}.tif",
        teacher_dir=str(urma_dir),
        teacher_pattern="URMA_1km_{date}.tif",
        static_tifs=[str(terrain_path)],
        train_days=train_days,
        increment_map={"tmax_c": "tmax_hrrr"},
        patch_size=64,
        tiles_per_day=1,
        base_seed=42,
    )

    ds.set_epoch(0)
    x0, _, _ = ds[0]
    ds.set_epoch(1)
    x1, _, _ = ds[0]
    # Different epochs should produce different tiles (different pixel content)
    assert not torch.allclose(x0, x1)


def test_collate(tmp_path):
    """Verify collate_dense_increment stacks correctly."""
    bg_dir, urma_dir, terrain_path = _make_test_rasters(tmp_path)
    train_days = {pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")}

    ds = DenseIncrementDataset(
        background_dir=str(bg_dir),
        background_pattern="HRRR_1km_{date}.tif",
        teacher_dir=str(urma_dir),
        teacher_pattern="URMA_1km_{date}.tif",
        static_tifs=[str(terrain_path)],
        train_days=train_days,
        increment_map={"tmax_c": "tmax_hrrr"},
        patch_size=64,
        tiles_per_day=1,
    )

    batch = [ds[0], ds[1]]
    x, y, v = collate_dense_increment(batch)
    assert x.shape == (2, ds.in_channels, 64, 64)
    assert y.shape == (2, 1, 64, 64)
    assert v.shape == (2, 1, 64, 64)


def test_multitarget(tmp_path):
    """Verify 2-target increment_map produces correct shapes."""
    bg_dir, urma_dir, terrain_path = _make_test_rasters(tmp_path)
    train_days = {pd.Timestamp("2020-01-01")}

    ds = DenseIncrementDataset(
        background_dir=str(bg_dir),
        background_pattern="HRRR_1km_{date}.tif",
        teacher_dir=str(urma_dir),
        teacher_pattern="URMA_1km_{date}.tif",
        static_tifs=[str(terrain_path)],
        train_days=train_days,
        increment_map={"tmax_c": "tmax_hrrr", "tmin_c": "tmin_hrrr"},
        patch_size=64,
        tiles_per_day=1,
    )

    x, y, v = ds[0]
    assert y.shape == (2, 64, 64)
    assert v.shape == (2, 64, 64)
    assert ds.target_names == ["delta_tmax_c", "delta_tmin_c"]
