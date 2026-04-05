"""Tests for Stage C day-tile dataset, loader settings, and config parsing."""

from __future__ import annotations


import torch

from models.hrrr_da.grid_da_dataset import (
    collate_grid_da,
    collate_grid_da_day_tile,
)
from models.hrrr_da.lit_grid_da import LitGridDA
from models.hrrr_da.train_grid_da import GridDAConfig, _resolve_loader_settings


# ---------------------------------------------------------------------------
# Loader-settings resolution
# ---------------------------------------------------------------------------


def test_benchmark_val_defaults_to_one_worker():
    """Benchmark mode forces val to 1 non-persistent worker."""
    cfg = GridDAConfig(benchmark_mode=True, num_workers=4)
    _, val_w, _, val_p = _resolve_loader_settings(cfg)
    assert val_w == 1
    assert val_p is False


def test_benchmark_val_zero_workers():
    """Benchmark with num_workers=0 keeps val at 0."""
    cfg = GridDAConfig(benchmark_mode=True, num_workers=0)
    _, val_w, _, val_p = _resolve_loader_settings(cfg)
    assert val_w == 0
    assert val_p is False


def test_explicit_val_workers_override():
    """Explicit val_num_workers overrides benchmark default."""
    cfg = GridDAConfig(benchmark_mode=True, num_workers=4, val_num_workers=2)
    _, val_w, _, val_p = _resolve_loader_settings(cfg)
    assert val_w == 2


def test_train_inherits_from_num_workers():
    """Train workers default to num_workers when not specified."""
    cfg = GridDAConfig(num_workers=8)
    train_w, _, train_p, _ = _resolve_loader_settings(cfg)
    assert train_w == 8
    assert train_p is True


def test_explicit_persistent_workers():
    """Explicit persistent worker settings are respected."""
    cfg = GridDAConfig(
        num_workers=4,
        train_persistent_workers=False,
        val_persistent_workers=False,
    )
    _, _, train_p, val_p = _resolve_loader_settings(cfg)
    assert train_p is False
    assert val_p is False


# ---------------------------------------------------------------------------
# Day-tile collation
# ---------------------------------------------------------------------------


def _make_da_sample(C=10, H=16, W=16, n_src=2, pay_dim=4, n_targets=1):
    """Create a single DA sample dict."""
    return {
        "x_patch": torch.randn(C, H, W),
        "sta_rows": torch.randint(0, H, (3,)),
        "sta_cols": torch.randint(0, W, (3,)),
        "sta_targets": torch.randn(3, n_targets),
        "sta_valid": torch.ones(3, n_targets, dtype=torch.bool),
        "sta_holdout": torch.tensor([True, False, False], dtype=torch.bool),
        "sta_is_center": torch.tensor([False, False, False], dtype=torch.bool),
        "sta_is_source": torch.tensor([False, True, False], dtype=torch.bool),
        "sta_is_query": torch.tensor([False, False, False], dtype=torch.bool),
        "src_rows": torch.randint(0, H, (n_src,)),
        "src_cols": torch.randint(0, W, (n_src,)),
        "src_ctx": torch.randn(n_src, C),
        "src_pay": torch.randn(n_src, pay_dim),
        "src_valid": torch.ones(n_src, dtype=torch.bool),
        "raw_elev_patch": torch.rand(1, H, W) * 2000,
        "src_elev": torch.rand(n_src) * 2000,
        "y_dense": None,
        "y_valid_dense": None,
    }


def test_collate_grid_da_day_tile_flattens():
    """Day-tile collate flattens K tiles from each day into one batch."""
    # Simulate 2 days, each with 3 tiles
    day1 = [_make_da_sample() for _ in range(3)]
    day2 = [_make_da_sample() for _ in range(3)]
    batch = collate_grid_da_day_tile([day1, day2])

    assert batch["x_patch"].shape[0] == 6  # 2 days * 3 tiles
    assert batch["sta_rows"].shape[0] == 6
    assert batch["src_rows"].shape[0] == 6


def test_collate_grid_da_day_tile_single_day():
    """Day-tile collate works with a single day (batch_size=1)."""
    day1 = [_make_da_sample() for _ in range(4)]
    batch = collate_grid_da_day_tile([day1])

    assert batch["x_patch"].shape[0] == 4
    assert batch["src_valid"].shape[0] == 4


def test_collate_handles_variable_source_counts():
    """Collation pads source tensors when tiles have different source counts."""
    s1 = _make_da_sample(n_src=0)
    s2 = _make_da_sample(n_src=5)
    batch = collate_grid_da([s1, s2])

    assert batch["src_rows"].shape == (2, 5)
    assert batch["src_valid"][0].sum() == 0  # first sample has no sources
    assert batch["src_valid"][1].sum() == 5


# ---------------------------------------------------------------------------
# DA-on forward on day-tile-style batch
# ---------------------------------------------------------------------------


def test_da_on_forward_day_tile_batch():
    """DA-on model runs on a batch produced by day-tile collation."""
    C, H_dim = 10, 8
    model = LitGridDA(
        in_channels=C,
        target_names=["delta_tmax"],
        source_ctx_dim=C,
        source_pay_dim=4,
        hidden_dim=H_dim,
        da_enabled=True,
        benchmark_mode=False,
    )
    model.eval()

    # Simulate day-tile batch: 1 day with 4 tiles
    tiles = [_make_da_sample(C=C) for _ in range(4)]
    batch = collate_grid_da_day_tile([tiles])

    with torch.no_grad():
        pred = model(batch)

    assert pred.shape == (4, 1, 16, 16)


def test_da_off_forward_day_tile_batch():
    """DA-off model runs on a batch produced by day-tile collation."""
    C, H_dim = 10, 8
    model = LitGridDA(
        in_channels=C,
        target_names=["delta_tmax"],
        source_ctx_dim=C,
        source_pay_dim=4,
        hidden_dim=H_dim,
        da_enabled=False,
        benchmark_mode=False,
    )
    model.eval()

    tiles = [_make_da_sample(C=C) for _ in range(4)]
    batch = collate_grid_da_day_tile([tiles])

    with torch.no_grad():
        pred = model(batch)

    assert pred.shape == (4, 1, 16, 16)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_smoke_da_on_config_parses():
    """DA-on smoke config parses and has expected geometry."""
    cfg = GridDAConfig.from_toml("models/hrrr_da/configs/grid_da_s_tmax_smoke.toml")
    assert cfg.da_enabled is True
    assert cfg.train_sample_mode == "day_tile"
    assert cfg.train_tile_stride == 64
    assert cfg.train_tiles_per_day == 8
    assert cfg.num_sanity_val_steps == 0
    assert cfg.val_num_workers == 0
    assert cfg.benchmark_mode is True


def test_smoke_da_off_config_parses():
    """DA-off smoke config parses and has expected geometry."""
    cfg = GridDAConfig.from_toml("models/hrrr_da/configs/grid_da_off_tmax_smoke.toml")
    assert cfg.da_enabled is False
    assert cfg.train_sample_mode == "day_tile"
    assert cfg.train_tile_stride == 64
    assert cfg.train_tiles_per_day == 8
    assert cfg.num_sanity_val_steps == 0
    assert cfg.val_num_workers == 0
    assert cfg.benchmark_mode is True


def test_smoke_configs_match_geometry():
    """DA-on and DA-off smoke configs differ only in da_enabled."""
    on = GridDAConfig.from_toml("models/hrrr_da/configs/grid_da_s_tmax_smoke.toml")
    off = GridDAConfig.from_toml("models/hrrr_da/configs/grid_da_off_tmax_smoke.toml")
    # Same training geometry
    assert on.train_sample_mode == off.train_sample_mode
    assert on.train_tile_stride == off.train_tile_stride
    assert on.train_tiles_per_day == off.train_tiles_per_day
    assert on.days_per_epoch == off.days_per_epoch
    assert on.val_days_per_epoch == off.val_days_per_epoch
    assert on.batch_size == off.batch_size
    assert on.num_sanity_val_steps == off.num_sanity_val_steps
    # Only da_enabled differs
    assert on.da_enabled is True
    assert off.da_enabled is False
