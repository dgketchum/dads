"""Tests for Stage C source/query partition, split losses, and min-distance."""

from __future__ import annotations

import torch

from models.hrrr_da.grid_da_dataset import collate_grid_da
from models.hrrr_da.lit_grid_da import LitGridDA


def _make_da_sample(
    C=10,
    H=16,
    W=16,
    n_src=2,
    n_sta=5,
    pay_dim=4,
    n_targets=1,
    n_holdout=1,
    n_source=2,
    n_query=2,
):
    """Create a single DA sample dict with source/query masks."""
    holdout = torch.zeros(n_sta, dtype=torch.bool)
    holdout[:n_holdout] = True
    is_source = torch.zeros(n_sta, dtype=torch.bool)
    is_query = torch.zeros(n_sta, dtype=torch.bool)
    # Assign roles to non-holdout stations
    idx = n_holdout
    for _ in range(n_source):
        if idx < n_sta:
            is_source[idx] = True
            idx += 1
    for _ in range(n_query):
        if idx < n_sta:
            is_query[idx] = True
            idx += 1

    return {
        "x_patch": torch.randn(C, H, W),
        "sta_rows": torch.randint(0, H, (n_sta,)),
        "sta_cols": torch.randint(0, W, (n_sta,)),
        "sta_targets": torch.randn(n_sta, n_targets),
        "sta_valid": torch.ones(n_sta, n_targets, dtype=torch.bool),
        "sta_holdout": holdout,
        "sta_is_center": torch.zeros(n_sta, dtype=torch.bool),
        "sta_is_source": is_source,
        "sta_is_query": is_query,
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


# ---------------------------------------------------------------------------
# Source/query disjointness
# ---------------------------------------------------------------------------


def test_source_query_masks_are_disjoint():
    """sta_is_source and sta_is_query never overlap."""
    s = _make_da_sample(n_sta=8, n_holdout=1, n_source=3, n_query=3)
    assert not (s["sta_is_source"] & s["sta_is_query"]).any()


def test_holdout_is_neither_source_nor_query():
    """Holdout stations are not in source or query sets."""
    s = _make_da_sample(n_sta=8, n_holdout=2, n_source=3, n_query=3)
    holdout = s["sta_holdout"]
    assert not (holdout & s["sta_is_source"]).any()
    assert not (holdout & s["sta_is_query"]).any()


# ---------------------------------------------------------------------------
# Collation preserves masks
# ---------------------------------------------------------------------------


def test_collate_preserves_source_query_masks():
    """Collation pads and preserves sta_is_source and sta_is_query."""
    s1 = _make_da_sample(n_sta=4, n_holdout=0, n_source=2, n_query=2)
    s2 = _make_da_sample(n_sta=6, n_holdout=1, n_source=2, n_query=2)
    batch = collate_grid_da([s1, s2])

    assert "sta_is_source" in batch
    assert "sta_is_query" in batch
    assert batch["sta_is_source"].shape == (2, 6)  # padded to max
    assert batch["sta_is_query"].shape == (2, 6)
    # First sample's masks preserved in padded positions
    assert batch["sta_is_source"][0, :4].sum() == s1["sta_is_source"].sum()
    assert batch["sta_is_query"][0, :4].sum() == s1["sta_is_query"].sum()
    # Padded positions are False
    assert not batch["sta_is_source"][0, 4:].any()
    assert not batch["sta_is_query"][0, 4:].any()


# ---------------------------------------------------------------------------
# Split loss computation
# ---------------------------------------------------------------------------


def _make_batch_with_split(B=2, C=10, H=16, W=16):
    """Create a collated batch with source/query partition."""
    samples = [
        _make_da_sample(
            C=C, H=H, W=W, n_sta=6, n_holdout=1, n_source=2, n_query=2, n_src=3
        )
        for _ in range(B)
    ]
    return collate_grid_da(samples)


def test_split_train_loss_runs():
    """_compute_split_train_loss runs without shape errors."""
    C, H_dim = 10, 8
    model = LitGridDA(
        in_channels=C,
        target_names=["delta_tmax"],
        source_ctx_dim=C,
        source_pay_dim=4,
        hidden_dim=H_dim,
        da_enabled=True,
        benchmark_mode=True,
        bg_loss_weight=1.0,
        da_query_loss_weight=1.0,
        gate_source_penalty_weight=0.01,
    )
    model.eval()

    batch = _make_batch_with_split(B=2, C=C)
    with torch.no_grad():
        pred = model(batch)
        loss, _, _, bg_loss, da_qry_loss = model._compute_split_train_loss(batch, pred)

    assert loss.ndim == 0
    assert bg_loss.ndim == 0
    assert da_qry_loss.ndim == 0
    assert loss.isfinite()


def test_split_loss_bg_only_uses_source_stations():
    """Background loss should only depend on source station predictions."""
    C, H_dim = 10, 8
    model = LitGridDA(
        in_channels=C,
        target_names=["delta_tmax"],
        source_ctx_dim=C,
        source_pay_dim=4,
        hidden_dim=H_dim,
        da_enabled=True,
        benchmark_mode=True,
    )
    model.eval()

    batch = _make_batch_with_split(B=1, C=C)

    # Zero out query targets — bg_loss should not change
    with torch.no_grad():
        pred = model(batch)
        _, _, _, bg_loss_1, _ = model._compute_split_train_loss(batch, pred)

    batch2 = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    # Zero query station targets
    qry_mask = batch2["sta_is_query"]
    batch2["sta_targets"][qry_mask.unsqueeze(-1).expand_as(batch2["sta_targets"])] = (
        999.0
    )

    with torch.no_grad():
        pred2 = model(batch2)
        _, _, _, bg_loss_2, _ = model._compute_split_train_loss(batch2, pred2)

    assert torch.allclose(bg_loss_1, bg_loss_2, atol=1e-6), (
        "bg_loss should not depend on query station targets"
    )


def test_da_off_uses_legacy_loss_path():
    """DA-off model uses legacy _compute_loss, not split loss."""
    C, H_dim = 10, 8
    model = LitGridDA(
        in_channels=C,
        target_names=["delta_tmax"],
        source_ctx_dim=C,
        source_pay_dim=4,
        hidden_dim=H_dim,
        da_enabled=False,
        benchmark_mode=True,
    )
    model.eval()

    batch = _make_batch_with_split(B=2, C=C)
    with torch.no_grad():
        pred = model(batch)
        # Legacy path should work fine
        loss, _, _ = model._compute_loss(batch, pred, is_train=True)
    assert loss.isfinite()


def test_training_step_with_split():
    """Full training_step uses split losses when partition is active."""
    C, H_dim = 10, 8
    model = LitGridDA(
        in_channels=C,
        target_names=["delta_tmax"],
        source_ctx_dim=C,
        source_pay_dim=4,
        hidden_dim=H_dim,
        da_enabled=True,
        benchmark_mode=True,
        bg_loss_weight=1.0,
        da_query_loss_weight=1.0,
    )
    model.train()

    batch = _make_batch_with_split(B=2, C=C)
    loss = model.training_step(batch, 0)
    assert loss.isfinite()
    assert loss.requires_grad


def test_gate_source_penalty():
    """Gate source penalty is non-negative and adds to loss."""
    C, H_dim = 10, 8
    model_no_penalty = LitGridDA(
        in_channels=C,
        target_names=["delta_tmax"],
        source_ctx_dim=C,
        source_pay_dim=4,
        hidden_dim=H_dim,
        da_enabled=True,
        benchmark_mode=True,
        gate_source_penalty_weight=0.0,
    )
    model_penalty = LitGridDA(
        in_channels=C,
        target_names=["delta_tmax"],
        source_ctx_dim=C,
        source_pay_dim=4,
        hidden_dim=H_dim,
        da_enabled=True,
        benchmark_mode=True,
        gate_source_penalty_weight=1.0,
    )
    # Copy weights so predictions are identical
    model_penalty.load_state_dict(model_no_penalty.state_dict())
    model_no_penalty.eval()
    model_penalty.eval()

    batch = _make_batch_with_split(B=2, C=C)
    with torch.no_grad():
        pred1 = model_no_penalty(batch)
        loss1, _, _, _, _ = model_no_penalty._compute_split_train_loss(batch, pred1)
        pred2 = model_penalty(batch)
        loss2, _, _, _, _ = model_penalty._compute_split_train_loss(batch, pred2)

    # Penalty should increase loss (or be zero if gate is zero)
    assert loss2 >= loss1 - 1e-6


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_smoke_config_has_query_source_fields():
    """DA-on smoke config has source/query partition fields."""
    from models.hrrr_da.train_grid_da import GridDAConfig

    cfg = GridDAConfig.from_toml("models/hrrr_da/configs/grid_da_s_tmax_smoke.toml")
    assert cfg.train_query_frac == 0.5
    assert cfg.train_min_query_source_dist_px == 2
    assert cfg.train_source_dropout_prob == 0.2
    assert cfg.bg_loss_weight == 1.0
    assert cfg.da_query_loss_weight == 1.0
    assert cfg.gate_source_penalty_weight == 0.01


def test_da_off_config_defaults_no_partition():
    """DA-off config defaults to no source/query partition."""
    from models.hrrr_da.train_grid_da import GridDAConfig

    cfg = GridDAConfig.from_toml("models/hrrr_da/configs/grid_da_off_tmax_smoke.toml")
    assert cfg.train_query_frac == 0.0
    assert cfg.train_min_query_source_dist_px == 0
    assert cfg.train_source_dropout_prob == 0.0
