"""Tests for the grid DA fusion module and LitGridDA."""

from __future__ import annotations

import torch

from models.hrrr_da.grid_da_fusion import GridDAFusion
from models.hrrr_da.lit_grid_da import LitGridDA
from models.rtma_bias.unet import UNetSmall


def _make_dummy_batch(B=2, C=10, H=16, W=16, n_src=3, pay_dim=4, n_targets=1) -> dict:
    """Create a synthetic batch dict matching GridDAPatchDataset output."""
    return {
        "x_patch": torch.randn(B, C, H, W),
        "sta_rows": torch.randint(0, H, (B, 4)),
        "sta_cols": torch.randint(0, W, (B, 4)),
        "sta_targets": torch.randn(B, 4, n_targets),
        "sta_valid": torch.ones(B, 4, n_targets, dtype=torch.bool),
        "sta_holdout": torch.tensor([[True, False, False, True]] * B, dtype=torch.bool),
        "sta_is_center": torch.tensor(
            [[True, False, False, False]] * B, dtype=torch.bool
        ),
        "src_rows": torch.randint(0, H, (B, n_src)),
        "src_cols": torch.randint(0, W, (B, n_src)),
        "src_ctx": torch.randn(B, n_src, C),
        "src_pay": torch.randn(B, n_src, pay_dim),
        "src_valid": torch.ones(B, n_src, dtype=torch.bool),
        "raw_elev_patch": torch.rand(B, 1, H, W) * 2000,
        "src_elev": torch.rand(B, n_src) * 2000,
        "y_dense": None,
        "y_valid_dense": None,
    }


def test_fusion_shapes():
    """GridDAFusion produces correct output shape."""
    C_latent, C_ctx, C_pay, H_dim = 8, 10, 4, 8
    fusion = GridDAFusion(
        grid_latent_dim=C_latent,
        source_ctx_dim=C_ctx,
        source_pay_dim=C_pay,
        hidden_dim=H_dim,
        support_radius_px=4,
    )
    B, H, W, n_src = 2, 16, 16, 3
    F_grid = torch.randn(B, C_latent, H, W)
    src_rows = torch.randint(0, H, (B, n_src))
    src_cols = torch.randint(0, W, (B, n_src))
    src_ctx = torch.randn(B, n_src, C_ctx)
    src_pay = torch.randn(B, n_src, C_pay)
    src_valid = torch.ones(B, n_src, dtype=torch.bool)
    raw_elev = torch.rand(B, 1, H, W) * 2000
    src_elev = torch.rand(B, n_src) * 2000

    out = fusion(
        F_grid, src_rows, src_cols, src_ctx, src_pay, src_valid, raw_elev, src_elev
    )
    assert out.shape == (B, H_dim, H, W)


def test_fusion_no_sources():
    """Fusion with no valid sources returns zeros."""
    fusion = GridDAFusion(
        grid_latent_dim=8, source_ctx_dim=10, source_pay_dim=4, hidden_dim=8
    )
    B, H, W = 1, 16, 16
    F_grid = torch.randn(B, 8, H, W)
    out = fusion(
        F_grid,
        src_rows=torch.zeros(B, 1, dtype=torch.long),
        src_cols=torch.zeros(B, 1, dtype=torch.long),
        src_ctx=torch.randn(B, 1, 10),
        src_pay=torch.randn(B, 1, 4),
        src_valid=torch.zeros(B, 1, dtype=torch.bool),  # no valid sources
        raw_elev_patch=torch.zeros(B, 1, H, W),
        src_elev=torch.zeros(B, 1),
    )
    assert (out == 0).all()


def test_fusion_radius_bound():
    """DA context is zero outside support radius."""
    fusion = GridDAFusion(
        grid_latent_dim=8,
        source_ctx_dim=10,
        source_pay_dim=4,
        hidden_dim=8,
        support_radius_px=3,
    )
    B, H, W = 1, 32, 32
    F_grid = torch.randn(B, 8, H, W)
    # Place source at (16, 16)
    out = fusion(
        F_grid,
        src_rows=torch.tensor([[16]]),
        src_cols=torch.tensor([[16]]),
        src_ctx=torch.randn(B, 1, 10),
        src_pay=torch.randn(B, 1, 4),
        src_valid=torch.ones(B, 1, dtype=torch.bool),
        raw_elev_patch=torch.zeros(B, 1, H, W),
        src_elev=torch.zeros(B, 1),
    )
    # Pixels far from (16, 16) should be zero
    assert (out[0, :, 0, 0] == 0).all(), "Corner should be zero"
    assert (out[0, :, 31, 31] == 0).all(), "Far corner should be zero"
    # Pixel at source should be nonzero
    assert not (out[0, :, 16, 16] == 0).all(), "Source pixel should have DA context"


def test_da_bypass_exact():
    """DA-off mode produces bg_head(backbone(x)) only."""
    C, H_dim = 10, 8
    model_off = LitGridDA(
        in_channels=C,
        target_names=["delta_tmax"],
        source_ctx_dim=C,
        source_pay_dim=4,
        hidden_dim=H_dim,
        da_enabled=False,
        benchmark_mode=False,
    )

    batch = _make_dummy_batch(B=1, C=C, H=16, W=16, n_src=2, pay_dim=4)
    model_off.eval()
    with torch.no_grad():
        pred = model_off(batch)

    # Manually compute bg_head(backbone(x))
    with torch.no_grad():
        F_grid = model_off.backbone(batch["x_patch"], return_features=True)
        expected = model_off.bg_head(F_grid)

    assert torch.allclose(pred, expected, atol=1e-6)


def test_da_on_forward():
    """DA-on mode runs without error and logs gate mean."""
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

    batch = _make_dummy_batch(B=2, C=C, H=16, W=16, n_src=3, pay_dim=4)
    model.eval()
    with torch.no_grad():
        pred = model(batch)

    assert pred.shape == (2, 1, 16, 16)
    assert hasattr(model, "_last_gate_mean")
    # Gate starts near sigmoid(-2.0) ≈ 0.12
    assert 0.0 < model._last_gate_mean.item() < 0.5


def test_unet_return_features():
    """UNetSmall return_features returns latent, not prediction."""
    unet = UNetSmall(in_channels=10, base=16, n_heads=1)
    x = torch.randn(1, 10, 64, 64)

    features = unet(x, return_features=True)
    pred = unet(x, return_features=False)

    # Features should be (B, base, H, W)
    assert features.shape == (1, 16, 64, 64)
    # Prediction should be (B, 1, H, W) — different from features
    assert pred.shape == (1, 1, 64, 64)
    assert features.shape != pred.shape


def test_payload_change_affects_nearby():
    """Changing source payload changes predictions at nearby pixels."""
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

    batch = _make_dummy_batch(B=1, C=C, H=16, W=16, n_src=1, pay_dim=4)
    # Place source at centre
    batch["src_rows"] = torch.tensor([[8]])
    batch["src_cols"] = torch.tensor([[8]])

    with torch.no_grad():
        pred1 = model(batch).clone()

    # Change payload
    batch["src_pay"] = batch["src_pay"] + 10.0
    with torch.no_grad():
        pred2 = model(batch)

    # Near source should change
    diff = (pred1 - pred2).abs()
    assert diff[0, 0, 8, 8] > 0, "Prediction at source should change"
