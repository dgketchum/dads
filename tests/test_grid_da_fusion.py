"""Tests for the grid DA fusion module and LitGridDA."""

from __future__ import annotations

from unittest.mock import patch

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
        "sta_is_source": torch.tensor(
            [[False, True, True, False]] * B, dtype=torch.bool
        ),
        "sta_is_query": torch.zeros(B, 4, dtype=torch.bool),
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
    """GridDAFusion produces correct output shape + coverage mask."""
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
    da_ctx, cov = fusion(
        F_grid,
        torch.randint(0, H, (B, n_src)),
        torch.randint(0, W, (B, n_src)),
        torch.randn(B, n_src, C_ctx),
        torch.randn(B, n_src, C_pay),
        torch.ones(B, n_src, dtype=torch.bool),
        torch.rand(B, 1, H, W) * 2000,
        torch.rand(B, n_src) * 2000,
    )
    assert da_ctx.shape == (B, H_dim, H, W)
    assert cov.shape == (B, 1, H, W)
    assert cov.min() >= 0 and cov.max() <= 1


def test_fusion_no_sources():
    """Fusion with no valid sources returns zeros and empty coverage."""
    fusion = GridDAFusion(
        grid_latent_dim=8, source_ctx_dim=10, source_pay_dim=4, hidden_dim=8
    )
    B, H, W = 1, 16, 16
    F_grid = torch.randn(B, 8, H, W)
    da_ctx, cov = fusion(
        F_grid,
        src_rows=torch.zeros(B, 1, dtype=torch.long),
        src_cols=torch.zeros(B, 1, dtype=torch.long),
        src_ctx=torch.randn(B, 1, 10),
        src_pay=torch.randn(B, 1, 4),
        src_valid=torch.zeros(B, 1, dtype=torch.bool),
        raw_elev_patch=torch.zeros(B, 1, H, W),
        src_elev=torch.zeros(B, 1),
    )
    assert (da_ctx == 0).all()
    assert (cov == 0).all()


def test_fusion_radius_bound():
    """DA context and coverage are zero outside support radius."""
    fusion = GridDAFusion(
        grid_latent_dim=8,
        source_ctx_dim=10,
        source_pay_dim=4,
        hidden_dim=8,
        support_radius_px=3,
    )
    B, H, W = 1, 32, 32
    F_grid = torch.randn(B, 8, H, W)
    da_ctx, cov = fusion(
        F_grid,
        src_rows=torch.tensor([[16]]),
        src_cols=torch.tensor([[16]]),
        src_ctx=torch.randn(B, 1, 10),
        src_pay=torch.randn(B, 1, 4),
        src_valid=torch.ones(B, 1, dtype=torch.bool),
        raw_elev_patch=torch.zeros(B, 1, H, W),
        src_elev=torch.zeros(B, 1),
    )
    # Far corners should be zero
    assert (da_ctx[0, :, 0, 0] == 0).all()
    assert (da_ctx[0, :, 31, 31] == 0).all()
    assert cov[0, 0, 0, 0] == 0
    assert cov[0, 0, 31, 31] == 0
    # Source pixel should be covered
    assert cov[0, 0, 16, 16] == 1
    assert not (da_ctx[0, :, 16, 16] == 0).all()


def test_fusion_casts_messages_to_grid_dtype():
    """Fusion tolerates a higher-precision softmax output before scatter-add."""
    fusion = GridDAFusion(
        grid_latent_dim=8,
        source_ctx_dim=10,
        source_pay_dim=4,
        hidden_dim=8,
        support_radius_px=3,
    )
    B, H, W = 1, 16, 16
    F_grid = torch.randn(B, 8, H, W, dtype=torch.float32)

    with patch(
        "models.hrrr_da.grid_da_fusion.pyg_softmax",
        side_effect=lambda src, index: torch.ones_like(src, dtype=torch.float64),
    ):
        da_ctx, cov = fusion(
            F_grid,
            src_rows=torch.tensor([[8]]),
            src_cols=torch.tensor([[8]]),
            src_ctx=torch.randn(B, 1, 10),
            src_pay=torch.randn(B, 1, 4),
            src_valid=torch.ones(B, 1, dtype=torch.bool),
            raw_elev_patch=torch.zeros(B, 1, H, W),
            src_elev=torch.zeros(B, 1),
        )

    assert da_ctx.dtype == F_grid.dtype
    assert cov.dtype == F_grid.dtype


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
        F_grid = model_off.backbone(batch["x_patch"], return_features=True)
        expected = model_off.bg_head(F_grid)

    assert torch.allclose(pred, expected, atol=1e-6)


def test_da_on_no_sources_equals_background():
    """DA-on with no valid sources equals background-only prediction."""
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
    batch["src_valid"] = torch.zeros(1, 1, dtype=torch.bool)  # no valid sources

    with torch.no_grad():
        pred_da = model(batch)
        F_grid = model.backbone(batch["x_patch"], return_features=True)
        pred_bg = model.bg_head(F_grid)

    assert torch.allclose(pred_da, pred_bg, atol=1e-6), (
        f"No-source DA-on should match background. Max diff: {(pred_da - pred_bg).abs().max()}"
    )


def test_da_on_radius_isolation():
    """DA-on with one source: far-corner prediction matches background."""
    C, H_dim = 10, 8
    model = LitGridDA(
        in_channels=C,
        target_names=["delta_tmax"],
        source_ctx_dim=C,
        source_pay_dim=4,
        hidden_dim=H_dim,
        da_enabled=True,
        support_radius_px=3,
        benchmark_mode=False,
    )
    model.eval()

    batch = _make_dummy_batch(B=1, C=C, H=32, W=32, n_src=1, pay_dim=4)
    batch["src_rows"] = torch.tensor([[16]])
    batch["src_cols"] = torch.tensor([[16]])

    with torch.no_grad():
        pred_da = model(batch)
        F_grid = model.backbone(batch["x_patch"], return_features=True)
        pred_bg = model.bg_head(F_grid)

    # Far corners should match background exactly (coverage_mask = 0 there)
    assert torch.allclose(pred_da[0, :, 0, 0], pred_bg[0, :, 0, 0], atol=1e-6), (
        "Far corner should equal background"
    )
    assert torch.allclose(pred_da[0, :, 31, 31], pred_bg[0, :, 31, 31], atol=1e-6), (
        "Far corner should equal background"
    )


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


def test_unet_return_features():
    """UNetSmall return_features returns latent, not prediction."""
    unet = UNetSmall(in_channels=10, base=16, n_heads=1)
    x = torch.randn(1, 10, 64, 64)

    features = unet(x, return_features=True)
    pred = unet(x, return_features=False)

    assert features.shape == (1, 16, 64, 64)
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
    batch["src_rows"] = torch.tensor([[8]])
    batch["src_cols"] = torch.tensor([[8]])

    with torch.no_grad():
        pred1 = model(batch).clone()

    batch["src_pay"] = batch["src_pay"] + 10.0
    with torch.no_grad():
        pred2 = model(batch)

    diff = (pred1 - pred2).abs()
    assert diff[0, 0, 8, 8] > 0, "Prediction at source should change"
