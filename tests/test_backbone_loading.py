"""Tests for backbone weight transfer between training stages."""

from __future__ import annotations

import torch

from models.hrrr_da.lit_dense_pretrain import LitDensePretraining
from models.hrrr_da.lit_patch_assim import LitPatchAssim
from models.rtma_bias.unet import UNetSmall


def test_namespace_contract():
    """LitDensePretraining and LitPatchAssim both store UNet as self.model."""
    stage_a = LitDensePretraining(in_channels=10, target_names=["delta_tmax"])
    stage_b = LitPatchAssim(
        in_channels=10, target_names=["delta_tmax"], benchmark_mode=False
    )
    assert isinstance(stage_a.model, UNetSmall)
    assert isinstance(stage_b.model, UNetSmall)


def test_stage_a_to_b_backbone_transfer():
    """Stage A backbone weights transfer to Stage B, heads are reinitialized."""
    in_ch = 10
    stage_a = LitDensePretraining(
        in_channels=in_ch, target_names=["delta_tmax"], hidden_dim=16
    )

    # Simulate a Lightning checkpoint state_dict
    sd = stage_a.state_dict()

    # Stage B has a different number of heads (e.g., 1 head via n_heads=1)
    stage_b = LitPatchAssim(
        in_channels=in_ch,
        target_names=["delta_tmax"],
        hidden_dim=16,
        benchmark_mode=False,
    )

    # Filter out output head keys (same logic as train_patch_assim.py)
    backbone_sd = {k: v for k, v in sd.items() if "out." not in k and "heads." not in k}

    missing, unexpected = stage_b.load_state_dict(backbone_sd, strict=False)

    # Missing keys should only be output head keys + benchmark metrics
    for key in missing:
        assert (
            "out." in key
            or "heads." in key
            or "val_mae" in key
            or "val_baseline" in key
        ), f"Unexpected missing key: {key}"

    # No unexpected keys
    assert len(unexpected) == 0

    # Verify backbone weights match
    for name, param in stage_a.model.named_parameters():
        if "out." not in name and "heads." not in name:
            b_param = dict(stage_b.model.named_parameters())[name]
            assert torch.equal(param, b_param), f"Mismatch in {name}"


def test_stage_a_to_b_forward_runs():
    """A Stage B model loaded with Stage A backbone weights runs forward."""
    in_ch = 10
    stage_a = LitDensePretraining(
        in_channels=in_ch, target_names=["delta_tmax"], hidden_dim=16
    )

    sd = stage_a.state_dict()
    backbone_sd = {k: v for k, v in sd.items() if "out." not in k and "heads." not in k}

    stage_b = LitPatchAssim(
        in_channels=in_ch,
        target_names=["delta_tmax"],
        hidden_dim=16,
        benchmark_mode=False,
    )
    stage_b.load_state_dict(backbone_sd, strict=False)

    x = torch.randn(2, in_ch, 64, 64)
    out = stage_b(x)
    assert out.shape == (2, 1, 64, 64)


def test_multitarget_transfer():
    """Stage A with 1 head -> Stage B with 1 head (single-target)."""
    in_ch = 10
    stage_a = LitDensePretraining(
        in_channels=in_ch, target_names=["delta_tmax"], hidden_dim=16
    )
    sd = stage_a.state_dict()
    backbone_sd = {k: v for k, v in sd.items() if "out." not in k and "heads." not in k}

    # Stage B: same target
    stage_b = LitPatchAssim(
        in_channels=in_ch,
        target_names=["delta_tmax"],
        hidden_dim=16,
        benchmark_mode=True,
    )
    stage_b.load_state_dict(backbone_sd, strict=False)

    x = torch.randn(1, in_ch, 64, 64)
    out = stage_b(x)
    assert out.shape == (1, 1, 64, 64)


def test_stage_b_to_c_da_off_parity():
    """LitGridDA with DA-off and Stage B weights matches LitPatchAssim output."""
    from models.hrrr_da.lit_grid_da import LitGridDA

    in_ch = 10
    stage_b = LitPatchAssim(
        in_channels=in_ch,
        target_names=["delta_tmax"],
        hidden_dim=16,
        benchmark_mode=False,
    )

    # Create LitGridDA with DA-off
    stage_c = LitGridDA(
        in_channels=in_ch,
        target_names=["delta_tmax"],
        source_ctx_dim=in_ch,
        source_pay_dim=4,
        hidden_dim=16,
        da_enabled=False,
        benchmark_mode=False,
    )

    # Transfer weights: Stage B model.* -> Stage C backbone.* + bg_head.*
    sd = stage_b.state_dict()
    c_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            inner = k[len("model.") :]
            if inner.startswith("out."):
                c_sd[f"bg_head.{inner[len('out.') :]}"] = v
            elif inner.startswith("heads."):
                continue  # skip multi-head keys
            else:
                c_sd[f"backbone.{inner}"] = v
    stage_c.load_state_dict(c_sd, strict=False)

    # Forward pass comparison
    x = torch.randn(2, in_ch, 32, 32)
    stage_b.eval()
    stage_c.eval()
    with torch.no_grad():
        out_b = stage_b(x)
        batch = {
            "x_patch": x,
            "src_rows": torch.zeros(2, 1, dtype=torch.long),
            "src_cols": torch.zeros(2, 1, dtype=torch.long),
            "src_ctx": torch.zeros(2, 1, in_ch),
            "src_pay": torch.zeros(2, 1, 4),
            "src_valid": torch.zeros(2, 1, dtype=torch.bool),
            "raw_elev_patch": torch.zeros(2, 1, 32, 32),
            "src_elev": torch.zeros(2, 1),
        }
        out_c = stage_c(batch)

    assert torch.allclose(out_b, out_c, atol=1e-6), (
        f"DA-off should match Stage B. Max diff: {(out_b - out_c).abs().max()}"
    )
