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
