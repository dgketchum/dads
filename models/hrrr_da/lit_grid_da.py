"""Lightning module for grid backbone + DA fusion (Stage C).

Two forward modes:
- ``da_enabled=True``:  full system (background + gated DA residual)
- ``da_enabled=False``: exact Stage B bypass (background head only)

Weight-loading contract (Stage B → C):
- ``self.backbone`` loaded from Stage B ``model.*`` keys
- ``self.bg_head``   loaded from Stage B ``model.out.*`` or ``model.heads.0.*``
- DA components (``da_fusion``, ``da_head``, ``gate_head``) initialised fresh
- Gate bias = −2.0 (conservative DA start)
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError

from models.hrrr_da.grid_da_fusion import GridDAFusion
from models.rtma_bias.lit_unet import tv_loss
from models.rtma_bias.unet import UNetSmall


class LitGridDA(L.LightningModule):
    """Grid backbone with optional source/query DA fusion.

    Parameters
    ----------
    in_channels : int
        Number of input raster channels.
    target_names : list[str]
        Target names (one output head per target).
    source_ctx_dim : int
        Source context feature dimension (typically == in_channels).
    source_pay_dim : int
        Source payload dimension (n_payload_cols * 2).
    hidden_dim : int
        UNet base channel width and DA hidden dim.
    da_enabled : bool
        If False, the DA branch is never executed (exact Stage B bypass).
    lr : float
        Learning rate.
    tv_weight : float
        Total-variation regularisation weight.
    huber_delta : float
        Huber loss delta.
    dense_loss_weight : float
        Weight on optional dense teacher loss (0 = disabled).
    support_radius_px : int
        DA support radius in pixels (1 px = 1 km).
    da_gate_init_bias : float
        Initial bias for the DA gate (negative = conservative start).
    benchmark_mode : bool
        If True, compute holdout-station benchmark MAE.
    """

    def __init__(
        self,
        in_channels: int,
        target_names: list[str],
        source_ctx_dim: int,
        source_pay_dim: int,
        hidden_dim: int = 32,
        da_enabled: bool = True,
        lr: float = 1e-4,
        tv_weight: float = 1e-3,
        huber_delta: float = 1.0,
        dense_loss_weight: float = 0.0,
        support_radius_px: int = 16,
        da_gate_init_bias: float = -2.0,
        benchmark_mode: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.target_names = target_names
        self.da_enabled = da_enabled
        self.lr = lr
        self.tv_weight = tv_weight
        self.huber_delta = huber_delta
        self.dense_loss_weight = dense_loss_weight
        n_targets = len(target_names)
        self.benchmark_mode = benchmark_mode

        # Grid backbone (shares namespace with LitPatchAssim / LitDensePretraining)
        self.backbone = UNetSmall(
            in_channels=in_channels,
            base=hidden_dim,
            n_heads=1,  # unused — we use return_features
        )

        # Background head (initialised from Stage B output head)
        self.bg_head = torch.nn.Conv2d(hidden_dim, n_targets, kernel_size=1)

        if da_enabled:
            self.da_fusion = GridDAFusion(
                grid_latent_dim=hidden_dim,
                source_ctx_dim=source_ctx_dim,
                source_pay_dim=source_pay_dim,
                hidden_dim=hidden_dim,
                support_radius_px=support_radius_px,
            )
            self.da_head = torch.nn.Conv2d(
                hidden_dim + hidden_dim, n_targets, kernel_size=1
            )
            self.gate_head = torch.nn.Conv2d(
                hidden_dim + hidden_dim, n_targets, kernel_size=1
            )
            # Initialise gate bias conservatively
            with torch.no_grad():
                self.gate_head.bias.fill_(da_gate_init_bias)

        if benchmark_mode:
            self.val_mae = MeanAbsoluteError()
            self._val_baseline_ae_sum = 0.0
            self._val_baseline_count = 0

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["x_patch"]
        F_grid = self.backbone(x, return_features=True)  # (B, base, H, W)
        bg_increment = self.bg_head(F_grid)  # (B, n_targets, H, W)

        if not self.da_enabled:
            return bg_increment

        da_ctx = self.da_fusion(
            F_grid,
            batch["src_rows"],
            batch["src_cols"],
            batch["src_ctx"],
            batch["src_pay"],
            batch["src_valid"],
            batch["raw_elev_patch"],
            batch["src_elev"],
        )
        fused = torch.cat([F_grid, da_ctx], dim=1)
        da_residual = self.da_head(fused)
        da_gate = torch.sigmoid(self.gate_head(fused))
        self._last_gate_mean = da_gate.mean()
        return bg_increment + da_gate * da_residual

    def _gather_at_stations(self, pred, sta_rows, sta_cols):
        """Sample predictions at station pixel locations."""
        B, n_t, H, W = pred.shape
        pred_flat = pred.view(B, n_t, H * W)
        sta_flat = (sta_rows * W + sta_cols).long()
        sta_flat_exp = sta_flat.unsqueeze(1).expand(B, n_t, -1)
        return pred_flat.gather(2, sta_flat_exp).permute(0, 2, 1)  # (B, N_sta, n_t)

    def _compute_loss(self, batch: dict, pred: torch.Tensor, is_train: bool):
        sta_rows = batch["sta_rows"]
        sta_cols = batch["sta_cols"]
        sta_targets = batch["sta_targets"]
        sta_valid = batch["sta_valid"]
        sta_holdout = batch["sta_holdout"]

        pred_at_sta = self._gather_at_stations(pred, sta_rows, sta_cols)

        # Station query loss
        if self.benchmark_mode:
            if is_train:
                mask = sta_valid & ~sta_holdout.unsqueeze(-1).expand_as(sta_valid)
            else:
                mask = sta_valid & sta_holdout.unsqueeze(-1).expand_as(sta_valid)
        else:
            mask = sta_valid

        if mask.any():
            query_loss = F.huber_loss(
                pred_at_sta[mask],
                sta_targets[mask],
                delta=self.huber_delta,
                reduction="mean",
            )
        else:
            query_loss = pred.sum() * 0.0

        loss = query_loss + self.tv_weight * tv_loss(pred)

        # Optional dense teacher loss
        if self.dense_loss_weight > 0 and batch.get("y_dense") is not None:
            y_dense = batch["y_dense"]
            y_vmask = batch["y_valid_dense"]
            if y_vmask.any():
                dense_loss = F.huber_loss(
                    pred[:, : y_dense.shape[1]][y_vmask],
                    y_dense[y_vmask],
                    delta=self.huber_delta,
                    reduction="mean",
                )
                loss = loss + self.dense_loss_weight * dense_loss

        return loss, pred_at_sta, mask

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss, _, _ = self._compute_loss(batch, pred, is_train=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.da_enabled and hasattr(self, "_last_gate_mean"):
            self.log("da_gate_mean", self._last_gate_mean, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss, pred_at_sta, mask = self._compute_loss(batch, pred, is_train=False)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        # Per-target MAE
        sta_targets = batch["sta_targets"]
        for i, name in enumerate(self.target_names):
            mask_i = mask[:, :, i]
            if mask_i.any():
                mae_i = (
                    (pred_at_sta[:, :, i][mask_i] - sta_targets[:, :, i][mask_i])
                    .abs()
                    .mean()
                )
                self.log(f"val/mae_{name}", mae_i, on_epoch=True, prog_bar=(i == 0))

        # Benchmark: deduplicated center-station MAE
        if self.benchmark_mode:
            sta_is_center = batch["sta_is_center"]
            center_mask = mask[:, :, 0] & sta_is_center
            if center_mask.any():
                self.val_mae.update(
                    pred_at_sta[:, :, 0][center_mask],
                    sta_targets[:, :, 0][center_mask],
                )
                self._val_baseline_ae_sum += (
                    sta_targets[:, :, 0][center_mask].abs().sum().item()
                )
                self._val_baseline_count += center_mask.sum().item()

        if self.da_enabled and hasattr(self, "_last_gate_mean"):
            self.log(
                "val/da_gate_mean", self._last_gate_mean, on_epoch=True, prog_bar=True
            )

        return loss

    def on_validation_epoch_end(self):
        if self.benchmark_mode:
            self.log("val/target_mae", self.val_mae, prog_bar=True)
            if self._val_baseline_count > 0:
                bl = self._val_baseline_ae_sum / self._val_baseline_count
                self.log("val/baseline_mae", bl)
                self.log("val/center_station_count", float(self._val_baseline_count))
            self._val_baseline_ae_sum = 0.0
            self._val_baseline_count = 0

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
