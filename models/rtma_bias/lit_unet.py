from __future__ import annotations

import lightning as L
import torch
from torch import nn
from torchmetrics import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanMetric,
    MeanSquaredError,
    R2Score,
)

from models.rtma_bias.unet import UNetSmall


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """Total-variation regularisation on a (B, C, H, W) tensor."""
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    return dx + dy


def center_pixel(x: torch.Tensor) -> torch.Tensor:
    """Extract the centre pixel: (B, C, H, W) -> (B, C)."""
    cy = x.shape[-2] // 2
    cx = x.shape[-1] // 2
    return x[:, :, cy, cx]


class LitPatchUNet(L.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base: int = 32,
        lr: float = 3e-4,
        tv_weight: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNetSmall(in_channels, out_channels, base)
        self.huber = nn.HuberLoss(delta=1.0)

        # torchmetrics: accumulation, reset, and distributed sync handled automatically.
        self.val_mae = MeanAbsoluteError()
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_r2 = R2Score()
        self.val_bias = MeanMetric()
        self.val_mape = MeanAbsolutePercentageError()
        self.baseline_mae = MeanAbsoluteError()
        self.val_rtma_mae = MeanAbsoluteError()
        self._has_rtma_baseline = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch):
        xb, yb, _meta = batch
        pred_patch = self(xb)
        pred_center = center_pixel(pred_patch)
        loss_fit = self.huber(pred_center, yb)
        loss_tv = tv_loss(pred_patch)
        loss = loss_fit + self.hparams.tv_weight * loss_tv
        return loss, pred_center, yb

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, target = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        pred_flat = pred.squeeze(-1)  # (B,)
        target_flat = target.squeeze(-1)  # (B,)
        zeros = torch.zeros_like(target_flat)

        self.val_mae.update(pred_flat, target_flat)
        self.val_rmse.update(pred_flat, target_flat)
        self.val_r2.update(pred_flat, target_flat)
        self.val_bias.update(pred_flat - target_flat)
        self.val_mape.update(pred_flat, target_flat)
        self.baseline_mae.update(zeros, target_flat)  # MAE of predicting 0

        # RTMA baseline: MAE of log(ea_rtma) vs target (when available)
        _, _, meta = batch
        if meta and "log_ea_rtma" in meta[0]:
            rtma_vals = torch.tensor(
                [m["log_ea_rtma"] for m in meta],
                dtype=torch.float32,
                device=target_flat.device,
            )
            self.val_rtma_mae.update(rtma_vals, target_flat)
            self._has_rtma_baseline = True

        return loss

    def on_validation_epoch_end(self):
        self.log("val_mae", self.val_mae, prog_bar=True)
        self.log("val_rmse", self.val_rmse)
        self.log("val_r2", self.val_r2)
        self.log("val_bias", self.val_bias)
        self.log("val_mape", self.val_mape)
        self.log("baseline_mae", self.baseline_mae)
        if self._has_rtma_baseline:
            self.log("val_rtma_mae", self.val_rtma_mae)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
