from __future__ import annotations

import lightning as L
import torch
from torch import nn
from torchmetrics import MeanAbsoluteError, MeanMetric, MeanSquaredError, R2Score

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
        self.baseline_mae = MeanAbsoluteError()

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
        self.baseline_mae.update(zeros, target_flat)  # MAE of predicting 0

        return loss

    def on_validation_epoch_end(self):
        self.log("val_mae", self.val_mae, prog_bar=True)
        self.log("val_rmse", self.val_rmse)
        self.log("val_r2", self.val_r2)
        self.log("val_bias", self.val_bias)
        self.log("baseline_mae", self.baseline_mae)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
