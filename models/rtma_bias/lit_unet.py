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

        # --- log-space metrics (continuity with prior runs) ---
        self.val_mae = MeanAbsoluteError()
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_r2 = R2Score()
        self.val_bias = MeanMetric()
        self.val_mape = MeanAbsolutePercentageError()
        self.baseline_mae = MeanAbsoluteError()
        self.val_rtma_mae = MeanAbsoluteError()

        # --- ea-space metrics (kPa, stakeholder-friendly) ---
        self.val_mae_ea = MeanAbsoluteError()
        self.val_mae_rtma_ea = MeanAbsoluteError()
        self.val_mape_ea = MeanAbsolutePercentageError()
        self.val_mape_rtma_ea = MeanAbsolutePercentageError()
        self._ea_mae_sum = MeanMetric()
        self._rtma_mae_sum = MeanMetric()

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

        # log-space metrics
        self.val_mae.update(pred_flat, target_flat)
        self.val_rmse.update(pred_flat, target_flat)
        self.val_r2.update(pred_flat, target_flat)
        self.val_bias.update(pred_flat - target_flat)
        self.val_mape.update(pred_flat, target_flat)
        self.baseline_mae.update(zeros, target_flat)

        # ea-space metrics (requires log_ea_rtma in metadata)
        _, _, meta = batch
        if meta and "log_ea_rtma" in meta[0]:
            log_ea_rtma = torch.tensor(
                [m["log_ea_rtma"] for m in meta],
                dtype=torch.float32,
                device=target_flat.device,
            )
            self.val_rtma_mae.update(log_ea_rtma, target_flat)
            self._has_rtma_baseline = True

            # reconstruct ea values in kPa
            ea_obs = torch.exp(log_ea_rtma + target_flat)
            ea_rtma = torch.exp(log_ea_rtma)
            ea_corrected = torch.exp(log_ea_rtma + pred_flat)

            self.val_mae_ea.update(ea_corrected, ea_obs)
            self.val_mae_rtma_ea.update(ea_rtma, ea_obs)
            self.val_mape_ea.update(ea_corrected, ea_obs)
            self.val_mape_rtma_ea.update(ea_rtma, ea_obs)

            # accumulate raw MAE values for pct_improvement
            self._ea_mae_sum.update(torch.abs(ea_corrected - ea_obs))
            self._rtma_mae_sum.update(torch.abs(ea_rtma - ea_obs))

        return loss

    def on_validation_epoch_end(self):
        # log-space metrics
        self.log("val_mae", self.val_mae, prog_bar=True)
        self.log("val_rmse", self.val_rmse)
        self.log("val_r2", self.val_r2)
        self.log("val_bias", self.val_bias)
        self.log("val_mape", self.val_mape)
        self.log("baseline_mae", self.baseline_mae)

        if self._has_rtma_baseline:
            self.log("val_rtma_mae", self.val_rtma_mae)

            # ea-space metrics
            self.log("val/mae_ea_kpa", self.val_mae_ea)
            self.log("val/mae_rtma_ea_kpa", self.val_mae_rtma_ea)
            self.log("val/mape_ea", self.val_mape_ea)
            self.log("val/mape_rtma_ea", self.val_mape_rtma_ea)

            # pct improvement
            model_mae = self._ea_mae_sum.compute()
            rtma_mae = self._rtma_mae_sum.compute()
            if rtma_mae > 0:
                pct = (1.0 - model_mae / rtma_mae) * 100.0
                self.log("val/pct_improvement", pct)
            self._ea_mae_sum.reset()
            self._rtma_mae_sum.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=7, min_lr=1e-6
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }
