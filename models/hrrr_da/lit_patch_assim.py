"""Lightning module for grid-core-v0 patch residual correction."""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError

from models.rtma_bias.lit_unet import tv_loss
from models.rtma_bias.unet import UNetSmall


class LitPatchAssim(L.LightningModule):
    """
    Raster-patch HRRR bias correction with holdout-aware sparse loss.

    Inputs:  (B, C, H, W) raster patch
    Outputs: (B, n_targets, H, W) correction field

    Loss: Huber at station pixel locations + TV regularization on full field.
    Train loss uses non-holdout stations; val loss uses holdout stations.
    """

    def __init__(
        self,
        in_channels: int,
        target_names: list[str],
        hidden_dim: int = 32,
        lr: float = 3e-4,
        tv_weight: float = 1e-3,
        huber_delta: float = 1.0,
        benchmark_mode: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.target_names = target_names
        self.tv_weight = tv_weight
        self.huber_delta = huber_delta
        self.lr = lr
        self.benchmark_mode = benchmark_mode
        self.model = UNetSmall(
            in_channels=in_channels,
            base=hidden_dim,
            n_heads=len(target_names),
        )

        if benchmark_mode:
            self.val_mae = MeanAbsoluteError()
            self._val_baseline_ae_sum = 0.0
            self._val_baseline_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, list):
            out = torch.cat(out, dim=1)
        return out

    def _gather_at_stations(self, pred, sta_rows, sta_cols):
        """Sample predictions at station pixel locations."""
        B, n_t, H, W = pred.shape
        pred_flat = pred.view(B, n_t, H * W)
        sta_flat = (sta_rows * W + sta_cols).long()
        sta_flat_exp = sta_flat.unsqueeze(1).expand(B, n_t, -1)
        return pred_flat.gather(2, sta_flat_exp).permute(0, 2, 1)  # (B, N_sta, n_t)

    def _compute_loss(self, pred_at_sta, sta_targets, mask, pred_field):
        """Huber loss at masked stations + TV regularization."""
        if mask.any():
            loss_sta = F.huber_loss(
                pred_at_sta[mask],
                sta_targets[mask],
                delta=self.huber_delta,
                reduction="mean",
            )
        else:
            loss_sta = pred_field.sum() * 0.0
        return loss_sta + self.tv_weight * tv_loss(pred_field)

    def training_step(self, batch, batch_idx):
        x, sta_rows, sta_cols, sta_targets, sta_valid, sta_holdout, sta_is_center = (
            batch
        )
        pred = self(x)
        pred_at_sta = self._gather_at_stations(pred, sta_rows, sta_cols)

        # Train: non-holdout stations only
        if self.benchmark_mode:
            mask = sta_valid & ~sta_holdout.unsqueeze(-1).expand_as(sta_valid)
        else:
            mask = sta_valid

        loss = self._compute_loss(pred_at_sta, sta_targets, mask, pred)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, sta_rows, sta_cols, sta_targets, sta_valid, sta_holdout, sta_is_center = (
            batch
        )
        pred = self(x)
        pred_at_sta = self._gather_at_stations(pred, sta_rows, sta_cols)

        # Val: holdout stations only
        if self.benchmark_mode:
            mask = sta_valid & sta_holdout.unsqueeze(-1).expand_as(sta_valid)
        else:
            mask = sta_valid

        loss = self._compute_loss(pred_at_sta, sta_targets, mask, pred)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        # Per-target MAE and benchmark metrics
        for i, name in enumerate(self.target_names):
            mask_i = mask[:, :, i]
            if mask_i.any():
                mae_i = (
                    (pred_at_sta[:, :, i][mask_i] - sta_targets[:, :, i][mask_i])
                    .abs()
                    .mean()
                )
                self.log(f"val/mae_{name}", mae_i, on_epoch=True, prog_bar=(i == 0))

        # Benchmark: deduplicated MAE using center station only
        if self.benchmark_mode:
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

        return loss

    def on_validation_epoch_end(self):
        if self.benchmark_mode:
            self.log("val/target_mae", self.val_mae, prog_bar=True)
            if self._val_baseline_count > 0:
                bl = self._val_baseline_ae_sum / self._val_baseline_count
                self.log("val/baseline_mae", bl)
                self.log(
                    "val/center_station_count",
                    float(self._val_baseline_count),
                )
            self._val_baseline_ae_sum = 0.0
            self._val_baseline_count = 0

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
