"""Lightning module for the E0 patch assimilation model."""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F

from models.rtma_bias.lit_unet import tv_loss
from models.rtma_bias.unet import UNetSmall


class LitPatchAssim(L.LightningModule):
    """
    Raster-patch HRRR bias correction (E0: no observation channels).

    Inputs:  (B, C, H, W) raster patch
    Outputs: (B, n_targets, H, W) correction field

    Loss: Huber at station pixel locations + TV regularization on full field.
    """

    def __init__(
        self,
        in_channels: int,
        target_names: list[str],
        hidden_dim: int = 32,
        lr: float = 3e-4,
        tv_weight: float = 1e-3,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.target_names = target_names
        self.tv_weight = tv_weight
        self.huber_delta = huber_delta
        self.lr = lr
        self.model = UNetSmall(
            in_channels=in_channels,
            base=hidden_dim,
            n_heads=len(target_names),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, list):
            out = torch.cat(out, dim=1)  # (B, n_targets, H, W)
        return out

    def _shared_step(
        self,
        batch: tuple,
        log_per_target: bool = False,
    ) -> torch.Tensor:
        x, sta_rows, sta_cols, sta_targets, sta_valid = batch
        pred = self(x)  # (B, n_targets, H, W)

        B, n_t, H, W = pred.shape

        # Gather predictions at station pixel locations
        pred_flat = pred.view(B, n_t, H * W)
        sta_flat = (sta_rows * W + sta_cols).long()  # (B, N_sta)
        sta_flat_exp = sta_flat.unsqueeze(1).expand(B, n_t, -1)  # (B, n_t, N_sta)
        pred_at_sta = pred_flat.gather(2, sta_flat_exp).permute(
            0, 2, 1
        )  # (B, N_sta, n_t)

        # Huber loss at valid (station, target) pairs
        if sta_valid.any():
            loss_sta = F.huber_loss(
                pred_at_sta[sta_valid],
                sta_targets[sta_valid],
                delta=self.huber_delta,
                reduction="mean",
            )
        else:
            loss_sta = pred.sum() * 0.0

        loss = loss_sta + self.tv_weight * tv_loss(pred)

        if log_per_target:
            for i, name in enumerate(self.target_names):
                mask_i = sta_valid[:, :, i]
                if mask_i.any():
                    mae_i = (
                        (pred_at_sta[:, :, i][mask_i] - sta_targets[:, :, i][mask_i])
                        .abs()
                        .mean()
                    )
                    self.log(f"val/mae_{name}", mae_i, on_epoch=True, prog_bar=(i == 0))

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, log_per_target=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
