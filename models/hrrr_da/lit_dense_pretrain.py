"""Lightning module for grid backbone dense pretraining (Stage A).

Trains ``UNetSmall`` to predict the dense ``URMA − HRRR`` increment field
at every pixel in each tile.  The loss is Huber on valid pixels plus TV
regularisation for spatial smoothness.

**Namespace contract**: the backbone is stored as ``self.model`` so that
Lightning checkpoint keys match ``LitPatchAssim`` (``lit_patch_assim.py:42``),
enabling ``strict=False`` weight loading across stages.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError

from models.rtma_bias.lit_unet import tv_loss
from models.rtma_bias.unet import UNetSmall


class LitDensePretraining(L.LightningModule):
    """Dense pretraining on URMA − HRRR increment fields.

    Parameters
    ----------
    in_channels : int
        Number of input raster channels.
    target_names : list[str]
        Names for each output head / target band (e.g. ``["delta_tmax"]``).
    hidden_dim : int
        UNet base channel width.
    lr : float
        Learning rate for AdamW.
    tv_weight : float
        Weight on total-variation regularisation.
    huber_delta : float
        Delta parameter for Huber loss.
    """

    def __init__(
        self,
        in_channels: int,
        target_names: list[str],
        hidden_dim: int = 32,
        lr: float = 3e-4,
        tv_weight: float = 1e-4,
        huber_delta: float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.target_names = target_names
        self.lr = lr
        self.tv_weight = tv_weight
        self.huber_delta = huber_delta

        # Attribute name ``model`` is a hard contract — see docstring.
        self.model = UNetSmall(
            in_channels=in_channels,
            base=hidden_dim,
            n_heads=len(target_names),
        )

        self.val_mae = torch.nn.ModuleDict(
            {name: MeanAbsoluteError() for name in target_names}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, list):
            out = torch.cat(out, dim=1)
        return out

    def _compute_dense_loss(
        self,
        pred: torch.Tensor,
        y_dense: torch.Tensor,
        y_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Huber loss on valid pixels + TV regularisation."""
        loss = pred.new_tensor(0.0)
        n_valid_targets = 0
        for i in range(pred.shape[1]):
            mask = y_valid[:, i]
            if mask.any():
                loss = loss + F.huber_loss(
                    pred[:, i][mask],
                    y_dense[:, i][mask],
                    delta=self.huber_delta,
                    reduction="mean",
                )
                n_valid_targets += 1
        if n_valid_targets > 0:
            loss = loss / n_valid_targets
        loss = loss + self.tv_weight * tv_loss(pred)
        return loss

    def training_step(self, batch, batch_idx):
        x, y_dense, y_valid = batch
        pred = self(x)
        loss = self._compute_dense_loss(pred, y_dense, y_valid)
        self.log("train_dense_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_dense, y_valid = batch
        pred = self(x)
        loss = self._compute_dense_loss(pred, y_dense, y_valid)
        self.log("val_dense_loss", loss, on_epoch=True, prog_bar=True)

        for i, name in enumerate(self.target_names):
            mask = y_valid[:, i]
            if mask.any():
                self.val_mae[name].update(pred[:, i][mask], y_dense[:, i][mask])

        return loss

    def on_validation_epoch_end(self):
        for name in self.target_names:
            self.log(f"val/mae_{name}", self.val_mae[name], prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
