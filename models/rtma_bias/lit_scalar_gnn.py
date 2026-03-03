"""
Lightning module for scalar bias-correction GNN (tmax, EA, etc.).

Target: 1 scalar per node (delta_tmax, delta_log_ea, ...)
Loss: Huber (delta=1.0)
Metrics: target_mae, target_rmse, target_r2, target_bias, baseline_mae
"""

from __future__ import annotations

import lightning as L
import torch
from torch import nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

from models.wind_bias.gnn import WindBiasGNN


class LitScalarGNN(L.LightningModule):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 7,
        hidden_dim: int = 128,
        n_hops: int = 1,
        use_graph: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        huber_delta: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = WindBiasGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            n_hops=n_hops,
            use_graph=use_graph,
            out_dim=1,
            dropout=dropout,
        )
        self.huber = nn.HuberLoss(delta=huber_delta, reduction="mean")

        # Val metrics
        self.val_mae = MeanAbsoluteError()
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_r2 = R2Score()
        self.val_bias_sum = 0.0
        self.val_bias_count = 0
        self.val_baseline_ae_sum = 0.0
        self.val_baseline_count = 0

    def forward(self, data):
        return self.model(
            data.x,
            edge_index=getattr(data, "edge_index", None),
            edge_attr=getattr(data, "edge_attr", None),
        )

    def training_step(self, batch, batch_idx):
        pred = self(batch).squeeze(-1)
        mask = batch.loss_mask
        loss = self.huber(pred[mask], batch.y[mask])
        self.log("train_loss", loss, prog_bar=True, batch_size=mask.sum().item())
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch).squeeze(-1)
        mask = batch.loss_mask
        pred_m = pred[mask]
        target_m = batch.y[mask]
        loss = self.huber(pred_m, target_m)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            batch_size=mask.sum().item(),
            sync_dist=True,
        )

        self.val_mae.update(pred_m, target_m)
        self.val_rmse.update(pred_m, target_m)
        self.val_r2.update(pred_m, target_m)

        # Bias: mean(pred - target)
        self.val_bias_sum += (pred_m - target_m).sum().item()
        self.val_bias_count += target_m.numel()

        # Baseline MAE: MAE of the uncorrected gridded product
        # (baseline prediction is 0 correction)
        self.val_baseline_ae_sum += target_m.abs().sum().item()
        self.val_baseline_count += target_m.numel()

        return loss

    def on_validation_epoch_end(self):
        self.log("val/target_mae", self.val_mae, prog_bar=True)
        self.log("val/target_rmse", self.val_rmse)
        self.log("val/target_r2", self.val_r2)

        if self.val_bias_count > 0:
            bias = self.val_bias_sum / self.val_bias_count
            self.log("val/target_bias", bias)
            baseline_mae = self.val_baseline_ae_sum / self.val_baseline_count
            self.log("val/baseline_mae", baseline_mae)

        self.val_bias_sum = 0.0
        self.val_bias_count = 0
        self.val_baseline_ae_sum = 0.0
        self.val_baseline_count = 0

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }
