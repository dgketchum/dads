"""
Unified Lightning module for DADS bias-correction GNN.

Supports two tasks:
  - "scalar": single-target bias correction (tmax, EA, etc.)
  - "wind": dual-target wind bias correction (delta_par, delta_perp)

Each task defines its own loss computation and validation metrics.
"""

from __future__ import annotations

import lightning as L
import torch
from torch import nn
from torchmetrics import MeanAbsoluteError, MeanMetric, MeanSquaredError, R2Score

from models.components.edge_gated_gnn import EdgeGatedGNN


class LitDadsGNN(L.LightningModule):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 7,
        hidden_dim: int = 64,
        n_hops: int = 1,
        use_graph: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        huber_delta: float = 1.0,
        dropout: float = 0.0,
        task: str = "scalar",
        # Wind-specific
        calm_threshold: float = 2.0,
        calm_min_weight: float = 0.1,
        correction_penalty: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        out_dim = 1 if task == "scalar" else 2
        self.model = EdgeGatedGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            n_hops=n_hops,
            use_graph=use_graph,
            out_dim=out_dim,
            dropout=dropout,
        )

        if task == "scalar":
            self.huber = nn.HuberLoss(delta=huber_delta, reduction="mean")
            self.val_mae = MeanAbsoluteError()
            self.val_rmse = MeanSquaredError(squared=False)
            self.val_r2 = R2Score()
            self.val_bias_sum = 0.0
            self.val_bias_count = 0
            self.val_baseline_ae_sum = 0.0
            self.val_baseline_count = 0
        else:
            self.huber = nn.HuberLoss(delta=huber_delta, reduction="none")
            self.val_mae_speed = MeanAbsoluteError()
            self.val_baseline_speed_mae = MeanAbsoluteError()
            self.val_mae_par = MeanAbsoluteError()
            self.val_mae_perp = MeanAbsoluteError()
            self.val_vector_mse_sum = MeanMetric()
            self._dir_ae_sum = 0.0
            self._dir_count = 0
            self._speed_mae_sum = MeanMetric()
            self._rtma_mae_sum = MeanMetric()

    def forward(self, data):
        return self.model(
            data.x,
            edge_index=getattr(data, "edge_index", None),
            edge_attr=getattr(data, "edge_attr", None),
        )

    # ---- Loss ----

    def _scalar_loss(self, pred, batch):
        pred = pred.squeeze(-1)
        mask = batch.loss_mask
        return self.huber(pred[mask], batch.y[mask]), mask.sum().item()

    def _wind_loss(self, pred, batch):
        loss_per_sample = self.huber(pred, batch.y).mean(dim=-1)
        w = torch.clamp(
            batch.baseline_wind / self.hparams.calm_threshold,
            min=self.hparams.calm_min_weight,
            max=1.0,
        )
        loss = (loss_per_sample * w).mean()
        if self.hparams.correction_penalty > 0:
            loss = loss + self.hparams.correction_penalty * pred.pow(2).mean()
        return loss, batch.num_nodes

    # ---- Training ----

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        if self.hparams.task == "scalar":
            loss, bs = self._scalar_loss(pred, batch)
        else:
            loss, bs = self._wind_loss(pred, batch)
        self.log("train_loss", loss, prog_bar=True, batch_size=bs)
        return loss

    # ---- Validation ----

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        if self.hparams.task == "scalar":
            return self._scalar_val_step(pred, batch)
        else:
            return self._wind_val_step(pred, batch)

    def _scalar_val_step(self, pred, batch):
        pred = pred.squeeze(-1)
        mask = batch.loss_mask
        pred_m, target_m = pred[mask], batch.y[mask]
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
        self.val_bias_sum += (pred_m - target_m).sum().item()
        self.val_bias_count += target_m.numel()
        self.val_baseline_ae_sum += target_m.abs().sum().item()
        self.val_baseline_count += target_m.numel()
        return loss

    def _wind_val_step(self, pred, batch):
        loss_per_sample = self.huber(pred, batch.y).mean(dim=-1)
        w = torch.clamp(
            batch.baseline_wind / self.hparams.calm_threshold,
            min=self.hparams.calm_min_weight,
            max=1.0,
        )
        loss = (loss_per_sample * w).mean()
        if self.hparams.correction_penalty > 0:
            loss = loss + self.hparams.correction_penalty * pred.pow(2).mean()
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            batch_size=batch.num_nodes,
            sync_dist=True,
        )

        self.val_mae_par.update(pred[:, 0], batch.y[:, 0])
        self.val_mae_perp.update(pred[:, 1], batch.y[:, 1])

        diff = pred - batch.y
        self.val_vector_mse_sum.update((diff**2).sum(dim=-1))

        eps = 1e-6
        rtma_spd = batch.baseline_wind.clamp(min=eps)
        e_par_x = batch.ugrd_baseline / rtma_spd
        e_par_y = batch.vgrd_baseline / rtma_spd
        e_perp_x = -e_par_y
        e_perp_y = e_par_x

        u_corr = batch.ugrd_baseline + pred[:, 0] * e_par_x + pred[:, 1] * e_perp_x
        v_corr = batch.vgrd_baseline + pred[:, 0] * e_par_y + pred[:, 1] * e_perp_y
        speed_corr = torch.sqrt(u_corr**2 + v_corr**2)
        speed_obs = torch.sqrt(batch.u_obs**2 + batch.v_obs**2)

        self.val_mae_speed.update(speed_corr, speed_obs)
        self.val_baseline_speed_mae.update(rtma_spd, speed_obs)
        self._speed_mae_sum.update(torch.abs(speed_corr - speed_obs))
        self._rtma_mae_sum.update(torch.abs(rtma_spd - speed_obs))

        fast = speed_obs >= 2.0
        if fast.any():
            dir_corr = torch.atan2(u_corr[fast], v_corr[fast])
            dir_obs = torch.atan2(batch.u_obs[fast], batch.v_obs[fast])
            dir_diff = (dir_corr - dir_obs + torch.pi) % (2 * torch.pi) - torch.pi
            dir_ae = torch.abs(dir_diff) * (180.0 / torch.pi)
            self._dir_ae_sum += dir_ae.sum().item()
            self._dir_count += int(fast.sum().item())

        return loss

    def on_validation_epoch_end(self):
        if self.hparams.task == "scalar":
            self._scalar_val_epoch_end()
        else:
            self._wind_val_epoch_end()

    def _scalar_val_epoch_end(self):
        self.log("val/target_mae", self.val_mae, prog_bar=True)
        self.log("val/target_rmse", self.val_rmse)
        self.log("val/target_r2", self.val_r2)
        if self.val_bias_count > 0:
            self.log("val/target_bias", self.val_bias_sum / self.val_bias_count)
            self.log(
                "val/baseline_mae", self.val_baseline_ae_sum / self.val_baseline_count
            )
        self.val_bias_sum = 0.0
        self.val_bias_count = 0
        self.val_baseline_ae_sum = 0.0
        self.val_baseline_count = 0

    def _wind_val_epoch_end(self):
        self.log("val/mae_par", self.val_mae_par, prog_bar=True)
        self.log("val/mae_perp", self.val_mae_perp)
        mean_mse = self.val_vector_mse_sum.compute()
        self.log("val/vector_rmse", torch.sqrt(mean_mse))
        self.val_vector_mse_sum.reset()
        self.log("val/speed_mae", self.val_mae_speed)
        self.log("val/baseline_speed_mae", self.val_baseline_speed_mae)
        model_mae = self._speed_mae_sum.compute()
        rtma_mae = self._rtma_mae_sum.compute()
        self.log("val/pct_improvement", 1.0 - model_mae / rtma_mae.clamp(min=1e-6))
        self._speed_mae_sum.reset()
        self._rtma_mae_sum.reset()
        if self._dir_count > 0:
            self.log("val/direction_mae_deg", self._dir_ae_sum / self._dir_count)
        self._dir_ae_sum = 0.0
        self._dir_count = 0

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
