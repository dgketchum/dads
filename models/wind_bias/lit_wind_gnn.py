"""
Lightning module for wind bias correction GNN.

Targets: (delta_w_par, delta_w_perp)
Loss: Huber with calm-wind weighting.
Metrics: speed MAE, direction MAE, vector RMSE, pct_improvement.
"""

from __future__ import annotations

import lightning as L
import torch
from torch import nn
from torchmetrics import MeanAbsoluteError, MeanMetric

from models.components.edge_gated_gnn import EdgeGatedGNN


class LitWindGNN(L.LightningModule):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 7,
        hidden_dim: int = 64,
        n_hops: int = 1,
        use_graph: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        huber_delta: float = 2.0,
        calm_threshold: float = 2.0,
        calm_min_weight: float = 0.1,
        correction_penalty: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = EdgeGatedGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            n_hops=n_hops,
            use_graph=use_graph,
        )
        self.huber = nn.HuberLoss(delta=huber_delta, reduction="none")

        # Val metrics — speed
        self.val_mae_speed = MeanAbsoluteError()
        self.val_mae_speed_rtma = MeanAbsoluteError()

        # Val metrics — par/perp
        self.val_mae_par = MeanAbsoluteError()
        self.val_mae_perp = MeanAbsoluteError()

        # Vector RMSE
        self.val_vector_mse_sum = MeanMetric()

        # Direction MAE (filtered to obs speed >= 2)
        self._dir_ae_sum = 0.0
        self._dir_count = 0

        # Pct improvement accumulators
        self._speed_mae_sum = MeanMetric()
        self._rtma_mae_sum = MeanMetric()

    def forward(self, data):
        return self.model(
            data.x,
            edge_index=getattr(data, "edge_index", None),
            edge_attr=getattr(data, "edge_attr", None),
        )

    def _compute_loss(self, pred, target, rtma_wind):
        """Huber loss with calm-wind weighting."""
        loss_per_sample = self.huber(pred, target).mean(dim=-1)  # (N,)

        # Calm weighting: w = clamp(rtma_wind / threshold, min_w, 1.0)
        w = torch.clamp(
            rtma_wind / self.hparams.calm_threshold,
            min=self.hparams.calm_min_weight,
            max=1.0,
        )
        loss = (loss_per_sample * w).mean()

        # Optional correction magnitude penalty
        if self.hparams.correction_penalty > 0:
            loss = loss + self.hparams.correction_penalty * pred.pow(2).mean()

        return loss

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self._compute_loss(pred, batch.y, batch.rtma_wind)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_nodes)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self._compute_loss(pred, batch.y, batch.rtma_wind)
        self.log(
            "val_loss", loss, prog_bar=True, batch_size=batch.num_nodes, sync_dist=True
        )

        # Par/perp MAE
        self.val_mae_par.update(pred[:, 0], batch.y[:, 0])
        self.val_mae_perp.update(pred[:, 1], batch.y[:, 1])

        # Vector RMSE (pred vs target residuals in par/perp space)
        diff = pred - batch.y
        vector_mse = (diff**2).sum(dim=-1)
        self.val_vector_mse_sum.update(vector_mse)

        # Reconstruct corrected u/v for speed and direction metrics.
        # pred = (delta_par, delta_perp) in the RTMA wind-aligned basis.
        # e_par = (ugrd_rtma, vgrd_rtma) / |rtma_wind|
        # e_perp = (-vgrd_rtma, ugrd_rtma) / |rtma_wind|
        # delta_u = delta_par * e_par_x + delta_perp * e_perp_x
        # corrected_u = ugrd_rtma + delta_u  (same for v)
        eps = 1e-6
        rtma_spd = batch.rtma_wind.clamp(min=eps)
        e_par_x = batch.ugrd_rtma / rtma_spd
        e_par_y = batch.vgrd_rtma / rtma_spd
        e_perp_x = -e_par_y
        e_perp_y = e_par_x

        pred_du = pred[:, 0] * e_par_x + pred[:, 1] * e_perp_x
        pred_dv = pred[:, 0] * e_par_y + pred[:, 1] * e_perp_y
        u_corr = batch.ugrd_rtma + pred_du
        v_corr = batch.vgrd_rtma + pred_dv

        speed_corr = torch.sqrt(u_corr**2 + v_corr**2)
        speed_obs = torch.sqrt(batch.u_obs**2 + batch.v_obs**2)

        # Speed MAE: corrected vs obs, and RTMA baseline vs obs
        self.val_mae_speed.update(speed_corr, speed_obs)
        self.val_mae_speed_rtma.update(rtma_spd, speed_obs)
        self._speed_mae_sum.update(torch.abs(speed_corr - speed_obs))
        self._rtma_mae_sum.update(torch.abs(rtma_spd - speed_obs))

        # Direction MAE: only where obs speed >= 2 m/s
        fast = speed_obs >= 2.0
        if fast.any():
            dir_corr = torch.atan2(u_corr[fast], v_corr[fast])
            dir_obs = torch.atan2(batch.u_obs[fast], batch.v_obs[fast])
            dir_diff = dir_corr - dir_obs
            # Wrap to [-pi, pi]
            dir_diff = (dir_diff + torch.pi) % (2 * torch.pi) - torch.pi
            dir_ae = torch.abs(dir_diff) * (180.0 / torch.pi)
            self._dir_ae_sum += dir_ae.sum().item()
            self._dir_count += int(fast.sum().item())

        return loss

    def on_validation_epoch_end(self):
        self.log("val/mae_par", self.val_mae_par, prog_bar=True)
        self.log("val/mae_perp", self.val_mae_perp)

        # Vector RMSE = sqrt(mean of squared vector errors)
        mean_mse = self.val_vector_mse_sum.compute()
        self.log("val/vector_rmse", torch.sqrt(mean_mse))
        self.val_vector_mse_sum.reset()

        # Speed MAE
        self.log("val/speed_mae", self.val_mae_speed)
        self.log("val/baseline_speed_mae", self.val_mae_speed_rtma)

        # Pct improvement
        model_mae = self._speed_mae_sum.compute()
        rtma_mae = self._rtma_mae_sum.compute()
        pct = 1.0 - model_mae / rtma_mae.clamp(min=1e-6)
        self.log("val/pct_improvement", pct)
        self._speed_mae_sum.reset()
        self._rtma_mae_sum.reset()

        # Direction MAE (deg)
        if self._dir_count > 0:
            self.log(
                "val/direction_mae_deg",
                self._dir_ae_sum / self._dir_count,
            )
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
