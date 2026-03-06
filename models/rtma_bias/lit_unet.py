from __future__ import annotations

import lightning as L
import torch
from torch import nn
from torchmetrics import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanMetric,
    MeanSquaredError,
    PearsonCorrCoef,
    R2Score,
)

from models.components.tasks import get_task_adapter
from models.rtma_bias.unet import UNetSmall


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """Total-variation regularisation on a (B, C, H, W) tensor."""
    if x.shape[-1] <= 1 and x.shape[-2] <= 1:
        return torch.tensor(0.0, device=x.device)
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    return dx + dy


def center_pixel(x: torch.Tensor) -> torch.Tensor:
    """Extract the centre pixel: (B, C, H, W) -> (B, C)."""
    cy = x.shape[-2] // 2
    cx = x.shape[-1] // 2
    return x[:, :, cy, cx]


def cc_loss(
    pred_delta_log_ea: torch.Tensor,
    pred_delta_tmax: torch.Tensor,
    log_ea_rtma: torch.Tensor,
    tmp_rtma: torch.Tensor,
) -> torch.Tensor:
    """Clausius-Clapeyron physics penalty (center pixel only)."""
    ea_corrected = torch.exp(log_ea_rtma + pred_delta_log_ea)  # kPa
    tmax_corrected = tmp_rtma + pred_delta_tmax  # °C
    e_s = 0.6108 * torch.exp(17.27 * tmax_corrected / (tmax_corrected + 237.3))
    return torch.relu(ea_corrected - e_s).mean()


class LitPatchUNet(L.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base: int = 32,
        lr: float = 3e-4,
        tv_weight: float = 1e-3,
        n_heads: int = 1,
        task_weights: list[float] | None = None,
        physics_weight: float = 0.0,
        task: str = "ea",
        pair_head_idx: int | None = None,
        use_pairwise_loss: bool = False,
        pair_loss_weight: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_heads = int(n_heads)
        self.task_adapter = get_task_adapter(task)

        self.model = UNetSmall(in_channels, out_channels, base, n_heads=self.n_heads)
        self.huber = nn.HuberLoss(delta=1.0, reduction="none")

        if task_weights is not None:
            self._task_weights = list(task_weights)
        else:
            self._task_weights = [1.0] * self.n_heads

        self._multi = self.n_heads > 1
        if self._multi:
            default_head = self.task_adapter.default_head_idx
            resolved = default_head if pair_head_idx is None else int(pair_head_idx)
            self._pair_head_idx = max(0, min(self.n_heads - 1, resolved))
        else:
            self._pair_head_idx = 0

        # --- ea metrics ---
        self.val_ea_mae = MeanAbsoluteError()
        self.val_ea_rmse = MeanSquaredError(squared=False)
        self.val_ea_r2 = R2Score()
        self.val_ea_bias = MeanMetric()
        self.val_ea_mape = MeanAbsolutePercentageError()
        self.baseline_mae = MeanAbsoluteError()
        self.val_rtma_mae = MeanAbsoluteError()

        # --- ea-space metrics (kPa) ---
        self.val_mae_ea = MeanAbsoluteError()
        self.val_mae_rtma_ea = MeanAbsoluteError()
        self.val_mape_ea = MeanAbsolutePercentageError()
        self.val_mape_rtma_ea = MeanAbsolutePercentageError()
        self._ea_mae_sum = MeanMetric()
        self._rtma_mae_sum = MeanMetric()
        self._has_rtma_baseline = False

        # --- tmax metrics ---
        self.val_tmax_mae = MeanAbsoluteError()
        self.val_tmax_rmse = MeanSquaredError(squared=False)
        self.val_tmax_r2 = R2Score()
        self.val_tmax_bias = MeanMetric()
        self.val_cc_violation = MeanMetric()

        # --- tmin metrics ---
        self.val_tmin_mae = MeanAbsoluteError()
        self.val_tmin_rmse = MeanSquaredError(squared=False)
        self.val_tmin_r2 = R2Score()
        self.val_tmin_bias = MeanMetric()

        # --- wind metrics ---
        self.val_wind_mae = MeanAbsoluteError()
        self.val_wind_rmse = MeanSquaredError(squared=False)
        self.val_wind_r2 = R2Score()
        self.val_wind_bias = MeanMetric()

        # --- pairwise metrics ---
        self.use_pairwise_loss = bool(use_pairwise_loss)
        self.val_pair_mae = MeanAbsoluteError()
        self.val_pair_r2 = R2Score()
        self.val_pair_corr = PearsonCorrCoef()
        self._has_pair_updates = False

        self._pair_train_loader = None
        self._pair_val_loader = None
        self._pair_train_iter = None
        self._pair_val_iter = None

    def attach_pair_dataloaders(self, train_loader=None, val_loader=None) -> None:
        self._pair_train_loader = train_loader
        self._pair_val_loader = val_loader

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        return self.model(x)

    def _next_pair_batch(self, train: bool):
        loader = self._pair_train_loader if train else self._pair_val_loader
        if loader is None:
            return None
        itr = self._pair_train_iter if train else self._pair_val_iter
        if itr is None:
            itr = iter(loader)
        try:
            batch = next(itr)
        except StopIteration:
            itr = iter(loader)
            try:
                batch = next(itr)
            except StopIteration:
                return None
        if train:
            self._pair_train_iter = itr
        else:
            self._pair_val_iter = itr
        return batch

    def _task_from_output(
        self, pred_out: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor:
        if not self._multi:
            return center_pixel(pred_out).squeeze(-1)
        assert isinstance(pred_out, list)
        head = self._pair_head_idx
        return center_pixel(pred_out[head]).squeeze(-1)

    def _task_from_target(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            return y
        if y.shape[1] == 1:
            return y[:, 0]
        return y[:, self._pair_head_idx]

    def _pair_loss_and_metrics(self, pair_batch):
        if pair_batch is None:
            return None
        xi, yi, _meta_i, xj, yj, _meta_j, _pair_meta = pair_batch
        device = self.device
        xi = xi.to(device)
        yi = yi.to(device)
        xj = xj.to(device)
        yj = yj.to(device)

        out_i = self(xi)
        out_j = self(xj)
        pred_i = self._task_from_output(out_i)
        pred_j = self._task_from_output(out_j)
        tgt_i = self._task_from_target(yi)
        tgt_j = self._task_from_target(yj)

        pred_pair = pred_i - pred_j
        tgt_pair = tgt_i - tgt_j
        valid = torch.isfinite(pred_pair) & torch.isfinite(tgt_pair)
        if not valid.any():
            return None

        pred_v = pred_pair[valid]
        tgt_v = tgt_pair[valid]
        pair_loss = self.huber(pred_v, tgt_v).mean()
        return pair_loss, pred_v, tgt_v

    # ------------------------------------------------------------------
    # Shared step (supports single-head and multi-head)
    # ------------------------------------------------------------------

    def _shared_step(self, batch):
        xb, yb, meta = batch
        pred_out = self(xb)

        if not self._multi:
            pred_center = center_pixel(pred_out)  # (B, 1)
            loss_fit = self.huber(pred_center, yb).mean()
            loss_tv = tv_loss(pred_out)
            loss = loss_fit + self.hparams.tv_weight * loss_tv
            return loss, pred_center, yb, meta

        total_loss = torch.tensor(0.0, device=xb.device)
        pred_centers = []
        for h, pred_patch in enumerate(pred_out):
            pc = center_pixel(pred_patch)  # (B, 1)
            pred_centers.append(pc)
            target_h = yb[:, h : h + 1]  # (B, 1)
            valid = torch.isfinite(target_h)
            if valid.any():
                per_sample = self.huber(pc, torch.where(valid, target_h, pc))
                masked = per_sample * valid.float()
                total_loss = (
                    total_loss + self._task_weights[h] * masked.sum() / valid.sum()
                )
            total_loss = total_loss + self.hparams.tv_weight * tv_loss(pred_patch)

        # CC physics penalty only for ea-centric multitask setup.
        if (
            self.hparams.physics_weight > 0
            and meta
            and self.task_adapter.supports_cc_physics
        ):
            has_both = all("log_ea_rtma" in m and "tmp_rtma" in m for m in meta)
            if has_both and len(pred_centers) > 1:
                log_ea_rtma = torch.tensor(
                    [m["log_ea_rtma"] for m in meta],
                    dtype=torch.float32,
                    device=xb.device,
                )
                tmp_rtma = torch.tensor(
                    [m["tmp_rtma"] for m in meta],
                    dtype=torch.float32,
                    device=xb.device,
                )
                pc_ea = pred_centers[0].squeeze(-1)
                pc_tmax = pred_centers[1].squeeze(-1)
                phys = cc_loss(pc_ea, pc_tmax, log_ea_rtma, tmp_rtma)
                total_loss = total_loss + self.hparams.physics_weight * phys

        return total_loss, pred_centers, yb, meta

    def on_train_epoch_start(self):
        if self.use_pairwise_loss and self._pair_train_loader is not None:
            self._pair_train_iter = iter(self._pair_train_loader)

    def on_validation_epoch_start(self):
        self._has_pair_updates = False
        self._has_rtma_baseline = False
        if self.use_pairwise_loss and self._pair_val_loader is not None:
            self._pair_val_iter = iter(self._pair_val_loader)

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._shared_step(batch)

        if self.use_pairwise_loss:
            pair_out = self._pair_loss_and_metrics(self._next_pair_batch(train=True))
            if pair_out is not None:
                pair_loss, _pred_pair, _tgt_pair = pair_out
                loss = loss + self.hparams.pair_loss_weight * pair_loss
                self.log("train/pair_loss", pair_loss, prog_bar=False)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, target, meta = self._shared_step(batch)

        if self.use_pairwise_loss:
            pair_out = self._pair_loss_and_metrics(self._next_pair_batch(train=False))
            if pair_out is not None:
                pair_loss, pred_pair, tgt_pair = pair_out
                loss = loss + self.hparams.pair_loss_weight * pair_loss
                self.log("val/pair_loss", pair_loss, prog_bar=False, sync_dist=True)
                self.val_pair_mae.update(pred_pair, tgt_pair)
                self.val_pair_r2.update(pred_pair, tgt_pair)
                self.val_pair_corr.update(pred_pair, tgt_pair)
                self._has_pair_updates = True

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        if not self._multi:
            pred_flat = pred.squeeze(-1)
            target_flat = target.squeeze(-1)
            zeros = torch.zeros_like(target_flat)

            if self.task_adapter.name == "tmax":
                self.val_tmax_mae.update(pred_flat, target_flat)
                self.val_tmax_rmse.update(pred_flat, target_flat)
                self.val_tmax_r2.update(pred_flat, target_flat)
                self.val_tmax_bias.update(pred_flat - target_flat)
                self.baseline_mae.update(zeros, target_flat)
            else:
                self.val_ea_mae.update(pred_flat, target_flat)
                self.val_ea_rmse.update(pred_flat, target_flat)
                self.val_ea_r2.update(pred_flat, target_flat)
                self.val_ea_bias.update(pred_flat - target_flat)
                self.val_ea_mape.update(pred_flat, target_flat)
                self.baseline_mae.update(zeros, target_flat)

                if meta and "log_ea_rtma" in meta[0]:
                    self._update_ea_space_metrics(pred_flat, target_flat, meta)
        else:
            self._update_multi_head_metrics(pred, target, meta)

        return loss

    def _update_ea_space_metrics(self, pred_flat, target_flat, meta):
        """Update ea-kPa-space metrics from log-space predictions."""
        log_ea_rtma = torch.tensor(
            [m["log_ea_rtma"] for m in meta],
            dtype=torch.float32,
            device=target_flat.device,
        )
        # In residual-log space, raw RTMA baseline is zero correction.
        self.val_rtma_mae.update(torch.zeros_like(target_flat), target_flat)
        self._has_rtma_baseline = True

        ea_obs = torch.exp(log_ea_rtma + target_flat)
        ea_rtma = torch.exp(log_ea_rtma)
        ea_corrected = torch.exp(log_ea_rtma + pred_flat)

        self.val_mae_ea.update(ea_corrected, ea_obs)
        self.val_mae_rtma_ea.update(ea_rtma, ea_obs)
        self.val_mape_ea.update(ea_corrected, ea_obs)
        self.val_mape_rtma_ea.update(ea_rtma, ea_obs)

        self._ea_mae_sum.update(torch.abs(ea_corrected - ea_obs))
        self._rtma_mae_sum.update(torch.abs(ea_rtma - ea_obs))

    def _update_multi_head_metrics(self, pred_centers, target, meta):
        """Update per-head metrics for multi-task mode."""
        device = target.device

        # Head 0: ea (delta_log_ea)
        pc_ea = pred_centers[0].squeeze(-1)
        tgt_ea = target[:, 0]
        ea_valid = torch.isfinite(tgt_ea)
        if ea_valid.any():
            p_ea = pc_ea[ea_valid]
            t_ea = tgt_ea[ea_valid]
            self.val_ea_mae.update(p_ea, t_ea)
            self.val_ea_rmse.update(p_ea, t_ea)
            self.val_ea_r2.update(p_ea, t_ea)
            self.val_ea_bias.update(p_ea - t_ea)
            self.baseline_mae.update(torch.zeros_like(t_ea), t_ea)

            has_ea_rtma = [
                i
                for i, (m, v) in enumerate(zip(meta, ea_valid.tolist()))
                if v and "log_ea_rtma" in m
            ]
            if has_ea_rtma:
                idx = torch.tensor(has_ea_rtma, device=device)
                sub_pred = pc_ea[idx]
                sub_tgt = tgt_ea[idx]
                sub_meta = [meta[i] for i in has_ea_rtma]
                self._update_ea_space_metrics(sub_pred, sub_tgt, sub_meta)

        # Head 1: tmax (delta_tmax)
        if len(pred_centers) > 1 and target.shape[1] > 1:
            pc_tmax = pred_centers[1].squeeze(-1)
            tgt_tmax = target[:, 1]
            tmax_valid = torch.isfinite(tgt_tmax)
            if tmax_valid.any():
                p_tmax = pc_tmax[tmax_valid]
                t_tmax = tgt_tmax[tmax_valid]
                self.val_tmax_mae.update(p_tmax, t_tmax)
                self.val_tmax_rmse.update(p_tmax, t_tmax)
                self.val_tmax_r2.update(p_tmax, t_tmax)
                self.val_tmax_bias.update(p_tmax - t_tmax)

            has_both_meta = [
                i for i, m in enumerate(meta) if "log_ea_rtma" in m and "tmp_rtma" in m
            ]
            if has_both_meta:
                idx = torch.tensor(has_both_meta, device=device)
                log_ea_rtma = torch.tensor(
                    [meta[i]["log_ea_rtma"] for i in has_both_meta],
                    dtype=torch.float32,
                    device=device,
                )
                tmp_rtma = torch.tensor(
                    [meta[i]["tmp_rtma"] for i in has_both_meta],
                    dtype=torch.float32,
                    device=device,
                )
                ea_corr = torch.exp(log_ea_rtma + pc_ea[idx])
                tmax_corr = tmp_rtma + pc_tmax[idx]
                e_s = 0.6108 * torch.exp(17.27 * tmax_corr / (tmax_corr + 237.3))
                violation = torch.relu(ea_corr - e_s).mean()
                self.val_cc_violation.update(violation)

        # Head 2: tmin (delta_tmin)
        if len(pred_centers) > 2 and target.shape[1] > 2:
            pc_tmin = pred_centers[2].squeeze(-1)
            tgt_tmin = target[:, 2]
            tmin_valid = torch.isfinite(tgt_tmin)
            if tmin_valid.any():
                p_tmin = pc_tmin[tmin_valid]
                t_tmin = tgt_tmin[tmin_valid]
                self.val_tmin_mae.update(p_tmin, t_tmin)
                self.val_tmin_rmse.update(p_tmin, t_tmin)
                self.val_tmin_r2.update(p_tmin, t_tmin)
                self.val_tmin_bias.update(p_tmin - t_tmin)

        # Head 3: wind (delta_wind)
        if len(pred_centers) > 3 and target.shape[1] > 3:
            pc_wind = pred_centers[3].squeeze(-1)
            tgt_wind = target[:, 3]
            wind_valid = torch.isfinite(tgt_wind)
            if wind_valid.any():
                p_wind = pc_wind[wind_valid]
                t_wind = tgt_wind[wind_valid]
                self.val_wind_mae.update(p_wind, t_wind)
                self.val_wind_rmse.update(p_wind, t_wind)
                self.val_wind_r2.update(p_wind, t_wind)
                self.val_wind_bias.update(p_wind - t_wind)

    def _safe_log_r2(self, name: str, metric: R2Score, **kwargs) -> None:
        """Log an R2Score only if it has >= 2 samples (required by torchmetrics)."""
        if metric.total >= 2:
            self.log(name, metric, **kwargs)

    def on_validation_epoch_end(self):
        if not self._multi and self.task_adapter.name == "tmax":
            self.log("val/tmax_mae", self.val_tmax_mae, prog_bar=True)
            self.log("val/tmax_rmse", self.val_tmax_rmse)
            self._safe_log_r2("val/tmax_r2", self.val_tmax_r2)
            self.log("val/tmax_bias", self.val_tmax_bias)
            self.log("baseline_mae", self.baseline_mae)
            self.log("val_mae", self.val_tmax_mae.compute())
            if self.val_tmax_r2.total >= 2:
                self.log("val_r2", self.val_tmax_r2.compute())
        else:
            self.log("val/ea_mae", self.val_ea_mae, prog_bar=True)
            self.log("val/ea_rmse", self.val_ea_rmse)
            self._safe_log_r2("val/ea_r2", self.val_ea_r2)
            self.log("val/ea_bias", self.val_ea_bias)
            self.log("baseline_mae", self.baseline_mae)

            if not self._multi:
                self.log("val_mae", self.val_ea_mae.compute())
                if self.val_ea_r2.total >= 2:
                    self.log("val_r2", self.val_ea_r2.compute())

            if self._has_rtma_baseline:
                self.log("val_rtma_mae", self.val_rtma_mae)
                self.log("val/mae_ea_kpa", self.val_mae_ea)
                self.log("val/mae_rtma_ea_kpa", self.val_mae_rtma_ea)
                self.log("val/mape_ea", self.val_mape_ea)
                self.log("val/mape_rtma_ea", self.val_mape_rtma_ea)

                model_mae = self._ea_mae_sum.compute()
                rtma_mae = self._rtma_mae_sum.compute()
                if rtma_mae > 0:
                    pct = (1.0 - model_mae / rtma_mae) * 100.0
                    self.log("val/pct_improvement", pct)
                self._ea_mae_sum.reset()
                self._rtma_mae_sum.reset()

        if self._multi:
            self.log("val/tmax_mae", self.val_tmax_mae)
            self.log("val/tmax_rmse", self.val_tmax_rmse)
            self._safe_log_r2("val/tmax_r2", self.val_tmax_r2)
            self.log("val/tmax_bias", self.val_tmax_bias)
            self.log("val/cc_violation", self.val_cc_violation)

            if self.n_heads > 2:
                self.log("val/tmin_mae", self.val_tmin_mae)
                self.log("val/tmin_rmse", self.val_tmin_rmse)
                self._safe_log_r2("val/tmin_r2", self.val_tmin_r2)
                self.log("val/tmin_bias", self.val_tmin_bias)

            if self.n_heads > 3:
                self.log("val/wind_mae", self.val_wind_mae)
                self.log("val/wind_rmse", self.val_wind_rmse)
                self._safe_log_r2("val/wind_r2", self.val_wind_r2)
                self.log("val/wind_bias", self.val_wind_bias)

        if self.use_pairwise_loss and self._has_pair_updates:
            prefix = self.task_adapter.metric_prefix
            self.log(f"val/{prefix}_pair_mae", self.val_pair_mae)
            self._safe_log_r2(f"val/{prefix}_pair_r2", self.val_pair_r2)
            self.log(f"val/{prefix}_pair_corr", self.val_pair_corr)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=7, min_lr=1e-6
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }
