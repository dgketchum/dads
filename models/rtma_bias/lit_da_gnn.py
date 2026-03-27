"""Lightning module for DA benchmark GNN.

Supports both da-graph-v0 (HeteroEdgeGatedGNN) and da-graph-v1 (GatedDAGNN).
Predictions on query nodes only. Loss gated by loss_mask + valid_mask.
"""

from __future__ import annotations

import lightning as L
import torch
from torch import nn
from torchmetrics import MeanAbsoluteError

from models.components.gated_da_gnn import GatedDAGNN
from models.components.hetero_edge_gated_gnn import HeteroEdgeGatedGNN


class LitDAGNN(L.LightningModule):
    def __init__(
        self,
        query_node_dim: int,
        source_context_dim: int,
        source_payload_dim: int = 0,
        edge_dim: int = 7,
        hidden_dim: int = 64,
        n_hops: int = 1,
        dropout: float = 0.3,
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
        huber_delta: float = 1.0,
        task: str = "scalar",
        target_names: list[str] | None = None,
        target_scales: list[float] | None = None,
        disable_payload: bool = False,
        da_gate_init_bias: float = -2.0,
        source_edge_dropout: float = 0.0,
        da_version: str = "v1",
    ):
        super().__init__()
        self.save_hyperparameters()

        out_dim = 1 if task == "scalar" else len(target_names or [])

        if da_version == "v1":
            self.model = GatedDAGNN(
                query_node_dim=query_node_dim,
                source_context_dim=source_context_dim,
                source_payload_dim=source_payload_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                n_hops=n_hops,
                out_dim=out_dim,
                dropout=dropout,
                da_gate_init_bias=da_gate_init_bias,
            )
        else:
            # v0 fallback
            self.model = HeteroEdgeGatedGNN(
                grid_node_dim=query_node_dim,
                station_node_dim=source_context_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                n_hops=n_hops,
                out_dim=out_dim,
                dropout=dropout,
            )

        self.huber = nn.HuberLoss(delta=huber_delta, reduction="none")

        if task == "scalar":
            self.val_mae = MeanAbsoluteError()
            self.val_baseline_ae_sum = 0.0
            self.val_baseline_count = 0
        else:
            self.val_mae_per_target = nn.ModuleDict(
                {name: MeanAbsoluteError() for name in (target_names or [])}
            )

    def forward(self, data):
        if self.hparams.da_version == "v1":
            pred, da_gate_mean = self.model(
                data,
                disable_payload=self.hparams.disable_payload,
                source_edge_dropout=self.hparams.source_edge_dropout
                if self.training
                else 0.0,
            )
            return pred, da_gate_mean
        else:
            # v0 mapping
            mapped = data.clone()
            mapped["grid"] = data["query"]
            mapped["station"] = data["source"]
            mapped["station", "influences", "grid"] = data[
                "source", "influences", "query"
            ]
            mapped["grid", "neighbors", "grid"] = data["query", "neighbors", "query"]
            return self.model(mapped), torch.tensor(0.0)

    # ---- Scalar loss ----

    def _scalar_loss(self, pred, batch):
        pred = pred.squeeze(-1)
        mask = batch["query"].loss_mask
        if not mask.any():
            return pred.sum() * 0.0, 0
        return self.huber(pred[mask], batch["query"].y[mask]).mean(), mask.sum().item()

    def _scalar_val_step(self, pred, batch):
        pred = pred.squeeze(-1)
        mask = batch["query"].loss_mask
        if not mask.any():
            return torch.tensor(0.0)
        pred_m = pred[mask]
        target_m = batch["query"].y[mask]
        loss = self.huber(pred_m, target_m).mean()
        self.log("val_loss", loss, prog_bar=True, batch_size=mask.sum().item())
        self.val_mae.update(pred_m, target_m)
        self.val_baseline_ae_sum += target_m.abs().sum().item()
        self.val_baseline_count += target_m.numel()
        return loss

    # ---- Multitask loss ----

    def _multitask_loss(self, pred, batch):
        target = batch["query"].y
        valid = batch["query"].valid_mask
        loss_mask = batch["query"].loss_mask
        scales = self.hparams.target_scales
        head_losses = []
        for i, _name in enumerate(self.hparams.target_names):
            mask = valid[:, i] & loss_mask
            if not mask.any():
                continue
            pred_i, tgt_i = pred[mask, i], target[mask, i]
            if scales:
                err = (pred_i - tgt_i) / scales[i]
                head_losses.append(self.huber(err, torch.zeros_like(err)).mean())
            else:
                head_losses.append(self.huber(pred_i, tgt_i).mean())
        if not head_losses:
            return pred.sum() * 0.0, 0
        return torch.stack(head_losses).mean(), int(loss_mask.sum().item())

    def _multitask_val_step(self, pred, batch):
        loss, n = self._multitask_loss(pred, batch)
        self.log("val_loss", loss, prog_bar=True, batch_size=max(n, 1))
        valid = batch["query"].valid_mask
        loss_mask = batch["query"].loss_mask
        target = batch["query"].y
        for i, name in enumerate(self.hparams.target_names):
            mask = valid[:, i] & loss_mask
            if mask.any():
                self.val_mae_per_target[name].update(pred[mask, i], target[mask, i])
        return loss

    # ---- Training / validation steps ----

    def training_step(self, batch, batch_idx):
        pred, da_gate_mean = self(batch)
        if self.hparams.task == "scalar":
            loss, n = self._scalar_loss(pred, batch)
        else:
            loss, n = self._multitask_loss(pred, batch)
        self.log("train_loss", loss, prog_bar=True, batch_size=max(n, 1))
        self.log("train/da_gate_mean", da_gate_mean, batch_size=max(n, 1))
        return loss

    def validation_step(self, batch, batch_idx):
        pred, da_gate_mean = self(batch)
        self.log("val/da_gate_mean", da_gate_mean, batch_size=1)
        if self.hparams.task == "scalar":
            return self._scalar_val_step(pred, batch)
        else:
            return self._multitask_val_step(pred, batch)

    def on_validation_epoch_end(self):
        if self.hparams.task == "scalar":
            self.log("val/target_mae", self.val_mae, prog_bar=True)
            if self.val_baseline_count > 0:
                bl_mae = self.val_baseline_ae_sum / self.val_baseline_count
                self.log("val/baseline_mae", bl_mae)
            self.val_baseline_ae_sum = 0.0
            self.val_baseline_count = 0
        else:
            for name, metric in self.val_mae_per_target.items():
                self.log(f"val/mae_{name}", metric, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
