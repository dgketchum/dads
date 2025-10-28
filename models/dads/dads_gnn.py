import csv
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import r2_score, root_mean_squared_error
from torch import nn
from torch import optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import LayerNorm, TransformerConv

from models.temporal.tcn import TemporalConvEncoder


class DadsMetGNN(pl.LightningModule):
    """DADS GNN with on-the-fly TCN contexts for neighbors.

    Ingests
    - graph.x: [embedding, exog] per node. Target: zeros(emb) + exog(day). Neighbors: embedding + zeros(exog).
    - graph.edge_attr: per-edge attributes (target - neighbor), optional bearing/distance.
    - neighbor sequences (provided in batch): shape [B, n_nodes, T, C] with boolean mask of availability.
    """
    def __init__(self, output_dim, n_nodes=5, hidden_dim=64, edge_attr_dim=20,
                 dropout=0.1, learning_rate=1e-3, log_csv=None, two_hop=False,
                 use_target_exog_branch=False, emb_dim=None, exog_dim=0,
                 scaler=None, column_indices=None,
                 # TCN params
                 tcn_in_channels=8, tcn_out_dim=256, tcn_channels=128, tcn_dilations=(1, 2, 4, 8),
                 tcn_kernel=3, tcn_dropout=0.1):
        super().__init__()
        self.save_hyperparameters(ignore=['scaler'])
        self.learning_rate = learning_rate
        self.n_nodes = n_nodes
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.edge_attr_dim = edge_attr_dim
        self.criterion = nn.MSELoss()
        self.log_csv = log_csv
        self.two_hop = two_hop
        self.use_target_exog_branch = bool(use_target_exog_branch)
        self.emb_dim = int(emb_dim) if emb_dim is not None else None
        self.exog_dim = int(exog_dim) if exog_dim is not None else 0

        self.column_indices = column_indices
        self.scaler = scaler

        # Temporal encoder producing neighbor contexts
        self.tcn = TemporalConvEncoder(
            in_channels=int(tcn_in_channels),
            channels=int(tcn_channels),
            kernel_size=int(tcn_kernel),
            dilations=tuple(tcn_dilations),
            dropout=float(tcn_dropout),
            out_dim=int(tcn_out_dim)
        )

        self.node_proj = nn.LazyLinear(hidden_dim)
        # Pre-GNN LayerNorm to tame attention logits in TransformerConv (helps fp16/bf16 stability)
        self.pre_norm = LayerNorm(hidden_dim)
        self.node_ctx_proj = nn.LazyLinear(hidden_dim)
        self.fuse = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gnn_layer = TransformerConv(hidden_dim, hidden_dim, heads=1, edge_dim=edge_attr_dim, dropout=dropout)
        self.norm = LayerNorm(hidden_dim)
        if self.two_hop:
            self.gnn_layer2 = TransformerConv(hidden_dim, hidden_dim, heads=1, edge_dim=edge_attr_dim, dropout=dropout)
            self.norm2 = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        if self.use_target_exog_branch and self.exog_dim > 0:
            self.target_exog_mlp = nn.Sequential(
                nn.Linear(self.exog_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            self.target_exog_mlp = None

    def _assert_finite(self, tensor, name):
        if tensor is None:
            return
        if not torch.isfinite(tensor).all():
            n_total = tensor.numel()
            n_nan = torch.isnan(tensor).sum().item()
            n_inf = torch.isinf(tensor).sum().item()
            raise ValueError(f"Non-finite values detected in {name}: nan={n_nan}, inf={n_inf}, total={n_total}")

    def _forward_gnn(self, graph):

        # Runtime guards
        self._assert_finite(graph.x, 'graph.x')
        x = self.node_proj(graph.x)
        # If provided, concatenate per-node context outputs with node features, then fuse
        if hasattr(graph, 'node_ctx'):
            node_ctx = graph.node_ctx
            if node_ctx.dim() == 1:
                node_ctx = node_ctx.unsqueeze(-1)
            node_ctx = node_ctx.float()
            assert node_ctx.shape[0] == graph.x.shape[0], "node_ctx rows must match nodes"
            self._assert_finite(node_ctx, 'graph.node_ctx')
            node_ctx_feat = self.node_ctx_proj(node_ctx)
            x = torch.cat([x, node_ctx_feat], dim=-1)
            x = self.fuse(x)
        # Optional target-only exog branch: add exog encoding to target rows only
        if self.target_exog_mlp is not None and self.emb_dim is not None and self.exog_dim > 0:
            exog_slice = graph.x[:, self.emb_dim:self.emb_dim + self.exog_dim]
            exog_feat = self.target_exog_mlp(exog_slice)
            # locate target row indices for each graph in batch
            if hasattr(graph, 'ptr'):
                target_idx = graph.ptr[:-1]
            else:
                b = graph.batch
                _, counts = torch.unique_consecutive(b, return_counts=True)
                starts = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)[:-1]])
                target_idx = starts
            # Scale down residual add for stability under mixed precision
            x[target_idx] = x[target_idx] + 0.1 * exog_feat[target_idx]
        if hasattr(graph, 'edge_attr'):
            assert graph.edge_attr.dim() == 2 and graph.edge_attr.shape[1] == self.edge_attr_dim, "edge_attr dim mismatch"
        # Run GNN layers in fp32 to avoid fp16 overflow in attention computations
        x = self.pre_norm(x)
        with torch.amp.autocast('cuda', enabled=False):
            x = self.gnn_layer(x.float(), graph.edge_index, graph.edge_attr.float())
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        if self.two_hop:
            res = x
            with torch.amp.autocast('cuda', enabled=False):
                x = self.gnn_layer2(x.float(), graph.edge_index, graph.edge_attr.float())
            x = self.norm2(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + res

        if hasattr(graph, 'ptr'):
            target_idx = graph.ptr[:-1]
        else:
            b = graph.batch
            _, counts = torch.unique_consecutive(b, return_counts=True)
            starts = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)[:-1]])
            target_idx = starts

        x_t = x[target_idx]
        out = self.fc(x_t)
        out = out.squeeze(-1)
        return out

    def _encode_contexts(self, neighbor_seq, neighbor_mask):
        # neighbor_seq: [B, n_nodes, T, C]
        B, K, T, C = neighbor_seq.shape
        x = neighbor_seq.view(B * K, T, C).permute(0, 2, 1).contiguous()  # [BK, C, T]
        # Guard inputs for finiteness
        self._assert_finite(x, 'neighbor_seq')
        with torch.amp.autocast('cuda', enabled=torch.is_autocast_enabled()):
            ctx = self.tcn(x)  # [BK, D]
        self._assert_finite(ctx, 'tcn_ctx')
        ctx = ctx.view(B, K, -1)
        # Impute missing per-sample with mean of available
        mask = neighbor_mask  # [B, K]
        D = ctx.shape[-1]
        ctx_filled = []
        for b in range(B):
            m = mask[b]
            if m.any():
                mean_vec = ctx[b, m].mean(dim=0)
            else:
                mean_vec = torch.zeros(D, device=ctx.device, dtype=ctx.dtype)
            row = torch.where(m.view(-1, 1), ctx[b], mean_vec.view(1, -1))
            ctx_filled.append(row)
        ctx_filled = torch.stack(ctx_filled, dim=0)  # [B, K, D]
        return ctx_filled

    def forward(self, graph, neighbor_seq=None, neighbor_mask=None):
        if neighbor_seq is not None and neighbor_mask is not None:
            # Build node_ctx from TCN contexts
            device = graph.x.device
            neighbor_seq = neighbor_seq.to(device)
            neighbor_mask = neighbor_mask.to(device)
            ctx = self._encode_contexts(neighbor_seq, neighbor_mask)  # [B, K, D]
            B, K, D = ctx.shape
            # expand with target zeros and flatten to match Batch node order
            zeros = torch.zeros(B, 1, D, device=ctx.device, dtype=ctx.dtype)
            node_ctx = torch.cat([zeros, ctx], dim=1)  # [B, 1+K, D]
            # scatter into graph order using ptr
            if hasattr(graph, 'ptr'):
                parts = []
                for b in range(B):
                    parts.append(node_ctx[b])
                graph.node_ctx = torch.cat(parts, dim=0)
            else:
                graph.node_ctx = node_ctx.view(-1, D)
        return self._forward_gnn(graph)

    def training_step(self, batch, batch_idx):
        graph, y_obs, neighbor_seq, neighbor_mask = batch
        y_hat = self(graph, neighbor_seq, neighbor_mask)
        # Match shapes for loss: y_hat [B], take last target as [B]
        y_obs = y_obs[:, -1]
        loss = F.mse_loss(y_hat, y_obs)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        graph, y_obs, neighbor_seq, neighbor_mask = batch
        y_hat = self(graph, neighbor_seq, neighbor_mask)

        y_hat = y_hat.squeeze()
        y_obs = y_obs[:, -1]
        loss_obs = self.criterion(y_hat, y_obs)
        self.log('val_loss', loss_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=graph.batch_size)

        if len(y_hat.shape) < 1:
            y_obs = y_obs.unsqueeze(0)
            y_hat = y_hat.unsqueeze(0)

        # Compute metrics in fp32 on CPU; clamp extremes to avoid overflow during early training
        y_obs = y_obs.float()
        y_hat = y_hat.float()
        y_obs = self.inverse_transform(y_obs, idx=self.column_indices[0])
        y_hat = self.inverse_transform(y_hat, idx=self.column_indices[0])
        y_obs = torch.nan_to_num(y_obs, nan=0.0, posinf=0.0, neginf=0.0)
        y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)
        y_hat = y_hat.clamp_(-1e4, 1e4)  # clamp for metrics only; model uses unclamped values for training

        y_obs = y_obs.detach().cpu().numpy()
        y_hat = y_hat.detach().cpu().numpy()

        rmse_dads = root_mean_squared_error(y_obs, y_hat)

        r2_dads = r2_score(y_obs, y_hat)

        self.log('r2_dads', r2_dads, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=graph.batch_size)

        self.log('rmse_dads', rmse_dads, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=graph.batch_size)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=graph.batch_size)
        # lr_ratio = current_lr / self.learning_rate
        # self.log('lr_ratio', lr_ratio, on_step=False, on_epoch=True, prog_bar=True)

        return loss_obs

    def on_validation_epoch_end(self):

        if self.log_csv:
            try:
                train_loss = self.trainer.callback_metrics['train_loss'].item()
            except KeyError:
                train_loss = torch.nan
            val_loss = self.trainer.callback_metrics['val_loss'].item()
            r2_dads = self.trainer.callback_metrics['r2_dads'].item()
            rmse_dads = self.trainer.callback_metrics['rmse_dads'].item()
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            lr_ratio = current_lr / self.learning_rate

            log_data = [self.current_epoch,
                        round(train_loss, 4),
                        round(val_loss, 4),
                        round(r2_dads, 4),
                        round(rmse_dads, 4),
                        round(current_lr, 4),
                        round(lr_ratio, 4)]

            with open(self.log_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(log_data)

            if self.current_epoch == 0:
                with open(self.log_csv, 'r+', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    f.seek(0)
                    writer = csv.writer(f)
                    writer.writerow(['epoch', 'train_loss', 'val_loss', 'r2_dads',
                                     'rmse_dads', 'lr', 'lr_ratio'])
                    writer.writerow(header)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                      factor=0.5, patience=2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

    def inverse_transform(self, a, idx):
        # Variable-specific MinMaxScaler stored as [1, width]
        a = a * (self.scaler.scale[0, idx] + 5e-8) + self.scaler.bias[0, idx]
        return a


class LRChangeCallback(Callback):
    def __init__(self, checkpoint_path):
        self.previous_lr = None
        self.checkpoint_path = checkpoint_path

    def on_train_epoch_start(self, trainer, pl_module):
        optimizer = trainer.optimizers[0]
        current_lr = optimizer.param_groups[0]['lr']

        if self.previous_lr is None:
            self.previous_lr = current_lr
        elif self.previous_lr != current_lr:
            DadsMetGNN.load_from_checkpoint(checkpoint_path=self.checkpoint_path)
            self.previous_lr = current_lr


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
