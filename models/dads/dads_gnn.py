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


class DadsMetGNN(pl.LightningModule):
    """DADS GNN for target-day inference with optional target exog branch.

    Ingests
    - graph.x: node features assembled by DadsDataset as [embedding, exog] per node
      (exog = day-of-interest exogenous features: rsun, NOAA CDR daily bands, and
      available GEO features such as lat/lon/doy/terrain).
      Target rows have zeros(emb_dim) + exog(day); neighbor rows have embedding + zeros(exog_dim).
    - graph.edge_attr: per-edge attribute differences (target - neighbor), optionally
      augmented with sin/cos(bearing) and distance (km) from prep.graph.
    - graph.node_lstm: neighbor node contexts for the day-of-interest; target row zeros.

    Behavior
    - When use_target_exog_branch is True, applies a small MLP to the exog slice and
      adds it only to target rows, separating target-day exogenous encoding from
      neighbor embedding features before message passing.
    """
    def __init__(self, pretrained_lstm_path, output_dim, n_nodes=5, hidden_dim=64, edge_attr_dim=20,
                 dropout=0.1, learning_rate=1e-3, log_csv=None, two_hop=False,
                 use_target_exog_branch=False, emb_dim=None, exog_dim=0, **lstm_meta):
        super().__init__()
        self.save_hyperparameters()
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

        self.column_indices = lstm_meta['column_indices']
        self.tensor_width = lstm_meta['tensor_width']
        self.scaler = lstm_meta['scaler']

        self.node_proj = nn.LazyLinear(hidden_dim)
        self.node_lstm_proj = nn.LazyLinear(hidden_dim)
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

    def forward(self, graph):

        x = self.node_proj(graph.x)
        # If provided, concatenate per-node LSTM outputs with node features, then fuse
        if hasattr(graph, 'node_lstm'):
            node_lstm = graph.node_lstm
            if node_lstm.dim() == 1:
                node_lstm = node_lstm.unsqueeze(-1)
            node_lstm = node_lstm.float()
            assert node_lstm.shape[0] == graph.x.shape[0], "node_lstm rows must match nodes"
            node_lstm_feat = self.node_lstm_proj(node_lstm)
            x = torch.cat([x, node_lstm_feat], dim=-1)
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
            x[target_idx] = x[target_idx] + exog_feat[target_idx]
        if hasattr(graph, 'edge_attr'):
            assert graph.edge_attr.dim() == 2 and graph.edge_attr.shape[1] == self.edge_attr_dim, "edge_attr dim mismatch"
        x = self.gnn_layer(x, graph.edge_index, graph.edge_attr)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        if self.two_hop:
            res = x
            x = self.gnn_layer2(x, graph.edge_index, graph.edge_attr)
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

    def training_step(self, batch, batch_idx):
        graph, y_obs = batch
        y_hat = self(graph)

        y_obs = y_obs[:, -1:]

        loss = F.mse_loss(y_hat, y_obs)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        graph, y_obs = batch
        y_hat = self(graph)

        y_hat = y_hat.squeeze()
        y_obs = y_obs[:, -1]
        loss_obs = self.criterion(y_hat, y_obs)
        self.log('val_loss', loss_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=graph.batch_size)

        if len(y_hat.shape) < 1:
            y_obs = y_obs.unsqueeze(0)
            y_hat = y_hat.unsqueeze(0)

        y_obs = self.inverse_transform(y_obs, idx=self.column_indices[0])
        y_hat = self.inverse_transform(y_hat, idx=self.column_indices[0])

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
