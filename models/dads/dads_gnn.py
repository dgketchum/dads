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

from models.lstm.lstm import LSTMPredictor


class DadsMetGNN(pl.LightningModule):
    def __init__(self, pretrained_lstm_path, output_dim, n_nodes=5, hidden_dim=64, edge_attr_dim=20,
                 dropout=0.1, learning_rate=1e-3, log_csv=None, two_hop=False, **lstm_meta):
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

        data_frequency = lstm_meta['data_frequency']
        self.column_indices = lstm_meta['column_indices']
        self.tensor_width = lstm_meta['tensor_width']
        self.scaler = lstm_meta['scaler']

        # Use daily-only LSTM from models.lstm.lstm
        hf_start = self.column_indices[3] if len(self.column_indices) > 3 else self.tensor_width
        lf_bands = hf_start - 2
        model_path = os.path.join(pretrained_lstm_path, 'best_model.ckpt')
        self.lstm = LSTMPredictor.load_from_checkpoint(model_path,
                                                       num_bands=lf_bands,
                                                       learning_rate=learning_rate,
                                                       expansion_factor=2,
                                                       log_csv=None)
        for param in self.lstm.parameters():
            param.requires_grad = False

        self.node_proj = nn.LazyLinear(hidden_dim)
        self.node_lstm_proj = nn.LazyLinear(hidden_dim)
        self.gnn_layer = TransformerConv(hidden_dim, hidden_dim, heads=1, edge_dim=edge_attr_dim, dropout=dropout)
        self.norm = LayerNorm(hidden_dim)
        if self.two_hop:
            self.gnn_layer2 = TransformerConv(hidden_dim, hidden_dim, heads=1, edge_dim=edge_attr_dim, dropout=dropout)
            self.norm2 = LayerNorm(hidden_dim)

        self.lstm_transform = nn.Linear(output_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, graph, sequence):

        lstm_output = self.lstm(sequence)
        if lstm_output.ndim == 0:
            lstm_output = lstm_output.unsqueeze(0)
        if lstm_output.ndim == 1:
            lstm_output = lstm_output.unsqueeze(-1)
        lstm_feat = self.lstm_transform(lstm_output)

        x = self.node_proj(graph.x)
        # If provided, include per-node LSTM outputs (neighbors carry signal; target can be zeros)
        if hasattr(graph, 'node_lstm'):
            node_lstm = graph.node_lstm
            if node_lstm.dim() == 1:
                node_lstm = node_lstm.unsqueeze(-1)
            node_lstm = node_lstm.float()
            assert node_lstm.shape[0] == graph.x.shape[0], "node_lstm rows must match nodes"
            node_lstm_feat = self.node_lstm_proj(node_lstm)
            x = x + node_lstm_feat
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
        combined = torch.cat([x_t, lstm_feat], dim=-1)
        out = self.fc(combined)
        out = out.squeeze(-1)
        return out, lstm_output.squeeze(-1)

    def training_step(self, batch, batch_idx):
        graph, y_obs, sequence = batch
        y_hat, lstm = self(graph, sequence)

        y_obs = y_obs[:, -1:]

        loss = F.mse_loss(y_hat, y_obs)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        graph, y_obs, sequence = batch
        y_hat, lstm = self(graph, sequence)

        y_hat = y_hat.squeeze()
        y_lstm = lstm.squeeze()
        y_obs = y_obs[:, -1]
        loss_obs = self.criterion(y_hat, y_obs)
        self.log('val_loss', loss_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=graph.batch_size)

        if len(y_hat.shape) < 1:
            y_obs = y_obs.unsqueeze(0)
            y_hat = y_hat.unsqueeze(0)
            y_lstm = y_lstm.unsqueeze(0)

        y_obs = self.inverse_transform(y_obs, idx=self.column_indices[0])
        y_hat = self.inverse_transform(y_hat, idx=self.column_indices[0])
        y_lstm = self.inverse_transform(y_lstm, idx=self.column_indices[0])

        y_obs = y_obs.detach().cpu().numpy()
        y_hat = y_hat.detach().cpu().numpy()
        y_lstm = y_lstm.detach().cpu().numpy()

        rmse_dads = root_mean_squared_error(y_obs, y_hat)
        rmse_lstm = root_mean_squared_error(y_obs, y_lstm)

        r2_dads = r2_score(y_obs, y_hat)
        r2_lstm = r2_score(y_obs, y_lstm)

        self.log('r2_dads', r2_dads, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=graph.batch_size)
        self.log('r2_lstm', r2_lstm, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=graph.batch_size)

        self.log('rmse_dads', rmse_dads, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                  batch_size=graph.batch_size)
        self.log('rmse_lstm', rmse_lstm, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
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
            r2_lstm = self.trainer.callback_metrics['r2_lstm'].item()
            rmse_dads = self.trainer.callback_metrics['rmse_dads'].item()
            rmse_lstm = self.trainer.callback_metrics['rmse_lstm'].item()
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            lr_ratio = current_lr / self.learning_rate

            log_data = [self.current_epoch,
                        round(train_loss, 4),
                        round(val_loss, 4),
                        round(r2_dads, 4),
                        round(r2_lstm, 4),
                        round(rmse_dads, 4),
                        round(rmse_lstm, 4),
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
                    writer.writerow(['epoch', 'train_loss', 'val_loss', 'r2_dads', 'r2_lstm',
                                     'rmse_dads', 'rmse_lstm',
                                     'lr', 'lr_ratio'])
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
        a = a * (self.scaler.scale[0, -1, idx] + 5e-8) + self.scaler.bias[0, -1, idx]
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
