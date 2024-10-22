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
from torch_geometric.nn import LayerNorm

from models.simple_lstm.lstm import LSTMPredictor


class AttentionGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_attr_dim):
        super().__init__()
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels)
        self.attn_mlp = nn.Sequential(nn.Linear(1, out_channels),
                                      nn.ReLU(),
                                      nn.Linear(out_channels, 1))
        self.edge_attr_transform = nn.Linear(edge_attr_dim, out_channels)

    def forward(self, data):
        x = self.linear(data.x.squeeze())
        row, col = data.edge_index
        row -= 1
        edge_attr = data.edge_attr
        edge_attr = self.edge_attr_transform(edge_attr)
        node_vec = torch.index_select(x, 0, col)

        combined_feat = torch.cat([node_vec, edge_attr], dim=-1)
        attn_weights = self.attn_mlp(combined_feat.unsqueeze(-1))
        attn_weights = torch.softmax(attn_weights, dim=1).squeeze()

        node_vec = torch.index_select(x, 0, col).repeat_interleave(2, dim=1)
        agg = node_vec * attn_weights

        return agg


class DadsMetGNN(pl.LightningModule):
    def __init__(self, pretrained_lstm_path, output_dim, n_nodes=5, hidden_dim=64, edge_attr_dim=20,
                 dropout=0.1, learning_rate=1e-3, log_csv=None, **lstm_meta):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.n_nodes = n_nodes
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.criterion = nn.MSELoss()
        self.log_csv = log_csv

        data_frequency = lstm_meta['data_frequency']
        lf_bands = data_frequency.count('lf') - 2
        self.column_indices = lstm_meta['column_indices']
        self.tensor_width = lstm_meta['tensor_width']
        self.scaler = lstm_meta['scaler']

        model_path = os.path.join(pretrained_lstm_path, 'best_model.ckpt')
        self.lstm = LSTMPredictor.load_from_checkpoint(model_path,
                                                       num_bands=lf_bands,
                                                       learning_rate=learning_rate,
                                                       dropout_rate=0.3,
                                                       log_csv=None)
        for param in self.lstm.parameters():
            param.requires_grad = False

        self.gnn_layer = AttentionGCNConv(hidden_dim, hidden_dim, edge_attr_dim)
        self.norm = LayerNorm(hidden_dim * 2)

        self.lstm_transform = nn.Linear(output_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, data, sequence):
        lstm_output = self.lstm(sequence)

        try:
            lstm_output = lstm_output.unsqueeze(1)
        except:
            lstm_output = lstm_output.unsqueeze(0)
            lstm_output = lstm_output.unsqueeze(0)

        try:
            lstm_output = lstm_output[:, :, -1:]
        except:
            lstm_output = lstm_output[:, -1:]

        lstm_feat = self.lstm_transform(lstm_output)

        x = self.gnn_layer(data)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.view(-1, self.n_nodes, self.hidden_dim * 2)

        lstm_feat = lstm_feat.repeat_interleave(self.n_nodes, dim=1)
        lstm_feat = lstm_feat.repeat_interleave(2, dim=2)
        combined_features = torch.cat([x, lstm_feat], dim=1)
        out = self.fc(combined_features)
        out = out.mean(dim=1)
        return out, lstm_output

    def training_step(self, batch, batch_idx):
        data, y, _, sequence = batch
        out, lstm = self(data, sequence)
        y = y[:, -1:]

        loss = F.mse_loss(out, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, y_obs, y_gm, sequence = batch
        y_hat, lstm = self(data, sequence)
        y_hat = y_hat.squeeze()
        y_lstm = lstm.squeeze()
        y_obs = y_obs[:, -1]
        y_gm = y_gm[:, -1]

        loss_obs = self.criterion(y_hat, y_obs)
        self.log('val_loss', loss_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=data.batch_size)

        if len(y_hat.shape) < 1:
            y_obs = y_obs.unsqueeze(0)
            y_hat = y_hat.unsqueeze(0)
            y_lstm = y_lstm.unsqueeze(0)
            y_gm = y_gm.unsqueeze(0)

        y_obs = self.inverse_transform(y_obs, idx=self.column_indices[0])
        y_hat = self.inverse_transform(y_hat, idx=self.column_indices[1])
        y_lstm = self.inverse_transform(y_lstm, idx=self.column_indices[1])
        y_gm = self.inverse_transform(y_gm, idx=self.column_indices[2])

        y_obs = y_obs.detach().cpu().numpy()
        y_hat = y_hat.detach().cpu().numpy()
        y_lstm = y_lstm.detach().cpu().numpy()
        y_gm = y_gm.detach().cpu().numpy()

        rmse_dads = root_mean_squared_error(y_obs, y_hat)
        rmse_lstm = root_mean_squared_error(y_obs, y_lstm)
        rmse_gm = root_mean_squared_error(y_obs, y_gm)

        r2_dads = r2_score(y_obs, y_hat)
        r2_lstm = r2_score(y_obs, y_lstm)
        r2_gm = r2_score(y_obs, y_gm)

        self.log('r2_dads', r2_dads, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=data.batch_size)
        self.log('r2_lstm', r2_lstm, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=data.batch_size)
        self.log('r2_gm', r2_gm, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=data.batch_size)

        self.log('rmse_dads', rmse_dads, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=data.batch_size)
        self.log('rmse_lstm', rmse_lstm, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=data.batch_size)
        self.log('rmse_gm', rmse_gm, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=data.batch_size)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=data.batch_size)
        # lr_ratio = current_lr / self.learning_rate
        # self.log('lr_ratio', lr_ratio, on_step=False, on_epoch=True, prog_bar=True)

        return loss_obs

    def on_validation_epoch_end(self):

        if self.log_csv:
            train_loss = self.trainer.callback_metrics['train_loss'].item()
            val_loss = self.trainer.callback_metrics['val_loss'].item()
            r2_dads = self.trainer.callback_metrics['r2_dads'].item()
            r2_lstm = self.trainer.callback_metrics['r2_lstm'].item()
            r2_gm = self.trainer.callback_metrics['r2_gm'].item()
            rmse_dads = self.trainer.callback_metrics['rmse_dads'].item()
            rmse_lstm = self.trainer.callback_metrics['rmse_lstm'].item()
            rmse_gm = self.trainer.callback_metrics['rmse_gm'].item()
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            lr_ratio = current_lr / self.learning_rate

            log_data = [self.current_epoch,
                        round(train_loss, 4),
                        round(val_loss, 4),
                        round(r2_dads, 4),
                        round(r2_lstm, 4),
                        round(r2_gm, 4),
                        round(rmse_dads, 4),
                        round(rmse_lstm, 4),
                        round(rmse_gm, 4),
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
                    writer.writerow(['epoch', 'train_loss', 'val_loss', 'r2_dads', 'r2_lstm', 'r2_gm',
                                     'rmse_dads', 'rmse_lstm', 'rmse_gm',
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
