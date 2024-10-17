import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import LayerNorm
import pytorch_lightning as pl
from sklearn.metrics import r2_score, root_mean_squared_error

from models.simple_lstm.lstm import LSTMPredictor


class AttentionGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_attr_dim):
        super().__init__()
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

        agg = node_vec * edge_attr
        attn_weights = self.attn_mlp(agg.unsqueeze(-1))
        attn_weights = torch.softmax(attn_weights, dim=1).squeeze()
        agg = agg * attn_weights
        agg = torch.zeros_like(x).scatter_add_(0, row.unsqueeze(-1).expand_as(agg), agg)

        return agg


class DadsMetGNN(pl.LightningModule):
    def __init__(self, pretrained_lstm_path, output_dim, n_nodes=5, hidden_dim=64, edge_attr_dim=20,
                 dropout=0.1, learning_rate=1e-3, **lstm_meta):

        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.n_nodes = n_nodes
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.criterion = nn.L1Loss()

        data_frequency = lstm_meta['data_frequency']
        lf_bands = data_frequency.count('lf') - 2

        model_path = os.path.join(pretrained_lstm_path, 'best_model.ckpt')
        self.lstm = LSTMPredictor.load_from_checkpoint(model_path,
                                                       num_bands=lf_bands,
                                                       learning_rate=learning_rate,
                                                       dropout_rate=0.3,
                                                       log_csv=None)
        for param in self.lstm.parameters():
            param.requires_grad = False

        self.gnn_layer = AttentionGCNConv(hidden_dim, hidden_dim, edge_attr_dim)
        self.norm = LayerNorm(hidden_dim)

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

        lstm_feat = self.lstm_transform(lstm_output)

        x = self.gnn_layer(data)
        x = F.relu(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = x.view(-1, self.n_nodes, self.hidden_dim)

        lstm_feat = lstm_feat.unsqueeze(1)
        lstm_feat = lstm_feat.repeat_interleave(self.n_nodes, dim=1)
        combined_features = torch.cat([x, lstm_feat], dim=2)
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
        data, y_obs, gm, sequence = batch
        y_hat, lstm = self(data, sequence)
        y_hat = y_hat.squeeze()
        y_lstm = lstm.squeeze()
        y_obs = y_obs[:, -1]
        y_gm = gm[:, -1]

        loss_obs = self.criterion(y_hat, y_obs)
        self.log('val_loss', loss_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if len(y_hat.shape) < 1:
            y_obs = y_obs.unsqueeze(0)
            y_hat = y_hat.unsqueeze(0)
            y_lstm = y_lstm.unsqueeze(0)
            y_gm = y_gm.unsqueeze(0)

        y_obs = y_obs.detach().cpu().numpy()
        y_hat = y_hat.detach().cpu().numpy()
        y_lstm = y_lstm.detach().cpu().numpy()
        y_gm = y_gm.detach().cpu().numpy()

        rmse_dads = root_mean_squared_error(y_obs, y_hat)
        rmse_lstm = root_mean_squared_error(y_obs, y_lstm)
        rmse_gm = root_mean_squared_error(y_obs, y_gm)

        # self.log('val_r2', r2_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log('val_r2_gm', r2_gm, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.log('rmse_dads', rmse_dads, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('rmse_lstm', rmse_lstm, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('rmse_gm', rmse_gm, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, on_step=False, on_epoch=True, prog_bar=True)
        # lr_ratio = current_lr / self.learning_rate
        # self.log('lr_ratio', lr_ratio, on_step=False, on_epoch=True, prog_bar=True)

        return loss_obs

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


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
