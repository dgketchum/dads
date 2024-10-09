import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GCNConv, LayerNorm
import pytorch_lightning as pl

from models.simple_lstm.lstm import LSTMPredictor


class AttentionGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.attn_mlp = nn.Sequential(
            nn.Linear(1, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        row, col = edge_index
        agg = torch.index_select(x, 0, col) * edge_attr.unsqueeze(-1)
        attn_weights = self.attn_mlp(edge_attr.unsqueeze(-1))
        attn_weights = torch.softmax(attn_weights, dim=1)
        agg = agg * attn_weights
        agg = torch.zeros_like(x).scatter_add_(0, row.unsqueeze(-1).expand_as(agg), agg)
        return agg


class DadsMetGNN(pl.LightningModule):
    def __init__(self, pretrained_lstm_path, output_dim, hidden_dim=64,
                 num_gnn_layers=5, dropout=0.2, learning_rate=1e-3, **lstm_meta):

        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

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

        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(AttentionGCNConv(hidden_dim, hidden_dim))
            self.norms.append(LayerNorm(hidden_dim))

        self.lstm_transform = nn.Linear(lstm_meta['hidden_dim'], hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, data, sequence):
        lstm_output = self.lstm(sequence)
        lstm_output = self.lstm_transform(lstm_output)
        x = data.x

        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x, data.edge_index, data.edge_attr)
            x = F.relu(x)
            x = self.norms[i](x)
            x = self.dropout(x)

        out = self.output_layer(x + lstm_output)
        return out

    def training_step(self, batch, batch_idx):
        data, y, _, sequence = batch
        out = self(data, sequence)
        loss = F.mse_loss(out, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                      factor=0.5, patience=5)
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
