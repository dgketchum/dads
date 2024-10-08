import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl

from models.simple_lstm.lstm import LSTMPredictor


class DadsMetGNN(pl.LightningModule):
    def __init__(self, pretrained_lstm_path, output_dim, edge_emb_dim=6, hidden_dim=64,
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

        self.gnn_layers = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)])
        self.edge_emb = nn.Linear(edge_emb_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_attr, edge_index):

        with torch.no_grad():
            x, _ = self.lstm(x)

        edge_attr = self.edge_emb(edge_attr)

        edge_index = edge_index
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x, edge_index, edge_attr if i == 0 else None)
            x = F.relu(x)
            x = self.dropout(x)

        out = self.output_layer(x)
        return out

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.mse_loss(out, batch.y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
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
