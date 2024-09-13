import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl


class MeteorologicalInterpolator(pl.LightningModule):
    def __init__(self, pretrained_lstm_path, output_dim, edge_emb_dim=6, hidden_dim=64,
                 num_gnn_layers=5, dropout=0.2, learning_rate=1e-3, freeze_lstm=True):

        super().__init__()
        self.save_hyperparameters()

        self.lstm = torch.load(pretrained_lstm_path)
        if freeze_lstm:
            for param in self.lstm.parameters():
                param.requires_grad = False

        self.gnn_layers = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)])
        self.edge_emb = nn.Linear(edge_emb_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):

        with torch.no_grad():
            x, _ = self.lstm(data.x)

        edge_attr = self.edge_emb(data.edge_attr)

        edge_index = data.edge_index
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x, edge_index, edge_attr if i == 0 else None)
            x = F.relu(x)
            x = self.dropout(x)

        out = self.output_layer(x)
        return out


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
