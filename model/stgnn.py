import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class STGNN(torch.nn.Module):
    def __init__(self, node_features, out_channels):
        super(STGNN, self).__init__()
        self.lstm = torch.nn.LSTM(node_features, 32, batch_first=True)
        self.conv1 = GCNConv(32, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index):
        h = None
        out, (h, c) = self.lstm(x)
        h = F.relu(self.conv1(out[:, -1, :], edge_index))
        h = self.conv2(h, edge_index)
        return h


def train_model(data, node_features=5, out_channels=1, num_epochs=100, learning_rate=0.01):
    x = data.x.unsqueeze(1).repeat(1, 5, 1)
    edge_index = data.edge_index
    y = data.y.unsqueeze(1)
    model = STGNN(node_features, out_channels)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x, edge_index)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss.item()}')
    model.eval()
    with torch.no_grad():
        val_output = model(x, edge_index)
        val_loss = criterion(val_output, y)
        print(f'Validation Loss: {val_loss.item()}')


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/dads'
    clim = '/media/research/IrrigationGIS/climate'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'
        clim = '/home/dgketchum/data/IrrigationGIS/climate'

    pass
# ========================= EOF ====================================================================
