import os

import torch
import torch.nn as nn
import pandas as pd


class WeatherAutoencoder(nn.Module):
    def __init__(self, input_size=8, embedding_size=4, hidden_size=64, num_layers=2):
        super(WeatherAutoencoder, self).__init__()

        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.embedding_layer = nn.Linear(hidden_size, 365 * embedding_size)
        self.decoder = nn.LSTM(365 * embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        embedding = self.embedding_layer(hidden[-1]).view(-1, 365, 4)
        decoded, _ = self.decoder(embedding.repeat(1, x.size(1), 1))
        output = self.output_layer(decoded)
        return output, embedding


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
