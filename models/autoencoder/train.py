import os

import random

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


class WeatherDataset(Dataset):
    def __init__(self, data_path, scaler, chunk_size=72 * 24):
        self.df = pd.read_csv(data_path)
        self.data = torch.tensor(self.df.values, dtype=torch.float32)
        self.data[:, :4], _ = robust_scale(self.data[:, :4].numpy(), scaler)
        self.chunk_size = chunk_size

    def __len__(self):
        return (len(self.data) - self.chunk_size) // 24 + 1

    def __getitem__(self, idx):
        start_idx = idx * 24
        end_idx = start_idx + self.chunk_size
        return self.data[start_idx:end_idx, :]


def sample_for_normalization(data_paths, sample_size=10000):
    """Samples data from multiple CSV files to gather normalization statistics."""
    all_data = []
    for path in data_paths:
        df = pd.read_csv(path)
        sample = df.sample(n=min(sample_size, len(df)), replace=False)
        all_data.append(sample)

    combined_data = pd.concat(all_data)
    return combined_data.iloc[:, :4].values  # Only consider the first 4 meteorological columns


def robust_scale(data, scaler=None):
    """Scales data using RobustScaler, optionally fitting a new scaler."""
    if scaler is None:
        scaler = RobustScaler()
        scaler.fit(data)
    return scaler.transform(data), scaler


def train_autoencoder(model, dataloader, optimizer, criterion, num_epochs, chunk_size=72 * 24):
    for epoch in range(num_epochs):
        for batch_data in dataloader:
            seq_len = batch_data.size(1)
            start_idx = random.randint(0, seq_len - chunk_size)
            chunk = batch_data[:, start_idx:start_idx + chunk_size, :]

            optimizer.zero_grad()
            output, _ = model(chunk)
            loss = criterion(output, chunk)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
