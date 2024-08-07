import os
import json

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import r2_score, root_mean_squared_error
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {device_name}")
else:
    print("CUDA is not available. PyTorch will use the CPU.")

torch.set_float32_matmul_precision('medium')
torch.cuda.get_device_name(torch.cuda.current_device())


class PTHLSTMDataset(Dataset):
    def __init__(self, file_list, window_size=5, transform=None):
        self.file_list = file_list
        self.window_size = window_size
        self.transform = transform
        self.chunks = self._find_chunks()

    def _find_chunks(self):
        chunks = []
        for file_path in self.file_list:
            data = torch.load(file_path)
            start_idx = 0
            while start_idx + self.window_size <= len(data):
                if not torch.isnan(data[start_idx:start_idx + self.window_size]).any():
                    chunks.append((file_path, start_idx))
                start_idx += 1
        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        file_path, start_idx = self.chunks[idx]
        data = torch.load(file_path)
        x = data[start_idx:start_idx + self.window_size, 2:]
        y_obs = data[start_idx + self.window_size - 1, 0]
        y_gm = data[start_idx + self.window_size - 1, 1]
        return x, y_obs, y_gm


class LSTMPredictor(pl.LightningModule):
    def __init__(self, num_bands=10, hidden_size=64, num_layers=2, learning_rate=0.001):
        super().__init__()
        self.lstm = nn.LSTM(num_bands, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :]).squeeze()

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_obs, y_gm = batch

        y_hat_obs = self(x).squeeze()

        loss_obs = self.criterion(y_hat_obs, y_obs)
        self.log("val_loss_obs", loss_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        y_hat_obs_np = y_hat_obs.detach().cpu().numpy()
        y_obs_np = y_obs.detach().cpu().numpy()
        y_gm_np = y_gm.detach().cpu().numpy()

        r2_obs = r2_score(y_obs_np, y_hat_obs_np)
        rmse_obs = root_mean_squared_error(y_obs_np, y_hat_obs_np)
        r2_gm = r2_score(y_obs_np, y_gm_np)
        rmse_gm = root_mean_squared_error(y_obs_np, y_gm_np)

        self.log("val_r2_obs", r2_obs, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_r2_gm", r2_gm, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse_obs", rmse_obs, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse_gm", rmse_gm, on_step=False, on_epoch=True, prog_bar=True)

        return loss_obs

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss_obs'
            }
        }


def train_model(csv, pth, metadata, learning_rate=0.001, window_size=5):
    """"""

    with open(metadata, 'r') as f:
        meta = json.load(f)

    features = [c for c, s in zip(meta['column_order'], meta['scaling_status']) if s == 'scaled']
    feature_len = len(features)
    df = pd.read_csv(csv, index_col='fid')
    np.random.seed(1234)
    fids = [f.split('.')[0] for f in os.listdir(pth)]
    df = df.loc[fids].sample(frac=1)
    df['rand'] = np.random.rand(len(df))

    train = df.loc[df['rand'] < 0.8].index
    train_files = [os.path.join(pth, '{}.pth'.format(f)) for f in train]
    train_dataset = PTHLSTMDataset(train_files, window_size=window_size)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=64,
                                  shuffle=True,
                                  num_workers=6,
                                  collate_fn=lambda batch: [x for x in batch if x is not None])

    val = df.loc[df['rand'] >= 0.8].index
    val_files = [os.path.join(pth, '{}.pth'.format(f)) for f in val]
    val_dataset = PTHLSTMDataset(val_files, window_size=window_size)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=6)

    model = LSTMPredictor(num_bands=feature_len, learning_rate=learning_rate)

    early_stopping = EarlyStopping(monitor="val_loss_obs", patience=20, mode="min")
    trainer = pl.Trainer(max_epochs=100, callbacks=[early_stopping])

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    target_var = 'vpd'

    fields = os.path.join(d, 'met', 'stations', 'dads_stations_WMT_mgrs.csv')
    pth_ = os.path.join(d, 'training', target_var, 'scaled_pth')
    metadata_ = os.path.join(d, 'training', target_var, 'training_metadata.json')

    train_model(fields, pth_, metadata_)
# ========================= EOF ====================================================================
