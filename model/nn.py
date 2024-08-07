import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import r2_score, root_mean_squared_error
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence


if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {device_name}")
else:
    print("CUDA is not available. PyTorch will use the CPU.")


def concatenate_and_shuffle_pth_data(dataset):
    all_data = []
    all_labels = []

    for i in range(len(dataset)):
        data, label = dataset[i]
        all_data.append(data)
        all_labels.append(label)

    all_data = torch.cat(all_data)
    all_labels = torch.cat(all_labels)

    nan_mask = torch.isnan(all_data).any(dim=1)
    all_data = all_data[~nan_mask]
    all_labels = all_labels[~nan_mask]

    indices = torch.randperm(all_data.size(0))
    all_data = all_data[indices]
    all_labels = all_labels[indices]

    return torch.utils.data.TensorDataset(all_data, all_labels)



class PTHDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = torch.load(file_path)
        label = data[:, 1]
        data = data[:, 2:]

        if self.transform:
            data = self.transform(data)
        return data, label


class Predictor(pl.LightningModule):
    def __init__(self, num_bands=10, learning_rate=0.001, hidden_layers=[64, 32, 16]):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_bands, hidden_layers[0]))
        self.layers.append(nn.ReLU())

        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_layers[-1], 1))

        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        y_hat_np = y_hat.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        r2 = r2_score(y_np, y_hat_np)
        rmse = root_mean_squared_error(y_np, y_hat_np)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_r2", r2, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_rmse", rmse, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


def test_relationship_bands(csv, pth, template, target='rsds_obs', learning_rate=0.001):
    """"""
    results = {}
    cols = pd.read_csv(template).columns
    df = pd.read_csv(csv, parse_dates=True, index_col='fid')
    np.random.seed(1234)
    fids = [f.split('.')[0] for f in os.listdir(pth)]
    df = df.loc[fids].sample(frac=1)
    df['rand'] = np.random.rand(len(df))

    train = df.loc[df['rand'] < 0.8].index
    train_files = [os.path.join(pth, '{}.pth'.format(f)) for f in train]
    train_dataset = PTHDataset(train_files)
    train_dataset = concatenate_and_shuffle_pth_data(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)

    val = df.loc[df['rand'] >= 0.8].index
    val_files = [os.path.join(pth, '{}.pth'.format(f)) for f in val]
    val_dataset = PTHDataset(val_files)
    val_dataset = concatenate_and_shuffle_pth_data(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)

    model = Predictor(8, learning_rate)
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    trainer = pl.Trainer(max_epochs=5, callbacks=[early_stopping])

    trainer.fit(model, train_dataloader, val_dataloader)

    val_results = trainer.validate(model, dataloaders=val_dataloader, verbose=False)
    val_r_squared = 1 - val_results[0]['val_loss'] / torch.var(y_v)
    results[f"{target}"] = val_r_squared


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    fields = os.path.join(d, 'met', 'stations', 'dads_stations_WMT_mgrs.csv')
    pth_ = os.path.join(d, 'training', 'scaled_pth')
    csv_template = os.path.join(d, 'training', 'compiled_csv', 'BFAM.csv')

    test_relationship_bands(fields, pth_, csv_template)
# ========================= EOF ====================================================================
