import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import r2_score, root_mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {device_name}")
else:
    print("CUDA is not available. PyTorch will use the CPU.")


class ResidualPredictor(pl.LightningModule):
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


def test_relationship_bands(csv, plot_dir, learning_rate=0.001):
    df = pd.read_csv(csv, index_col='Unnamed: 0', parse_dates=True)
    df['doy'] = df.index.dayofyear
    variables = ['rsds', 'vpd', 'min_temp', 'max_temp', 'mean_temp', 'wind', 'eto']
    bands = ['B10', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'doy']
    results = {}

    fids = np.unique(df['FID'])
    np.random.seed(236544)
    train = np.random.choice(fids, int(len(fids) * 0.8))
    df['train'] = [1 if f in train else 0 for f in df['FID']]

    for var in variables[-1:]:

        select = bands + [f"{v}_gm" for v in variables if v != var]
        sub = df[[f"{var}_obs", f"{var}_gm"] + select].copy()
        sub.index = df['train']
        sub['residual'] = sub[f"{var}_obs"] - sub[f"{var}_gm"]
        sub.dropna(how='any', inplace=True, axis=0)
        sub[select] = sub[select].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        x_t = torch.tensor(sub.loc[sub.index == 1, select].values, dtype=torch.float32)
        y_t = torch.tensor(sub.loc[sub.index == 1, 'residual'].values, dtype=torch.float32).unsqueeze(1)
        x_v = torch.tensor(sub.loc[sub.index == 0, select].values, dtype=torch.float32)
        y_v = torch.tensor(sub.loc[sub.index == 0, 'residual'].values, dtype=torch.float32).unsqueeze(1)

        train_dataset, val_dataset = TensorDataset(x_t, y_t), TensorDataset(x_v, y_v)

        val = val_dataset.tensors[1].numpy()
        mean_ = val.mean()
        mean_ = np.ones_like(val) * mean_
        rmse = root_mean_squared_error(val, mean_)
        print('\n===============================================================')
        print('Training Predictor on {}, predicting the mean gives RMSE: {:.2f}'.format(var.upper(), rmse))
        print('===============================================================\n')

        train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=11)
        val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=11)

        model = ResidualPredictor(len(select), learning_rate)
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
        trainer = pl.Trainer(max_epochs=50, callbacks=[early_stopping])

        trainer.fit(model, train_dataloader, val_dataloader)

        val_results = trainer.validate(model, dataloaders=val_dataloader, verbose=False)
        val_r_squared = 1 - val_results[0]['val_loss'] / torch.var(y_v)
        results[f"{var}_residual_vs_bands"] = val_r_squared


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    fields = os.path.join(d, 'met', 'stations', 'gwx_stations.csv')
    sta = os.path.join(d, 'met', 'obs', 'gwx')
    gm = os.path.join(d, 'met', 'gridded', 'gridmet')
    rs = os.path.join(d, 'rs', 'gwx_stations')
    joined = os.path.join(d, 'tables', 'gridmet', 'western_lst_metvars_all.csv')
    plots = os.path.join(d, 'plots', 'gridmet')

    test_relationship_bands(joined, plots)
# ========================= EOF ====================================================================
