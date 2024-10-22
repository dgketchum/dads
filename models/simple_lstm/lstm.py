import csv

import torch
import pytorch_lightning as pl

import torch.nn as nn
from torch import optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score, root_mean_squared_error


class LSTMPredictor(pl.LightningModule):
    def __init__(self,
                 num_bands=8,
                 hidden_size=64,
                 num_layers=2,
                 learning_rate=0.01,
                 expansion_factor=2,
                 dropout_rate=0.1,
                 log_csv=None,
                 scaler=None):

        super().__init__()

        self.input_expansion = nn.Sequential(
            nn.Linear(num_bands, num_bands * expansion_factor),
            nn.ReLU(),
            nn.Linear(num_bands * expansion_factor, num_bands * expansion_factor * 2),
            nn.ReLU(),
            nn.Linear(num_bands * expansion_factor * 2, num_bands * expansion_factor * 4),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(num_bands * expansion_factor * 4, hidden_size, num_layers, batch_first=True,
                            bidirectional=True)

        self.fc1 = nn.Linear(2 * hidden_size, hidden_size * 4)

        self.output_layers = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1)
        )

        self.criterion = nn.L1Loss()
        self.learning_rate = learning_rate
        self.log_csv = log_csv
        self.scaler = scaler

    def forward(self, x):
        x = x.squeeze()
        x = self.input_expansion(x)
        out, _ = self.lstm(x)

        if len(x.shape) == 2:
            out = out.unsqueeze(0)

        out = out[:, -1, :]

        out = self.fc1(out)
        out = self.output_layers(out).squeeze()
        return out

    def training_step(self, batch, batch_idx):
        y_obs, gm, lf = stack_batch(batch)
        y_hat = self(lf)
        y_obs = y_obs[:, -1]

        loss = self.criterion(y_hat, y_obs)
        self.log('train_loss', loss)
        return loss

    def on_train_epoch_end(self):

        if self.log_csv:
            train_loss = self.trainer.callback_metrics['train_loss'].item()
            val_loss = self.trainer.callback_metrics['val_loss'].item()
            r2_lstm = self.trainer.callback_metrics['r2_lstm'].item()
            r2_gm = self.trainer.callback_metrics['r2_gm'].item()
            r2_nl = self.trainer.callback_metrics['r2_nl'].item()
            rmse_lstm = self.trainer.callback_metrics['rmse_lstm'].item()
            rmse_gm = self.trainer.callback_metrics['rmse_gm'].item()
            rmse_nl = self.trainer.callback_metrics['rmse_nl'].item()
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            lr_ratio = current_lr / self.learning_rate

            log_data = [self.current_epoch,
                        round(train_loss, 4),
                        round(val_loss, 4),
                        round(r2_lstm, 4),
                        round(r2_gm, 4),
                        round(r2_nl, 4),
                        round(rmse_lstm, 4),
                        round(rmse_gm, 4),
                        round(rmse_nl, 4),
                        round(current_lr, 4),
                        round(lr_ratio, 4)]

            with open(self.log_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(log_data)

            if self.current_epoch == 0:
                with open(self.log_csv, 'r+', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    f.seek(0)
                    writer = csv.writer(f)
                    writer.writerow(['epoch', 'train_loss', 'val_loss', 'r2_lstm', 'r2_gm', 'r2_nl', 'rmse_lstm',
                                     'rmse_gm', 'rmse_nl', 'lr', 'lr_ratio'])
                    writer.writerow(header)

    def validation_step(self, batch, batch_idx):
        y_obs, y_gm, lf = batch
        y_hat = self(lf)
        y_obs = y_obs[:, -1]
        y_gm = y_gm[:, -1]
        y_nl = lf[:, -1, 0]

        loss_obs = self.criterion(y_hat, y_obs)
        self.log('val_loss', loss_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        y_obs = self.inverse_transform(y_obs, idx=0)
        y_hat = self.inverse_transform(y_hat, idx=0)
        y_gm = self.inverse_transform(y_gm, idx=1)
        y_nl = self.inverse_transform(y_nl, idx=2)

        y_hat = y_hat.detach().cpu().numpy()
        y_obs = y_obs.detach().cpu().numpy()
        y_gm = y_gm.detach().cpu().numpy()
        y_nl = y_nl.detach().cpu().numpy()

        rmse_obs = root_mean_squared_error(y_obs, y_hat)
        rmse_gm = root_mean_squared_error(y_obs, y_gm)
        rmse_nl = root_mean_squared_error(y_obs, y_nl)

        r2_obs = r2_score(y_obs, y_hat)
        r2_gm = r2_score(y_obs, y_gm)
        r2_nl = r2_score(y_obs, y_nl)

        self.log('r2_lstm', r2_obs, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('r2_gm', r2_gm, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('r2_nl', r2_nl, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        self.log('rmse_lstm', rmse_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('rmse_gm', rmse_gm, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('rmse_nl', rmse_nl, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, on_step=False, on_epoch=True, prog_bar=True)
        # lr_ratio = current_lr / self.learning_rate
        # self.log('lr_ratio', lr_ratio, on_step=False, on_epoch=True, prog_bar=True)

        return loss_obs

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

    def inverse_transform(self, a, idx):
        a = a * (self.scaler.scale[0, -1, idx] + 5e-8) + self.scaler.bias[0, -1, idx]
        return a


def stack_batch(batch):
    y = torch.stack([item[0] for item in batch])
    gm = torch.stack([item[1] for item in batch])
    lf = torch.stack([item[2] for item in batch])
    return y, gm, lf


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
