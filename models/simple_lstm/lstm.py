import csv

import torch
import pytorch_lightning as pl

import torch.nn as nn
from torch import optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from sklearn.metrics import r2_score, root_mean_squared_error


class LSTMPredictor(pl.LightningModule):
    def __init__(self,
                 num_bands=8,
                 hidden_size=64,
                 num_layers=2,
                 learning_rate=0.01,
                 expansion_factor=2,
                 dropout_rate=0.1,
                 log_csv=None):
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
        y, gm, lf = stack_batch(batch)
        y_hat = self(lf)
        y_obs = y[:, -1]

        loss = self.criterion(y_hat, y_obs)
        self.log('train_loss', loss)
        return loss

    def on_train_epoch_end(self):

        if self.log_csv:
            train_loss = self.trainer.callback_metrics['train_loss'].item()
            val_loss = self.trainer.callback_metrics['val_loss'].item()
            val_r2 = self.trainer.callback_metrics['val_r2'].item()
            gm_r2 = self.trainer.callback_metrics['val_r2_gm'].item()
            val_rmse = self.trainer.callback_metrics['val_rmse'].item()
            gm_rmse = self.trainer.callback_metrics['val_rmse_gm'].item()
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            lr_ratio = current_lr / self.learning_rate

            log_data = [self.current_epoch,
                        round(train_loss, 4),
                        round(val_loss, 4),
                        round(val_r2, 4),
                        round(gm_r2, 4),
                        round(val_rmse, 4),
                        round(gm_rmse, 4),
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
                    writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_r2', 'gm_r2', 'val_rmse', 'gm_rmse',
                                     'lr', 'lr_ratio'])
                    writer.writerow(header)

    def validation_step(self, batch, batch_idx):
        y, gm, lf = batch
        y_hat = self(lf)
        y_obs = y[:, -1]
        y_gm = gm[:, -1]

        loss_obs = self.criterion(y_hat, y_obs)
        self.log('val_loss', loss_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        y_hat_obs_np = y_hat.detach().cpu().numpy()
        y_obs_np = y_obs.detach().cpu().numpy()
        y_gm_np = y_gm.detach().cpu().numpy()

        r2_obs = r2_score(y_obs_np, y_hat_obs_np)
        rmse_obs = root_mean_squared_error(y_obs_np, y_hat_obs_np)
        r2_gm = r2_score(y_obs_np, y_gm_np)
        rmse_gm = root_mean_squared_error(y_obs_np, y_gm_np)

        self.log('val_r2', r2_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_r2_gm', r2_gm, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.log('val_rmse', rmse_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_rmse_gm', rmse_gm, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, on_step=False, on_epoch=True, prog_bar=True)
        lr_ratio = current_lr / self.learning_rate
        self.log('lr_ratio', lr_ratio, on_step=False, on_epoch=True, prog_bar=True)

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


def stack_batch(batch):
    y = torch.stack([item[0] for item in batch])
    gm = torch.stack([item[1] for item in batch])
    lf = torch.stack([item[2] for item in batch])
    return y, gm, lf


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
