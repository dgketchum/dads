import csv

import torch
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import r2_score, root_mean_squared_error
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


class FCEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size, latent_size, dropout):
        super(FCEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h1 = F.relu(self.fc1(self.dropout(x)))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar


class FCDecoder(nn.Module):
    def __init__(self, input_dim, hidden_size, latent_size, sigmoid=True):
        super(FCDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_dim)
        self.sigmoid = sigmoid

    def forward(self, z):
        h1 = F.relu(self.fc1(z))
        x_hat = self.fc2(h1)
        if self.sigmoid:
            x_hat = torch.sigmoid(x_hat)
        return x_hat


class WeatherAutoencoder(pl.LightningModule):
    def __init__(self, input_dim, learning_rate, latent_size=2, hidden_size=400,
                 dropout=0.1, log_csv=None, scaler=None, **kwargs):

        super(WeatherAutoencoder, self).__init__()
        self.latent_size = latent_size
        self.encoder = FCEncoder(input_dim, hidden_size, latent_size, dropout)
        self.decoder = FCDecoder(input_dim, hidden_size, latent_size, sigmoid=False)
        self.input_dim = input_dim

        self.criterion = nn.L1Loss()
        self.learning_rate = learning_rate

        self.log_csv = log_csv
        self.scaler = scaler

        self.data_columns = []
        self.column_order = []

        for k, v in kwargs.items():
            self.__setattr__(k, v)

        self.data_width = len(self.data_columns)
        self.tensor_width = len(self.column_order)

        # hack to get model output to on_validation_epoch_end hook
        self.y_last = []
        self.y_hat_last = []
        self.mask = []

    @staticmethod
    def reparameterization(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x.float().view(-1, self.input_dim))
        z = self.reparameterization(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z

    def init_bias(self):
        for module in [self.input_projection, self.output_projection, self.embedding_layer]:
            if module.bias is not None:
                nn.init.uniform_(module.bias, -0.1, 0.1)

    def training_step(self, batch, batch_idx):
        x, mask = stack_batch(batch)
        x = torch.nan_to_num(x)

        x_mean = x[mask].mean(dim=0)
        x[mask] = x_mean

        y_hat, mu, logvar, z = self(x)
        y_hat = y_hat.flatten().unsqueeze(1)
        y = x.flatten().unsqueeze(1)
        loss = self.criterion(y_hat, y)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y, mask = stack_batch(batch)
        y = torch.nan_to_num(y)

        y_hat, mu, logvar, z = self(y)

        self.y_last.append(y)
        self.y_hat_last.append(y_hat.view(-1, self.tensor_width))
        self.mask.append(mask)

        y_flat = y[:, :, :self.data_width].flatten()
        yh_flat = y_hat[:, :self.data_width].flatten()
        loss_mask = mask.unsqueeze(2).repeat_interleave(y.size(2), dim=2)
        loss_mask = loss_mask[:, :, :self.data_width].flatten()

        loss_obs = self.criterion(y_flat[loss_mask], yh_flat[loss_mask])

        self.log('val_loss', loss_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, on_step=False, on_epoch=True, prog_bar=True)
        lr_ratio = current_lr / self.learning_rate
        self.log('lr_ratio', lr_ratio, on_step=False, on_epoch=True, prog_bar=True)

        return loss_obs

    def on_validation_epoch_end(self):

        if self.log_csv:

            if self.current_epoch == 0:
                train_loss = np.nan

                val_cols = []
                for p in self.data_columns:
                    val_cols.append(f'{p}_r2')
                    val_cols.append(f'{p}_rmse')

                headers = ['epoch', 'train_loss', 'val_loss', 'lr', 'lr_ratio'] + val_cols

                with open(self.log_csv, 'w', newline='\n') as f:
                    f.seek(0)
                    writer = csv.writer(f)
                    writer.writerow(headers)

            else:
                train_loss = self.trainer.callback_metrics['train_loss'].item()

            val_loss = self.trainer.callback_metrics['val_loss'].item()

            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            lr_ratio = current_lr / self.learning_rate

            log_data = [self.current_epoch,
                        round(train_loss, 4),
                        round(val_loss, 4),
                        round(current_lr, 4),
                        round(lr_ratio, 4)]

            y = torch.cat(self.y_last, dim=0)
            y = y.cpu()
            y[:, :, :self.data_width] = self.scaler.inverse_transform(y[:, :, :self.data_width])
            y = y.view(-1, self.tensor_width)

            y_hat = torch.cat(self.y_hat_last, dim=0)
            y_hat = y_hat.cpu()
            y_hat[:, :self.data_width] = self.scaler.inverse_transform(y_hat[:, :self.data_width])
            y_hat = y_hat.view(-1, self.tensor_width)

            mask = torch.cat(self.mask, dim=0)
            mask = mask.view(-1)
            mask = mask.cpu()

            for i, col in enumerate(self.data_columns):

                obs = y[mask][:, i]
                pred = y_hat[mask][:, i]

                try:
                    r2_obs = r2_score(obs, pred)
                    rmse_obs = root_mean_squared_error(obs, pred)
                    log_data.extend([round(r2_obs, 4), round(rmse_obs, 4)])
                except ValueError:

                    log_data.extend([round(np.nan, 4), round(np.nan, 4)])

            with open(self.log_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(log_data)

        self.y_hat_last = []
        self.y_last = []
        self.mask = []

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

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)


def stack_batch(batch):
    x = torch.stack([item[0] for item in batch])
    mask = torch.stack([item[1] for item in batch])
    return x, mask


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
