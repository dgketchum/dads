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
    def __init__(self, input_dim, nhead, dim_feedforward, num_encoder_layers, latent_dim, num_decoder_layers,
                 learning_rate=0.01, dropout=0.1, log_csv=None, scaler=None, **kwargs):

        super().__init__()

        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=input_dim,
                                             nhead=nhead,
                                             dim_feedforward=dim_feedforward,
                                             dropout=dropout),
            num_layers=num_encoder_layers)

        self.decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(d_model=input_dim,
                                             nhead=nhead,
                                             dim_feedforward=dim_feedforward,
                                             dropout=dropout),
            num_layers=num_decoder_layers)

        self.linear_encode = torch.nn.Linear(input_dim, latent_dim)
        self.linear_decode = torch.nn.Linear(latent_dim, input_dim)

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

    def init_bias(self):
        for module in [self.input_projection, self.output_projection, self.embedding_layer]:
            if module.bias is not None:
                nn.init.uniform_(module.bias, -0.1, 0.1)

    def forward(self, x, mask):
        encoder_out = self.encoder(x, src_key_padding_mask=mask.T)
        latent = self.linear_encode(encoder_out)
        decoder_input = self.linear_decode(latent)
        decoder_out = self.decoder(decoder_input, encoder_out, tgt_key_padding_mask=mask.T)
        return decoder_out, latent

    def training_step(self, batch, batch_idx):
        x, mask = stack_batch(batch)

        x_mean = x[~mask].mean(dim=0)
        x[mask] = x_mean

        y_hat, _ = self(x, mask)
        y_hat = y_hat.flatten().unsqueeze(1)
        y = x.flatten().unsqueeze(1)
        loss = self.criterion(y_hat, y)

        self.log('train_loss', loss)
        return loss

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

            y = torch.cat(self.y_last)
            y = y.cpu()
            y[:, :, :self.data_width] = self.scaler.inverse_transform(y[:, :, :self.data_width])
            y = y.view(-1, self.tensor_width)

            y_hat = torch.cat(self.y_hat_last)
            y_hat = y_hat.cpu()
            y_hat[:, :, :self.data_width] = self.scaler.inverse_transform(y_hat[:, :, :self.data_width])
            y_hat = y_hat.view(-1, self.tensor_width)

            mask = torch.cat(self.mask)
            mask = mask.reshape((mask.shape[0] * mask.shape[1]))
            mask = mask.unsqueeze(1).repeat_interleave(y.size(1), dim=1)
            mask = mask.cpu()

            for i, col in enumerate(self.data_columns):

                obs = y[mask].reshape(-1, self.tensor_width)[:, i]
                pred = y_hat[mask].reshape(-1, self.tensor_width)[:, i]

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

    def validation_step(self, batch, batch_idx):
        y, mask = stack_batch(batch)

        y_hat, _ = self(y, mask)

        self.y_last.append(y)
        self.y_hat_last.append(y_hat)
        self.mask.append(mask)

        yh_flat = y_hat.flatten().unsqueeze(1)
        y_flat = y.flatten().unsqueeze(1)
        loss_obs = self.criterion(yh_flat, y_flat)

        self.log('val_loss', loss_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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

    # def on_before_optimizer_step(self, optimizer):
    #     torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)


def stack_batch(batch):
    x = torch.stack([item[0] for item in batch])
    mask = torch.stack([item[1] for item in batch])
    return x, mask


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
