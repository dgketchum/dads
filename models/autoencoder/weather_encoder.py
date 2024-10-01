import csv

import torch
import numpy as np
import pytorch_lightning as pl
from distributed.dashboard.components.shared import profile_interval
from sklearn.metrics import r2_score, root_mean_squared_error
from torch import nn as nn, optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WeatherAutoencoder(pl.LightningModule):
    def __init__(self, input_size=7, sequence_len=72, embedding_size=16, d_model=16, nhead=4, num_layers=2,
                 learning_rate=0.01, log_csv=None, scaler=None, **kwargs):
        super().__init__()

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead), num_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(embedding_size, nhead), num_layers)

        self.input_projection = nn.Linear(input_size, d_model)
        self.output_projection = nn.Linear(d_model, input_size)
        self.embedding_layer = nn.Linear(d_model, sequence_len * embedding_size)

        self.sequence_len = sequence_len
        self.embedding_size = embedding_size

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
        x = self.input_projection(x)
        encoded = self.encoder(x, src_key_padding_mask=mask.T)
        embedding = self.embedding_layer(encoded[:, 0]).view(-1, self.sequence_len, self.embedding_size)
        decoded = self.decoder(embedding, encoded, tgt_key_padding_mask=mask.T)
        return self.output_projection(decoded), embedding

    def training_step(self, batch, batch_idx):
        x, mask = stack_batch(batch)

        x_mean = x[~mask].mean(dim=0)
        x[mask] = x_mean

        y_hat, _ = self(x, mask)
        y_hat = y_hat.flatten().unsqueeze(1)
        y = x.flatten().unsqueeze(1)
        loss = self.criterion(y_hat, y)

        if not np.isfinite(loss.item()):
            a = 1

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
        y, mask = batch

        y_arr = y.cpu().numpy()
        y_nanct = np.count_nonzero(np.isnan(y_arr))

        if y_nanct > 0:
            nan_handle = True
            y_mod = y.clone()
            y_mod = torch.nan_to_num(y_mod, nan=0.0)
            y_hat, _ = self(y_mod, mask)

        else:
            nan_handle = False
            y_hat, _ = self(y, mask)

        yh_arr = y_hat.cpu().numpy()
        yh_nanct = np.count_nonzero(np.isnan(yh_arr))

        if yh_nanct > 0:
            y_hat_mean = y_hat[~torch.isnan(y_hat)].mean(dim=0)
            y_hat[mask] = y_hat_mean

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

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)


def stack_batch(batch):
    x = torch.stack([item[0] for item in batch])
    mask = torch.stack([item[1] for item in batch])
    return x, mask


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
