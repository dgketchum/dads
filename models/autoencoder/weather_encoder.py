import csv

import torch
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import r2_score, root_mean_squared_error
from torch import nn as nn, optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WeatherAutoencoder(pl.LightningModule):
    def __init__(self, input_size=7, sequence_len=72, embedding_size=16, d_model=16, nhead=4, num_layers=2,
                 learning_rate=0.01, log_csv=None, scaler=None, feature_strings=None):
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
        self.feature_strings = feature_strings

        # hack to get model output to on_validation_epoch_end hook
        self.y_last = []
        self.y_hat_last = []

    def forward(self, x, mask=None):
        x = self.input_projection(x)
        encoded = self.encoder(x, src_key_padding_mask=mask)
        embedding = self.embedding_layer(encoded[:, 0]).view(-1, self.sequence_len, self.embedding_size)
        decoded = self.decoder(embedding, encoded, tgt_key_padding_mask=mask)
        return self.output_projection(decoded), embedding

    def training_step(self, batch, batch_idx):
        x = stack_batch(batch)
        y_hat, _ = self(x)
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
                for p in self.feature_strings:
                    if p in ['year_sin', 'year_cos']:
                        continue
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
            y = self.scaler.inverse_transform(y)

            y_hat = torch.cat(self.y_hat_last)
            y_hat = y_hat.cpu()
            y_hat = self.scaler.inverse_transform(y_hat)

            for i, col in enumerate(self.feature_strings):

                if col in ['year_sin', 'year_cos']:
                    continue

                r2_obs = r2_score(y[:, i], y_hat[:, i])
                rmse_obs = root_mean_squared_error(y[:, i], y_hat[:, i])
                log_data.extend([round(r2_obs, 4), round(rmse_obs, 4)])

            with open(self.log_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(log_data)

        self.y_hat_last = []
        self.y_last = []

    def validation_step(self, batch, batch_idx):
        y = batch
        y_hat, _ = self(y)

        self.y_last.append(y)
        self.y_hat_last.append(y_hat)

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


def stack_batch(batch):
    y = torch.stack(batch, dim=0)
    return y


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
