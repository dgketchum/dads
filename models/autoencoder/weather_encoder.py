import csv

import torch
import pytorch_lightning as pl
from sklearn.metrics import r2_score, root_mean_squared_error
from torch import nn as nn, optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WeatherAutoencoder(pl.LightningModule):
    def __init__(self, input_size=7, sequence_len=72, embedding_size=16, d_model=16, nhead=4, num_layers=2,
                 learning_rate=0.01, log_csv=None):
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

    def validation_step(self, batch, batch_idx):
        x = batch
        y_hat, _ = self(x)
        y_hat = y_hat.flatten().unsqueeze(1)
        y = x.flatten().unsqueeze(1)
        loss_obs = self.criterion(y_hat, y)
        self.log('val_loss', loss_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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
