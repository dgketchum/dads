import csv

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import r2_score, root_mean_squared_error
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


class FCEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size, dropout):
        super(FCEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h1 = F.relu(self.fc1(self.dropout(x)))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar


class FCDecoder(nn.Module):
    def __init__(self, output_dim, hidden_size, latent_size, dropout, sigmoid=True):
        super(FCDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = sigmoid

    def forward(self, z):
        h1 = F.relu(self.fc1(z))
        x_hat = self.fc2(h1)
        if self.sigmoid:
            x_hat = torch.sigmoid(x_hat)
        return x_hat


class WeatherAutoencoder(pl.LightningModule):
    def __init__(self, input_dim=14, output_dim=5, learning_rate=0.0001, latent_size=2, hidden_size=400,
                 dropout=0.1, margin=1.0, log_csv=None, scaler=None, **kwargs):

        super(WeatherAutoencoder, self).__init__()
        self.latent_size = latent_size
        self.encoder = FCEncoder(input_dim, hidden_size, dropout)
        self.decoder = FCDecoder(output_dim, latent_size, hidden_size, dropout, sigmoid=False)
        self.input_dim = input_dim
        self.output_dim = output_dim

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
        self.hidden_size = hidden_size

        # hack to get model output to on_validation_epoch_end hook
        self.y_last = []
        self.y_hat_last = []
        self.mask = []

        self.margin = margin

    def triplet_loss(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

    @staticmethod
    def reparameterization(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x.float().view(-1, self.input_dim))
        z = self.reparameterization(mu, logvar)
        embed = z.view(-1, self.hidden_size, self.chunk_size)
        embed = torch.max(embed, dim=2).values
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, embed

    def init_bias(self):
        for module in [self.input_projection, self.output_projection, self.embedding_layer]:
            if module.bias is not None:
                nn.init.uniform_(module.bias, -0.1, 0.1)

    def training_step(self, batch, batch_idx):
        x, y, mask, x_pos, x_neg = stack_batch(batch)

        x = torch.nan_to_num(x)
        y = y.view(-1, self.output_dim)

        x_mean = x[mask].mean(dim=0)
        x[mask] = x_mean

        y_hat, mu, logvar, z = self(x)

        mask = mask[:, :, :self.output_dim].view(-1, self.output_dim)
        loss = self.criterion(y_hat[mask], y[mask])

        if x_pos is not None and x_neg is not None:
            x_pos = torch.nan_to_num(x_pos)
            x_neg = torch.nan_to_num(x_neg)
            _, _, _, z_pos = self(x_pos)
            _, _, _, z_neg = self(x_neg)
            triplet_loss = self.triplet_loss(z, z_pos, z_neg)
            loss += triplet_loss * 2.0
            self.log('triplet_loss', triplet_loss)

        self.log('reconstruction_loss', loss)
        self.log('train_loss', loss, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask, x_pos, x_neg = stack_batch(batch)

        x = torch.nan_to_num(x)
        y = y.view(-1, self.output_dim)
        mask = mask[:, :, :self.output_dim].view(-1, self.output_dim)

        y_hat, mu, logvar, z = self(x)

        self.y_hat_last.append(y_hat)
        self.y_last.append(y)
        self.mask.append(mask)

        y = y.flatten()
        y_hat = y_hat.flatten()
        loss_mask = mask.flatten()

        y = y[loss_mask]
        y_hat = y_hat[loss_mask]

        # TODO: this should not be necessary
        nan_mask = ~torch.isnan(y)

        loss_obs = self.criterion(y[nan_mask], y_hat[nan_mask])

        if torch.isnan(loss_obs):
            a = 1

        self.log('val_loss', loss_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, on_step=False, on_epoch=True, prog_bar=True)
        lr_ratio = current_lr / self.learning_rate
        self.log('lr_ratio', lr_ratio, on_step=False, on_epoch=True, prog_bar=True)

        return loss_obs

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
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
    x, y, mask, pos, neg = [], [], [], [], []

    for item in batch:
        if any([i is None for i in item]):
            continue
        else:
            x.append(item[0])
            y.append(item[1])
            mask.append(item[2])
            pos.append(item[3])
            neg.append(item[4])

    x = torch.stack(x)
    y = torch.stack(y)
    mask = torch.stack(mask)
    pos = torch.stack(pos)
    neg = torch.stack(neg)

    return x, y, mask, pos, neg


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
