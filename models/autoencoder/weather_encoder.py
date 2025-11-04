import csv

import pytorch_lightning as pl
import torch
from sklearn.metrics import r2_score, root_mean_squared_error
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x):
        w = self.score(x).squeeze(-1)
        a = torch.softmax(w, dim=1)
        z = torch.einsum('btd,bt->bd', x, a)
        return z


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size, num_heads, num_layers, latent_size, dropout=0.1, max_len=1024):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_latent = nn.Linear(hidden_size, latent_size)
        self.pos_enc = PositionalEncoding(hidden_size, max_len=max_len)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = x.permute(1, 0, 2)
        enc_output = self.transformer_encoder(x)
        enc_output = enc_output.permute(1, 0, 2)
        latent = self.fc_latent(enc_output)
        return latent


class TransformerDecoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_dim, num_heads, num_layers, seq_length, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_size, output_dim)
        self.seq_length = seq_length
        self.pos_queries = nn.Parameter(torch.randn(seq_length, 1, hidden_size))

    def forward(self, z):
        mem = self.latent_to_hidden(z).unsqueeze(0)
        tgt = self.pos_queries.expand(self.seq_length, mem.size(1), -1)
        dec_output = self.transformer_decoder(tgt, mem)
        x_hat = self.fc_out(dec_output)
        x_hat = x_hat.permute(1, 0, 2)
        return x_hat


class WeatherAutoencoder(pl.LightningModule):
    """Transformer autoencoder for station-year sequences with mask-aware inputs.

    Intent
    - Encoder consumes selected inputs (target + GEO + RS missingness flags) over a
      fixed 365-day window and produces a per-sequence latent embedding.
    - Decoder reconstructs the specified target columns; reconstruction is computed
      only on valid elements via an external boolean mask from the dataset.
    - Missing numeric inputs are set to zero inside the model (nan_to_num); the
      RS missingness flags allow the network to distinguish truly missing RS from
      valid near-zero values.
    - Optional triplet loss encourages station-level embedding structure using
      positive/negative samples (when available).
    - Optional exogenous-only ablation: when `zero_target_in_encoder=True`, the target
      input channel is zeroed before encoding so embeddings reflect exogenous drivers
      rather than the target itself.
    """
    def __init__(self, input_dim, output_dim, learning_rate, latent_size, hidden_size,
                 dropout=0.1, margin=1.0, sequence_length=365, log_csv=None, scaler=None,
                 scaler_bias=None, scaler_scale=None, **kwargs):

        super(WeatherAutoencoder, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.encoder = TransformerEncoder(input_dim, hidden_size, 1, 2, latent_size, dropout, max_len=sequence_length)
        self.decoder = TransformerDecoder(latent_size, hidden_size, output_dim, 1, 2, sequence_length, dropout)
        self.attn_pool = AttentionPooling(latent_size)

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

        self.y_last = []
        self.y_hat_last = []
        self.mask = []

        # Register scaler parameters as buffers for on-device scaling.
        # Expect scaler_bias/scale sized to the selected input features (C_in).
        if scaler_bias is not None and scaler_scale is not None:
            sb = torch.as_tensor(scaler_bias, dtype=torch.float32)
            ss = torch.as_tensor(scaler_scale, dtype=torch.float32)
            if sb.ndim == 1:
                sb = sb.view(1, 1, -1)
                ss = ss.view(1, 1, -1)
            elif sb.ndim == 2:
                sb = sb.view(1, *sb.shape)
                ss = ss.view(1, *ss.shape)
            self.register_buffer('x_bias', sb)
            self.register_buffer('x_scale', ss)
            # Derive target scaler slices assuming target indices map into inputs.
            tgt_idx = int(getattr(self, 'target_input_idx', 0))
            t_bias = sb[..., tgt_idx:tgt_idx + self.output_dim]
            t_scale = ss[..., tgt_idx:tgt_idx + self.output_dim]
            self.register_buffer('y_bias', t_bias)
            self.register_buffer('y_scale', t_scale)

    def _scale_inputs(self, x):
        if hasattr(self, 'x_bias') and hasattr(self, 'x_scale'):
            return (x - self.x_bias) / self.x_scale + 5e-8
        return x

    def _scale_targets(self, y):
        if hasattr(self, 'y_bias') and hasattr(self, 'y_scale'):
            return (y - self.y_bias) / self.y_scale + 5e-8
        return y

        self.margin = margin

    def forward(self, x):
        # Optional ablation: zero out target input channel to force exogenous-only encoding
        # Scale inputs on-device, then replace NaNs.
        x = self._scale_inputs(x)
        x = torch.nan_to_num(x)

        if hasattr(self, 'zero_target_in_encoder') and self.zero_target_in_encoder:
            idx = int(getattr(self, 'target_input_idx', 0))
            x = x.clone()
            x[:, :, idx] = 0.0
        latent_seq = self.encoder(x)
        z = self.attn_pool(latent_seq)
        x_hat = self.decoder(z)
        return x_hat, z

    def triplet_loss(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

    def init_bias(self):
        for module in [self.input_projection, self.output_projection, self.embedding_layer]:  # likely dead code
            if module.bias is not None:
                nn.init.uniform_(module.bias, -0.1, 0.1)

    def training_step(self, batch, batch_idx):
        # Mark step boundary for cudagraph safety (no-op if unsupported)
        try:
            torch.compiler.cudagraph_mark_step_begin()
        except Exception:
            pass
        x, y, mask, x_pos, x_neg = stack_batch(batch)
        bsz = x.size(0)

        # Forward scales inputs internally. Scale targets to the same space.
        y = self._scale_targets(y)
        y = torch.nan_to_num(y)

        y_hat, z = self(x)

        # Flatten for masked loss
        y_hat = y_hat.reshape(-1, self.output_dim)
        y = y.reshape(-1, self.output_dim)
        mask = mask[:, :, :self.output_dim].reshape(-1, self.output_dim)

        y = y[mask]
        y_hat = y_hat[mask]
        nan_mask = ~torch.isnan(y_hat)

        loss = self.criterion(y[nan_mask], y_hat[nan_mask])

        if x_pos is not None and x_neg is not None:
            valid_pos = (~torch.isnan(x_pos)).float().mean()
            valid_neg = (~torch.isnan(x_neg)).float().mean()
            if valid_pos > 0.3 and valid_neg > 0.3:
                _, z_pos = self(x_pos)
                _, z_neg = self(x_neg)
                triplet_loss = self.triplet_loss(z, z_pos, z_neg)
                loss += triplet_loss * 2.0
                self.log('triplet_loss', triplet_loss, batch_size=bsz)

        # Log CPU scalars to avoid compiled graph interactions in Lightning internals
        rec_val = float(loss.detach().cpu())
        self.log('reconstruction_loss', rec_val, batch_size=bsz)
        self.log('train_loss', rec_val, on_step=True, batch_size=bsz)

        return loss

    def validation_step(self, batch, batch_idx):
        # Mark step boundary for cudagraph safety (no-op if unsupported)
        try:
            torch.compiler.cudagraph_mark_step_begin()
        except Exception:
            pass
        x, y, mask, x_pos, x_neg = stack_batch(batch)
        bsz = x.size(0)

        # Scale targets to match network output space.
        y = self._scale_targets(y)

        y_hat, z = self(x)

        # Flatten and mask
        y_hat = y_hat.reshape(-1, self.output_dim)
        y = y.reshape(-1, self.output_dim)
        mask = mask[:, :, :self.output_dim].reshape(-1, self.output_dim)

        # Detach and move to CPU to avoid GPU memory growth and compiled graph capture
        self.y_hat_last.append(y_hat.detach().cpu())
        self.y_last.append(y.detach().cpu())
        self.mask.append(mask.detach().cpu())

        y = y.flatten()
        y_hat = y_hat.flatten()
        loss_mask = mask.flatten()

        y = y[loss_mask]
        y_hat = y_hat[loss_mask]

        nan_mask = ~torch.isnan(y)

        if y[nan_mask].numel() == 0:
            loss_obs = torch.zeros((), device=y_hat.device)
        else:
            loss_obs = self.criterion(y[nan_mask], y_hat[nan_mask])

        val_val = float(loss_obs.detach().cpu())
        self.log('val_loss', val_val, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bsz)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', float(current_lr), on_step=False, on_epoch=True, prog_bar=True, batch_size=bsz)
        lr_ratio = current_lr / self.learning_rate
        self.log('lr_ratio', float(lr_ratio), on_step=False, on_epoch=True, prog_bar=True, batch_size=bsz)

        return loss_obs

    def configure_optimizers(self):
        # Prefer fused AdamW when available for GPU speedups.
        try:
            optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4, fused=True)
        except TypeError:
            optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
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
    if isinstance(batch, (tuple, list)) and len(batch) == 5 and torch.is_tensor(batch[0]):
        x, y, mask, pos, neg = batch
        return x, y, mask, pos, neg

    x, y, mask, pos, neg = [], [], [], [], []

    for item in batch:
        xi, yi, mi, pi, ni = item
        if pi is None:
            pi = torch.full_like(xi, torch.nan)
        if ni is None:
            ni = torch.full_like(xi, torch.nan)
        x.append(xi)
        y.append(yi)
        mask.append(mi)
        pos.append(pi)
        neg.append(ni)

    x = torch.stack(x)
    y = torch.stack(y)
    mask = torch.stack(mask)
    pos = torch.stack(pos)
    neg = torch.stack(neg)

    return x, y, mask, pos, neg


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
