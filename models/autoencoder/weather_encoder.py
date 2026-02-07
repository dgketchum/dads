import lightning.pytorch as pl
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

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
        z = torch.einsum("btd,bt->bd", x, a)
        return z


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size,
        num_heads,
        num_layers,
        latent_size,
        dropout=0.1,
        max_len=1024,
    ):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
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
    def __init__(
        self,
        latent_size,
        hidden_size,
        output_dim,
        num_heads,
        num_layers,
        seq_length,
        dropout=0.1,
    ):
        super(TransformerDecoder, self).__init__()
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(hidden_size, output_dim)
        self.seq_length = seq_length
        self.pos_queries = nn.Parameter(torch.randn(seq_length, 1, hidden_size))

    def forward(self, latent_seq_or_z):
        """Decode from either a latent sequence (preferred) or a pooled code.

        - If input is (B, T, D): treats it as the encoder latent sequence and
          uses it as the memory for cross-attention.
        - If input is (B, D): falls back to prior behavior by expanding the
          pooled code across the sequence length as memory.
        """
        if latent_seq_or_z.dim() == 3:
            # (B, T, D_latent)
            mem_seq = self.latent_to_hidden(latent_seq_or_z)  # (B, T, H)
            mem = mem_seq.permute(1, 0, 2)  # (T, B, H)
        else:
            # Backward-compatible path: pooled code
            mem = self.latent_to_hidden(latent_seq_or_z).unsqueeze(0)  # (1, B, H)
        tgt = self.pos_queries.expand(self.seq_length, mem.size(1), -1)
        dec_output = self.transformer_decoder(tgt, mem)
        x_hat = self.fc_out(dec_output)
        x_hat = x_hat.permute(1, 0, 2)  # (B, T, C_out)
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

    def __init__(
        self,
        input_dim,
        output_dim,
        learning_rate,
        latent_size,
        hidden_size,
        dropout=0.1,
        margin=1.0,
        sequence_length=365,
        log_csv=None,
        scaler=None,
        scaler_bias=None,
        scaler_scale=None,
        target_channel_dropout_p=0.3,
        span_mask_frac=0.5,
        contrastive_weight=0.05,
        contrastive_temperature=0.1,
        span_mask_warmup_epochs=4,
        target_dropout_warmup_epochs=2,
        **kwargs,
    ):

        super(WeatherAutoencoder, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.encoder = TransformerEncoder(
            input_dim, hidden_size, 1, 2, latent_size, dropout, max_len=sequence_length
        )
        self.decoder = TransformerDecoder(
            latent_size, hidden_size, output_dim, 1, 2, sequence_length, dropout
        )
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
        self.target_channel_dropout_p = float(target_channel_dropout_p)
        self.span_mask_frac = float(span_mask_frac)
        self.contrastive_weight = float(contrastive_weight)
        self.contrastive_temperature = float(contrastive_temperature)
        self.span_mask_warmup_epochs = int(span_mask_warmup_epochs)
        self.target_dropout_warmup_epochs = int(target_dropout_warmup_epochs)
        self._last_plateau_step_epoch = -1

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
            self.register_buffer("x_bias", sb)
            self.register_buffer("x_scale", ss)
            # Derive target scaler slices assuming target indices map into inputs.
            tgt_idx = int(getattr(self, "target_input_idx", 0))
            t_bias = sb[..., tgt_idx : tgt_idx + self.output_dim]
            t_scale = ss[..., tgt_idx : tgt_idx + self.output_dim]
            self.register_buffer("y_bias", t_bias)
            self.register_buffer("y_scale", t_scale)

    def _scale_inputs(self, x):
        if hasattr(self, "x_bias") and hasattr(self, "x_scale"):
            return (x - self.x_bias) / self.x_scale + 5e-8
        return x

    def _scale_targets(self, y):
        if hasattr(self, "y_bias") and hasattr(self, "y_scale"):
            return (y - self.y_bias) / self.y_scale + 5e-8
        return y

    def forward(self, x):
        # Optional ablation: zero out target input channel to force exogenous-only encoding
        # Scale inputs on-device, then replace NaNs.
        x = self._scale_inputs(x)
        x = torch.nan_to_num(x)

        if hasattr(self, "zero_target_in_encoder") and self.zero_target_in_encoder:
            idx = int(getattr(self, "target_input_idx", 0))
            x = x.clone()
            x[:, :, idx] = 0.0
        latent_seq = self.encoder(x)
        z = self.attn_pool(latent_seq)
        # Decode using the full latent sequence as memory for better fidelity
        x_hat = self.decoder(latent_seq)
        return x_hat, z

    @staticmethod
    def _make_span_mask(
        batch_size, seq_len, frac=0.6, min_span=5, max_span=30, device=None
    ):
        device = device or "cpu"
        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        target_cov = int(frac * seq_len)
        for b in range(batch_size):
            covered = 0
            # guard in case of tiny seq_len
            if min_span > seq_len:
                mask[b, :] = True
                continue
            while covered < target_cov:
                L = torch.randint(
                    low=min_span,
                    high=max(min_span + 1, max_span + 1),
                    size=(1,),
                    device=device,
                ).item()
                start = torch.randint(
                    low=0, high=max(1, seq_len - L + 1), size=(1,), device=device
                ).item()
                mask[b, start : start + L] = True
                covered = int(mask[b].sum().item())
        return mask

    def _sup_contrastive_loss(self, z, station_ids):
        # supervised contrastive loss (NT-Xent style) with station labels
        # z: (B, D), station_ids: (B,)
        B = z.size(0)
        if B < 2 or station_ids is None:
            return None
        # normalize
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.t())  # (B, B)
        logits = sim / max(1e-8, self.contrastive_temperature)
        labels = station_ids.view(-1, 1)
        # positives where labels equal and not same index
        pos_mask = (labels == labels.t()) & (
            ~torch.eye(B, dtype=torch.bool, device=z.device)
        )
        # For each anchor, compute loss over positives
        exp_logits = torch.exp(logits) * (
            ~torch.eye(B, dtype=torch.bool, device=z.device)
        )
        denom = exp_logits.sum(dim=1, keepdim=True)  # (B, 1)
        # avoid division by zero
        denom = denom + 1e-8
        # positive logits
        pos_logits = torch.exp(logits) * pos_mask
        # sum over positives for each anchor
        pos_sum = pos_logits.sum(dim=1)
        # loss per anchor (only if has a positive)
        # - (1/|P(i)|) * sum_p log( exp(sim)/sum_all ) == - log( pos_sum / denom ) averaged by positives count mask
        loss_i = -torch.log((pos_sum / denom.squeeze(1)).clamp(min=1e-12))
        # mask anchors without positives so they don't contribute
        valid = (pos_mask.sum(dim=1) > 0).float()
        if valid.sum() == 0:
            return None
        loss = (loss_i * valid).sum() / valid.sum()
        return loss

    def triplet_loss(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

    def init_bias(self):
        for module in [
            self.input_projection,
            self.output_projection,
            self.embedding_layer,
        ]:  # likely dead code
            if module.bias is not None:
                nn.init.uniform_(module.bias, -0.1, 0.1)

    def training_step(self, batch, batch_idx):
        # Mark step boundary for cudagraph safety (no-op if unsupported)
        try:
            torch.compiler.cudagraph_mark_step_begin()
        except Exception:
            pass
        out = stack_batch(batch)
        # backward compatibility: allow 5 or 6-tuple
        if len(out) == 5:
            x, y, mask, x_pos, x_neg = out
            station_ids = None
        else:
            x, y, mask, x_pos, x_neg, station_ids = out
        bsz = x.size(0)

        # Schedule target-channel dropout (per-sample) over warmup epochs
        drop_p = float(self.target_channel_dropout_p)
        if drop_p > 0 and self.target_dropout_warmup_epochs > 0:
            f = min(
                1.0,
                float(self.current_epoch) / float(self.target_dropout_warmup_epochs),
            )
            drop_p = drop_p * f
        if drop_p > 0:
            drop = torch.rand((bsz,), device=x.device) < drop_p
            if drop.any():
                tidx = int(getattr(self, "target_input_idx", 0))
                x = x.clone()
                x[drop, :, tidx] = torch.nan  # becomes 0 after scaling + nan_to_num

        # Schedule span masking on target timesteps (per-sample) over warmup epochs
        span_frac = float(self.span_mask_frac)
        if span_frac > 0 and self.span_mask_warmup_epochs > 0:
            f = min(
                1.0, float(self.current_epoch) / float(self.span_mask_warmup_epochs)
            )
            span_frac = span_frac * f
        if span_frac > 0:
            T = x.size(1)
            span_mask = self._make_span_mask(bsz, T, frac=span_frac, device=x.device)
            tidx = int(getattr(self, "target_input_idx", 0))
            x = x.clone()
            # mask target channel at masked timesteps
            xt = x[:, :, tidx]
            xt[span_mask] = torch.nan
            x[:, :, tidx] = xt
        else:
            span_mask = torch.zeros((bsz, x.size(1)), dtype=torch.bool, device=x.device)

        # Forward scales inputs internally. Scale targets to the same space.
        y = self._scale_targets(y)
        y = torch.nan_to_num(y)

        y_hat, z = self(x)

        # Flatten for masked loss
        y_hat = y_hat.reshape(-1, self.output_dim)
        y = y.reshape(-1, self.output_dim)
        mask = mask[:, :, : self.output_dim].reshape(-1, self.output_dim)
        # Do not restrict reconstruction to masked spans; supervise all valid targets
        # This increases gradient signal and helps prevent early plateau.
        if span_mask.any():
            sm = span_mask.unsqueeze(-1).expand(-1, span_mask.size(1), self.output_dim)
            sm = sm.reshape(-1, self.output_dim)
            mask = mask  # previously restricted to masked-only spans

        # Log mean mask fraction for diagnostics
        try:
            self.log(
                "mask_frac",
                float(mask.float().mean().detach().cpu()),
                on_step=True,
                prog_bar=False,
                batch_size=bsz,
            )
        except Exception:
            pass

        y = y[mask]
        y_hat = y_hat[mask]
        nan_mask = ~torch.isnan(y_hat)

        loss = self.criterion(y[nan_mask], y_hat[nan_mask])

        # Supervised contrastive on station ids
        if station_ids is not None:
            c_loss = self._sup_contrastive_loss(z, station_ids)
            if c_loss is not None:
                loss = loss + self.contrastive_weight * c_loss
                try:
                    self.log(
                        "contrastive_loss", float(c_loss.detach().cpu()), batch_size=bsz
                    )
                except Exception:
                    pass

        if x_pos is not None and x_neg is not None:
            valid_pos = (~torch.isnan(x_pos)).float().mean()
            valid_neg = (~torch.isnan(x_neg)).float().mean()
            if valid_pos > 0.3 and valid_neg > 0.3:
                _, z_pos = self(x_pos)
                _, z_neg = self(x_neg)
                triplet_loss = self.triplet_loss(z, z_pos, z_neg)
                loss += triplet_loss * 2.0
                self.log("triplet_loss", triplet_loss, batch_size=bsz)

        # Log CPU scalars to avoid compiled graph interactions in Lightning internals
        rec_val = float(loss.detach().cpu())
        self.log("reconstruction_loss", rec_val, batch_size=bsz)
        self.log("train_loss", rec_val, on_step=True, batch_size=bsz)
        try:
            self.log(
                "span_mask_frac",
                float(span_mask.float().mean().detach().cpu()),
                on_step=True,
                batch_size=bsz,
            )
        except Exception:
            pass

        return loss

    def validation_step(self, batch, batch_idx):
        # Mark step boundary for cudagraph safety (no-op if unsupported)
        try:
            torch.compiler.cudagraph_mark_step_begin()
        except Exception:
            pass
        out = stack_batch(batch)
        if len(out) == 5:
            x, y, mask, x_pos, x_neg = out
        else:
            x, y, mask, x_pos, x_neg, _ = out
        bsz = x.size(0)

        # Scale targets to match network output space.
        y = self._scale_targets(y)

        y_hat, z = self(x)

        # Flatten and mask
        y_hat = y_hat.reshape(-1, self.output_dim)
        y = y.reshape(-1, self.output_dim)
        mask = mask[:, :, : self.output_dim].reshape(-1, self.output_dim)

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
        self.log(
            "val_loss",
            val_val,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=bsz,
        )
        # Explicit epoch-aggregated metric for LR scheduler monitoring
        self.log(
            "val_loss_epoch",
            val_val,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=bsz,
        )

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "lr",
            float(current_lr),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=bsz,
        )
        lr_ratio = current_lr / self.learning_rate
        self.log(
            "lr_ratio",
            float(lr_ratio),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=bsz,
        )

        return loss_obs

    # Removed manual scheduler stepping; Lightning steps ReduceLROnPlateau using monitor='val_loss_epoch'.

    def configure_optimizers(self):
        # Prefer fused AdamW when available for GPU speedups.
        try:
            optimizer = optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=1e-4, fused=True
            )
        except TypeError:
            optimizer = optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=1e-4
            )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, threshold=1e-3, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss_epoch",
                "interval": "epoch",
                "reduce_on_plateau": True,
            },
        }

    def on_before_optimizer_step(self, optimizer):
        try:
            total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            # Log scalar grad norm (CPU) for stability checks
            self.log(
                "grad_norm",
                float(total_norm.detach().cpu())
                if hasattr(total_norm, "detach")
                else float(total_norm),
                on_step=True,
                prog_bar=False,
            )
        except Exception:
            pass


def stack_batch(batch):
    # Support tuples of length 5 or 6 (with optional station ids)
    if (
        isinstance(batch, (tuple, list))
        and len(batch) in (5, 6)
        and torch.is_tensor(batch[0])
    ):
        if len(batch) == 5:
            x, y, mask, pos, neg = batch
            return x, y, mask, pos, neg
        else:
            x, y, mask, pos, neg, sids = batch
            return x, y, mask, pos, neg, sids

    x, y, mask, pos, neg = [], [], [], [], []

    for item in batch:
        if len(item) == 6:
            xi, yi, mi, pi, ni, si = item
        else:
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
        # station ids collected separately below

    x = torch.stack(x)
    y = torch.stack(y)
    mask = torch.stack(mask)
    pos = torch.stack(pos)
    neg = torch.stack(neg)
    # return sids if present
    if len(batch[0]) == 6:
        sids = torch.as_tensor([it[5] for it in batch], dtype=torch.long)
        return x, y, mask, pos, neg, sids
    return x, y, mask, pos, neg


if __name__ == "__main__":
    pass
# ========================= EOF ====================================================================
