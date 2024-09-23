import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random


# ... (WeatherDataset and robust_scale functions from previous responses) ...

class ClimateAutoencoder(nn.Module):
    def __init__(self, input_size=8, embedding_size=64, hidden_size=128, num_layers=3):
        super(ClimateAutoencoder, self).__init__()

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=4),
            num_layers=num_layers
        )
        self.embedding_layer = nn.Linear(input_size, embedding_size)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_size, nhead=4),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(embedding_size, input_size)

        # Auxiliary task head for predicting year (adjust if using decade or other period)
        self.year_predictor = nn.Linear(embedding_size, 1)

    def forward(self, x, mask=None):
        encoded = self.encoder(x, src_key_padding_mask=mask)
        embedding = self.embedding_layer(encoded[:, 0, :])  # Use first token's embedding
        decoded = self.decoder(embedding.unsqueeze(1), encoded, tgt_key_padding_mask=mask)
        output = self.output_layer(decoded)

        year_pred = self.year_predictor(embedding)
        return output, embedding, year_pred


def train_autoencoder(model, dataloaders, optimizer, criterion, num_epochs, chunk_size=72 * 24, alpha=0.5):
    # Dataloaders should be a dictionary: {'short': short_seq_dataloader, 'long': long_seq_dataloader}

    for epoch in range(num_epochs):
        for phase in ['short', 'long']:  # Curriculum learning: start with short sequences
            if phase == 'long' and epoch < num_epochs // 2:
                continue

            model.train() if phase == 'train' else model.eval()
            for batch_data in dataloaders[phase]:
                seq_len = batch_data.size(1)

                if phase == 'long':
                    start_idx = random.randint(0, seq_len - chunk_size)
                    chunk = batch_data[:, start_idx:start_idx + chunk_size, :]
                    mask = None
                else:
                    chunk = batch_data
                    mask = torch.zeros(batch_data.size(0), batch_data.size(1), dtype=torch.bool)
                    for i in range(batch_data.size(0)):
                        actual_len = (batch_data[i, :, 0] != 0).sum().item()
                        mask[i, actual_len:] = True

                optimizer.zero_grad()
                output, _, year_pred = model(chunk, mask)

                recon_loss = criterion(output, chunk)
                if mask is not None:
                    recon_loss = (recon_loss * ~mask.unsqueeze(-1)).mean()
                else:
                    recon_loss = recon_loss.mean()

                year_loss = criterion(year_pred, batch_data[:, 0, -1].unsqueeze(-1))  # Assuming last column is year

                loss = recon_loss + alpha * year_loss

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            epoch_loss = loss.item()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Phase: {phase}, Loss: {epoch_loss:.4f}')

if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
