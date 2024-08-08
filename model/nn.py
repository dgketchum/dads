import json
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import r2_score, root_mean_squared_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {device_name}")
else:
    print("CUDA is not available. PyTorch will use the CPU.")

torch.set_float32_matmul_precision('medium')
torch.cuda.get_device_name(torch.cuda.current_device())


class PTHLSTMDataset(Dataset):
    def __init__(self, file_list, chunks_per_file, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.chunks_per_file = chunks_per_file

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_idx = idx // self.chunks_per_file
        chunk_idx = idx % self.chunks_per_file

        file_path = self.file_list[file_idx]
        data = torch.load(file_path, weights_only=True)
        chunk = data[chunk_idx]
        return chunk[:, 2:], chunk[:, 0], chunk[:, 1]


def stack_batch(batch):
    x = torch.stack([item[0] for item in batch])
    y = torch.stack([item[1] for item in batch])
    g = torch.stack([item[2] for item in batch])
    return x, y, g


class LSTMPredictor(pl.LightningModule):
    def __init__(self, num_bands=10, hidden_size=64, num_layers=2, learning_rate=0.001, expansion_factor=2,
                 dropout_rate=0.5):
        super().__init__()

        self.input_expansion = nn.Sequential(
            nn.Linear(num_bands, num_bands * expansion_factor),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.bilstm = nn.LSTM(num_bands * expansion_factor, hidden_size, num_layers, batch_first=True,
                              bidirectional=True)

        self.output_layers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1)
        )

        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.input_expansion(x)
        out, _ = self.bilstm(x)
        out = out[:, -1, :]
        out = self.output_layers(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y, _ = stack_batch(batch)
        y = y[:, -1]
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_obs, y_gm = batch
        y_hat_obs = self(x).squeeze()
        y_obs = y_obs[:, -1].squeeze()

        loss_obs = self.criterion(y_hat_obs, y_obs)
        self.log("val_loss_obs", loss_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        y_hat_obs_np = y_hat_obs.detach().cpu().numpy()
        y_obs_np = y_obs.detach().cpu().numpy()
        y_gm_np = y_gm.detach().cpu().numpy()

        r2_obs = r2_score(y_obs_np, y_hat_obs_np)
        rmse_obs = root_mean_squared_error(y_obs_np, y_hat_obs_np)
        r2_gm = r2_score(y_obs_np, y_gm_np[:, -1])
        rmse_gm = root_mean_squared_error(y_obs_np, y_gm_np[:, -1])

        self.log("val_r2_obs", r2_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_r2_gm", r2_gm, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.log("val_rmse_obs", rmse_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_rmse_gm", rmse_gm, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.log("mean_pred", y_hat_obs_np.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("mean_gm", y_gm_np.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss_obs

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss_obs'}}


def train_model(pth, metadata, learning_rate=0.01):
    """"""

    with open(metadata, 'r') as f:
        meta = json.load(f)

    chunk_size = meta['chunk_size']
    chunks_per_file = meta['chunks_per_file']

    features = [c for c, s in zip(meta['column_order'], meta['scaling_status']) if s == 'scaled']
    feature_len = len(features)

    train_dir = os.path.join(pth, 'train')
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    train_dataset = PTHLSTMDataset(train_files, chunks_per_file=chunks_per_file)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=256,
                                  shuffle=True,
                                  num_workers=6,
                                  collate_fn=lambda batch: [x for x in batch if x is not None])

    val_dir = os.path.join(pth, 'val')
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir)]
    val_dataset = PTHLSTMDataset(val_files, chunks_per_file=chunks_per_file)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=6)

    model = LSTMPredictor(num_bands=feature_len, learning_rate=learning_rate)

    early_stopping = EarlyStopping(monitor="val_loss_obs", patience=20, mode="min")
    trainer = pl.Trainer(max_epochs=100, callbacks=[early_stopping])

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    target_var = 'vpd'

    fields = os.path.join(d, 'met', 'stations', 'dads_stations_WMT_mgrs.csv')
    pth_ = os.path.join(d, 'training', target_var, 'scaled_pth')
    metadata_ = os.path.join(d, 'training', target_var, 'training_metadata.json')

    train_model(pth_, metadata_)
# ========================= EOF ====================================================================
