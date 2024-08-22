import os
import json
import resource

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.metrics import r2_score, root_mean_squared_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

device_name = None
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {device_name}")
else:
    print("CUDA is not available. PyTorch will use the CPU.")

torch.set_float32_matmul_precision('medium')
torch.cuda.get_device_name(torch.cuda.current_device())

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

torch.multiprocessing.set_sharing_strategy('file_system')


class PTHLSTMDataset(Dataset):
    def __init__(self, file_path, chunk_size, transform=None):
        self.data = torch.load(file_path, weights_only=True)
        self.chunk_size = chunk_size
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]

        valid_rows = ~torch.isnan(chunk).any(dim=1)
        chunk = chunk[valid_rows]

        if len(chunk) < self.chunk_size:
            return None

        x, y, gm = chunk[:, 2:], chunk[:, 0], chunk[:, 1]
        return x, y, gm


def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return default_collate(batch)


def stack_batch(batch):
    x = torch.stack([item[0] for item in batch])
    y = torch.stack([item[1] for item in batch])
    g = torch.stack([item[2] for item in batch])
    return x, y, g


class LSTMPredictor(pl.LightningModule):
    def __init__(self, num_bands=10, hidden_size=64, num_layers=2, learning_rate=0.001,
                 expansion_factor=2, dropout_rate=0.5):
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
        x = x.squeeze()
        x = self.input_expansion(x)
        out, _ = self.bilstm(x)
        out = out[:, -1, :]
        out = self.output_layers(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y_obs, y_gm = stack_batch(batch)

        x = x.squeeze()
        y_hat = self(x).squeeze()
        y_obs = y_obs.squeeze()[:, -1]

        loss = self.criterion(y_hat, y_obs)
        self.log("train_loss", loss)

        self.logger.experiment.add_scalar("Loss/Train", loss, self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_obs, y_gm = batch

        y_hat = self(x).squeeze()
        y_obs = y_obs.squeeze()[:, -1]
        y_gm = y_gm.squeeze()[:, -1]

        loss_obs = self.criterion(y_hat, y_obs)
        self.log("val_loss", loss_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        y_hat_obs_np = y_hat.detach().cpu().numpy()
        y_obs_np = y_obs.detach().cpu().numpy()
        y_gm_np = y_gm.detach().cpu().numpy()

        r2_obs = r2_score(y_obs_np, y_hat_obs_np)
        rmse_obs = root_mean_squared_error(y_obs_np, y_hat_obs_np)
        r2_gm = r2_score(y_obs_np, y_gm_np)
        rmse_gm = root_mean_squared_error(y_obs_np, y_gm_np)

        self.log("val_r2", r2_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_r2_gm", r2_gm, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.log("val_rmse", rmse_obs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_rmse_gm", rmse_gm, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.logger.experiment.add_scalar("Loss/Val_Obs", loss_obs, self.current_epoch)

        self.logger.experiment.add_scalar("Metrics/Val_R2", r2_obs, self.current_epoch)
        self.logger.experiment.add_scalar("Metrics/Val_R2_GM", r2_gm, self.current_epoch)
        self.logger.experiment.add_scalar("Metrics/Val_RMSE", rmse_obs, self.current_epoch)
        self.logger.experiment.add_scalar("Metrics/Val_RMSE_GM", rmse_gm, self.current_epoch)

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


def train_model(pth, metadata, batch_size=1, learning_rate=0.01, n_workers=1):
    """"""

    with open(metadata, 'r') as f:
        meta = json.load(f)

    chunk_size = meta['chunk_size']

    features = [c for c, s in zip(meta['column_order'], meta['scaling_status']) if s == 'scaled']
    feature_len = len(features)

    train_file = os.path.join(pth, 'train', 'all_data.pth')
    train_dataset = PTHLSTMDataset(train_file, chunk_size=chunk_size)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers,
                                  collate_fn=lambda batch: [x for x in batch if x is not None])

    val_file = os.path.join(pth, 'val', 'all_data.pth')
    val_dataset = PTHLSTMDataset(val_file, chunk_size=chunk_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers,
                                collate_fn=custom_collate)

    model = LSTMPredictor(num_bands=feature_len, learning_rate=learning_rate)

    logger = TensorBoardLogger(save_dir=param_dir, name="lstm_logs")

    early_stopping = EarlyStopping(monitor="val_loss", patience=50, mode="min")
    trainer = pl.Trainer(max_epochs=1000, callbacks=[early_stopping], logger=logger)

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    target_var = 'mean_temp'

    if device_name == 'NVIDIA GeForce RTX 2080':
        workers = 6
    elif device_name == 'NVIDIA RTX A6000':
        workers = 6
    else:
        raise NotImplementedError('Specify the machine this is running on')

    zoran = '/home/dgketchum/training'
    nvm = '/media/nvm/training'
    if os.path.exists(zoran):
        print('writing to zoran')
        training = zoran
    elif os.path.exists(nvm):
        print('writing to NVM drive')
        training = nvm
    else:
        print('writing to UM drive')
        training = os.path.join(d, 'training')

    print('========================== modeling {} =========================='.format(target_var))

    param_dir = os.path.join(training, target_var)
    pth_ = os.path.join(param_dir, 'scaled_pth')
    metadata_ = os.path.join(param_dir, 'training_metadata.json')

    train_model(pth_, metadata_, batch_size=256, learning_rate=0.0001, n_workers=workers)
# ========================= EOF ====================================================================
