import os
import csv
import json
import resource
from datetime import datetime

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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
    def __init__(self, file_paths, col_index, chunk_size, transform=None):
        self.chunk_size = chunk_size
        self.transform = transform
        self.col_index = col_index

        all_data = []
        for file_path in file_paths:
            data = torch.load(file_path, weights_only=True)
            all_data.append(data)

        self.data = torch.cat(all_data, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]

        valid_rows = ~torch.isnan(chunk).any(dim=1)
        chunk = chunk[valid_rows]

        if len(chunk) < self.chunk_size:
            return None

        y, gm, lf, hf = (chunk[:, self.col_index[0]],
                         chunk[:, self.col_index[1]],
                         chunk[:, self.col_index[2]: self.col_index[3]],
                         chunk[:, self.col_index[3]:])

        return y, gm, lf, hf


def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return default_collate(batch)


def stack_batch(batch):
    y = torch.stack([item[0] for item in batch])
    gm = torch.stack([item[1] for item in batch])
    lf = torch.stack([item[2] for item in batch])
    hf = torch.stack([item[3] for item in batch])
    return y, gm, lf, hf


class LSTMPredictor(pl.LightningModule):
    def __init__(self,
                 num_bands_lf=32,
                 num_bands_hf=14,
                 hidden_size=64,
                 num_layers=2,
                 learning_rate=0.01,
                 expansion_factor=2,
                 dropout_rate=0.1,
                 log_csv=None):
        super().__init__()

        self.input_expansion_hf = nn.Sequential(
            nn.Linear(num_bands_hf, num_bands_hf * expansion_factor),
            nn.ReLU(),
            nn.Linear(num_bands_hf * expansion_factor, num_bands_hf * expansion_factor * 2),
            nn.ReLU(),
            nn.Linear(num_bands_hf * expansion_factor * 2, num_bands_hf * expansion_factor * 4),
            nn.ReLU(),
        )

        self.input_expansion_lf = nn.Sequential(
            nn.Linear(num_bands_lf, num_bands_lf * expansion_factor),
            nn.ReLU(),
            nn.Linear(num_bands_lf * expansion_factor, num_bands_lf * expansion_factor * 2),
            nn.ReLU(),
        )

        self.lstm_hf = nn.LSTM(num_bands_hf * expansion_factor * 4, hidden_size, num_layers, batch_first=True,
                               bidirectional=True)

        self.lstm_lf = nn.LSTM(num_bands_lf * expansion_factor * 2, hidden_size, num_layers, batch_first=True,
                               bidirectional=True)

        self.fc1 = nn.Linear(4 * hidden_size, hidden_size * 4)

        self.output_layers = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1)
        )

        self.criterion = nn.L1Loss()
        self.learning_rate = learning_rate

        self.log_csv = log_csv

    def forward(self, x_lf, x_hf):
        x_lf = x_lf.squeeze()
        x_lf = self.input_expansion_lf(x_lf)
        out_lf, _ = self.lstm_lf(x_lf)
        out_lf = out_lf[:, -1, :]

        x_hf = x_hf.squeeze()
        x_hf = self.input_expansion_hf(x_hf)
        out_hf, _ = self.lstm_hf(x_hf)
        out_hf = out_hf[:, -1, :]

        combined = torch.cat((out_hf, out_lf), dim=1)
        out = self.fc1(combined)
        out = self.output_layers(out).squeeze()
        return out

    def training_step(self, batch, batch_idx):
        y, gm, lf, hf = stack_batch(batch)
        y_hat = self(lf, hf)
        y_obs = y[:, -1]

        loss = self.criterion(y_hat, y_obs)
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):

        if self.log_csv:
            train_loss = self.trainer.callback_metrics["train_loss"].item()
            val_loss = self.trainer.callback_metrics["val_loss"].item()
            val_r2 = self.trainer.callback_metrics["val_r2"].item()
            gm_r2 = self.trainer.callback_metrics["val_r2_gm"].item()
            val_rmse = self.trainer.callback_metrics["val_rmse"].item()
            gm_rmse = self.trainer.callback_metrics["val_rmse_gm"].item()
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']

            log_data = [self.current_epoch,
                        round(train_loss, 4),
                        round(val_loss, 4),
                        round(val_r2, 4),
                        round(gm_r2, 4),
                        round(val_rmse, 4),
                        round(gm_rmse, 4),
                        round(current_lr, 4)]

            with open(self.log_csv, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(log_data)

            if self.current_epoch == 0:
                with open(self.log_csv, "r+", newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    f.seek(0)
                    writer = csv.writer(f)
                    writer.writerow(["epoch", "train_loss", "val_loss", "val_r2", "val_rmse", "gm_r2", "gm_rmse",
                                     "lr"])
                    writer.writerow(header)

    def validation_step(self, batch, batch_idx):
        y, gm, lf, hf = batch
        y_hat = self(lf, hf).squeeze()
        y_obs = y[:, -1]
        y_gm = gm[:, -1]

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

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr, on_step=False, on_epoch=True, prog_bar=True)
        lr_ratio = current_lr / self.learning_rate
        self.log("lr_rat", lr_ratio, on_step=False, on_epoch=True, prog_bar=True)

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


def train_model(pth, metadata, batch_size=1, learning_rate=0.01, n_workers=1, logging_csv=None):
    """"""

    with open(metadata, 'r') as f:
        meta = json.load(f)

    data_frequency = meta['data_frequency']
    # idxs: obs, gm, start_daily: end_daily, start_hourly:
    hf_idx = data_frequency.index('hf')
    idxs = (0, 1, 2, hf_idx)
    hf_bands = data_frequency.count('hf')
    lf_bands = data_frequency.count('lf') - 2
    chunk_size = meta['chunk_size']

    model = LSTMPredictor(num_bands_lf=lf_bands,
                          num_bands_hf=hf_bands,
                          learning_rate=learning_rate,
                          log_csv=logging_csv)

    tdir = os.path.join(pth, 'train')
    t_files = [os.path.join(tdir, f) for f in os.listdir(tdir)]
    train_dataset = PTHLSTMDataset(t_files, idxs, chunk_size=chunk_size)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers,
                                  collate_fn=lambda batch: [x for x in batch if x is not None])

    vdir = os.path.join(pth, 'train')
    v_files = [os.path.join(vdir, f) for f in os.listdir(vdir)]
    val_dataset = PTHLSTMDataset(v_files, idxs, chunk_size=chunk_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers,
                                collate_fn=custom_collate)

    early_stopping = EarlyStopping(monitor="val_loss", patience=100, mode="min")
    trainer = pl.Trainer(max_epochs=1000, callbacks=[early_stopping])

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    target_var = 'vpd'

    if device_name == 'NVIDIA GeForce RTX 2080':
        workers = 1
    elif device_name == 'NVIDIA RTX A6000':
        workers = 6
    else:
        raise NotImplementedError('Specify the machine this is running on')

    zoran = '/home/dgketchum/training'
    nvm = '/media/nvm/training'
    if os.path.exists(zoran):
        print('modeling with data from zoran')
        training = zoran
    elif os.path.exists(nvm):
        print('modeling with data from NVM drive')
        training = nvm
    else:
        print('modeling with data from UM drive')
        training = os.path.join(d, 'training')

    print('========================== modeling {} =========================='.format(target_var))

    param_dir = os.path.join(training, target_var)
    pth_ = os.path.join(param_dir, 'scaled_pth')
    metadata_ = os.path.join(param_dir, 'training_metadata.json')

    now = datetime.now().strftime('%m%d%H%M')
    logger_csv = os.path.join(param_dir, 'training_{}.csv'.format(now))

    train_model(pth_, metadata_, batch_size=64, learning_rate=0.01, n_workers=workers, logging_csv=logger_csv)
# ========================= EOF ====================================================================
