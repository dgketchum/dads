import json
import os
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import resource
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from models.scalers import MinMaxScaler
from models.simple_lstm.lstm import LSTMPredictor

device_name = None
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f'Using GPU: {device_name}')
else:
    print('CUDA is not available. PyTorch will use the CPU.')

torch.set_float32_matmul_precision('medium')
torch.cuda.get_device_name(torch.cuda.current_device())

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

from prep.simple_lstm.build_training_sequences import MET_FEATURES, GEO_FEATURES, ADDED_FEATURES


class PTHLSTMDataset(Dataset):
    def __init__(self, file_paths, station_names, col_names, col_ind, chunk_size, strided=False, transform=None,
                 return_station_name=False, scaler_sample_n=10000):

        self.file_paths = file_paths
        self.chunk_size = chunk_size
        self.transform = transform
        self.col_names = col_names
        self.col_indices = col_ind
        self.return_station_name = return_station_name
        self.strided = strided
        self.data_info = file_paths
        self.station_names = station_names
        self.scaler_sample_n = scaler_sample_n
        self.sample_stats = None

        self._fit_scaler()

    def _fit_scaler(self):
        sample_data = []
        nan_ct = 0
        for file_path in self.file_paths:
            try:
                data = torch.load(file_path, weights_only=True)
                data = data[:, self.col_indices]
                if torch.any(torch.isnan(data)):
                    nan_ct += 1
                    continue
                sample_data.append(data)
                if len(sample_data) > self.scaler_sample_n:
                    break

            except (FileNotFoundError, RuntimeError) as e:
                print(f"Error reading file {file_path}: {e}")
                continue
        print(f'{nan_ct} training samples had nan')
        if sample_data:
            sample_data = torch.cat(sample_data, dim=0)
            if len(sample_data) > 0:
                variable_stats = {}
                for i, (col, col_str) in enumerate(zip(self.col_indices, self.col_names)):
                    variable_stats[col] = {'column': col_str,
                                           'min': sample_data[:, i].min().item(),
                                           'max': sample_data[:, i].max().item()}

                self.sample_stats = variable_stats
                all_data = np.array(sample_data)
                self.scaler = MinMaxScaler(axis=0)
                self.scaler.fit(all_data)
            else:
                print("Not enough data to estimate scaler parameters. Using default scaler.")
                self.scaler = MinMaxScaler()

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):

        while True:
            file_path = self.file_paths[idx]
            chunk = torch.load(file_path, weights_only=True)
            chunk = chunk[:, self.col_indices]

            if not torch.any(torch.isnan(chunk)):
                break

            idx = random.randint(0, len(self.file_paths) - 1)

        if self.scaler:
            chunk = torch.tensor(self.scaler.transform(chunk.numpy()), dtype=torch.float32)

        y = chunk[:, 0]
        gm = chunk[:, 1]
        lf = chunk[:, 2:]

        if self.return_station_name:
            station_name = self.station_names[idx]
            return y, gm, lf, station_name
        else:
            return y, gm, lf

    def save_scaler(self, scaler_path):
        if self.scaler:
            bias_ = self.scaler.bias.flatten().tolist()
            scale_ = self.scaler.scale.flatten().tolist()
            dct = {'bias': bias_, 'scale': scale_}
            with open(scaler_path, 'w') as fp:
                json.dump(dct, fp, indent=4)
            print(f"Scaler saved to {scaler_path}")
        else:
            print("Scaler has not been fitted yet.")


def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return default_collate(batch)


def train_model(dirpath, sequence_data, target_var, columns, batch_size=1, learning_rate=0.01, n_workers=1,
                chunk_size=16, strided=False, logging_csv=None, scaler_sample_n=10000):
    """"""

    target, comparison = f'{target_var}_obs', f'{target_var}_dm'

    target_idx = [columns.index(target)]
    comp_idx = [columns.index(comparison)]
    met_idx = [columns.index(p) for p in MET_FEATURES]
    geo_idx = [columns.index(p) for p in GEO_FEATURES]
    add_idx = [columns.index(p) for p in ADDED_FEATURES]

    idx = target_idx + comp_idx + met_idx + geo_idx + add_idx

    # limit = 1000
    file_map = defaultdict(list)
    for filename in os.listdir(sequence_data):
        station = filename.split('_')[0]
        file_map[station].append(os.path.join(sequence_data, filename))
        # if len(file_map) > limit:
        #     break

    stations = list(file_map.keys())
    random.seed(1234)
    random.shuffle(stations)

    t_stations = stations[:int(len(stations) * 0.7)]
    v_stations = stations[int(len(stations) * 0.7):]

    t_files = [f for station in t_stations for f in file_map[station]]
    v_files = [f for station in v_stations for f in file_map[station]]

    selected_columns = list(np.array(columns)[idx])

    train_dataset = PTHLSTMDataset(file_paths=t_files, station_names=t_stations, col_names=selected_columns,
                                   col_ind=idx, chunk_size=chunk_size, strided=strided, scaler_sample_n=scaler_sample_n)

    if strided:
        print(f'\nTrain dataset: {len(train_dataset)} {chunk_size} x {len(columns)} strided samples')
    else:
        print(f'\nTrain dataset: {len(train_dataset)} {chunk_size} x {len(columns)} non-overlapping samples')

    print(f'Batch size: {batch_size}, Sequence Length: {chunk_size}, GPU: {device_name}, '
          f'Scaler Sample: {scaler_sample_n}')

    train_dataset.save_scaler(os.path.join(dirpath, 'scaler.json'))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers,
                                  collate_fn=custom_collate)

    val_dataset = PTHLSTMDataset(file_paths=v_files, station_names=v_stations, col_names=selected_columns, col_ind=idx,
                                 chunk_size=chunk_size, scaler_sample_n=scaler_sample_n)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers,
                                collate_fn=custom_collate)

    model = LSTMPredictor(num_bands=len(idx) - 2,
                          learning_rate=learning_rate,
                          dropout_rate=0.1,
                          expansion_factor=4,
                          log_csv=logging_csv,
                          scaler=train_dataset.scaler)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="best_model",
        dirpath=dirpath,
        save_top_k=1,
        mode="min"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=100,
        verbose=False,
        mode="min",
        check_finite=True,
    )

    trainer = pl.Trainer(max_epochs=1000, callbacks=[checkpoint_callback, early_stop_callback],
                         accelerator='gpu', devices=1)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    target_var_ = 'mean_temp'

    if device_name == 'NVIDIA GeForce RTX 2080':
        workers = 6
    elif device_name == 'NVIDIA RTX A6000':
        workers = 12
    else:
        raise NotImplementedError('Specify the machine this is running on')

    zoran = '/data/ssd2/dads/training/simple_lstm'
    nvm = '/media/nvm/training/simple_lstm'
    if os.path.exists(zoran):
        print('modeling with data from zoran')
        training = zoran
    elif os.path.exists(nvm):
        print('modeling with data from NVM drive')
        training = nvm
    else:
        print('modeling with data from UM drive')
        training = os.path.join(d, 'training')

    print('========================== modeling {} =========================='.format(target_var_))

    metadata_ = os.path.join(training, 'combined_stats.json')
    sequences = os.path.join(training, 'pth')

    now = datetime.now().strftime('%m%d%H%M')
    chk = os.path.join(training, 'checkpoints', now)
    os.mkdir(chk)
    logger_csv = os.path.join(chk, 'training_{}.csv'.format(now))

    with open(os.path.join(training, 'pth_columns.json'), 'r') as fp:
        cols = json.load(fp)['columns']

    train_model(chk, sequences, target_var_, cols, batch_size=512, learning_rate=0.01, n_workers=workers,
                logging_csv=logger_csv, scaler_sample_n=11000)
# ========================= EOF ====================================================================
