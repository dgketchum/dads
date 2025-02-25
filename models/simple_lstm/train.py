import os
import json
import resource
import random
from datetime import datetime

import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
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
    def __init__(self, file_paths, station_names, col_ind, chunk_size, stats, strided=False,
                 transform=None, return_station_name=False,
                 scaler_json=None):
        self.file_paths = file_paths
        self.chunk_size = chunk_size
        self.transform = transform
        self.col_indices = col_ind
        self.return_station_name = return_station_name
        self.strided = strided
        self.data_info = file_paths
        self.station_names = station_names
        self.scaler_json = scaler_json

        if scaler_json:
            with open(scaler_json, 'r') as f:
                dct = json.load(f)
            self.scaler = MinMaxScaler()
            self.scaler.bias_ = np.array(dct['bias']).reshape(1, -1)
            self.scaler.scale_ = np.array(dct['scale']).reshape(1, -1)
        else:
            self.scaler = None
            self._fit_scaler(stats)

    def _fit_scaler(self, stats_json):
        with open(stats_json, 'r') as f:
            stats_dict = json.load(f)

        # Initialize a dictionary to store min/max values for each variable
        variable_stats = {}
        for station, stats in stats_dict.items():
            for col, values in stats.items():
                if col not in variable_stats:
                    variable_stats[col] = {'min': values['min'], 'max': values['max']}
                else:
                    variable_stats[col]['min'] = min(variable_stats[col]['min'], values['min'])
                    variable_stats[col]['max'] = max(variable_stats[col]['max'], values['max'])

        all_data = []
        for col, values in variable_stats.items():
            all_data.append([values['min'], values['max']])

        all_data = np.array(all_data)
        self.scaler = MinMaxScaler()
        self.scaler.fit(all_data)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        file_path = self.data_info[idx]
        chunk = torch.load(file_path, weights_only=True)

        if self.scaler:
            chunk = torch.tensor(self.scaler.transform(chunk.numpy()), dtype=torch.float32)

        y = chunk[:, self.col_indices[0]]
        gm = chunk[:, self.col_indices[1]]
        lf = chunk[:, self.col_indices[2:]]

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


def train_model(dirpath, sequence_data, metadata, target_var, columns, batch_size=1, learning_rate=0.01, n_workers=1,
                chunk_size=16, strided=False, logging_csv=None):
    """"""

    target, comparison = f'{target_var}_obs', f'{target_var}_dm'

    target_idx = [columns.index(target)]
    comp_idx = [columns.index(comparison)]
    met_idx = [columns.index(p) for p in MET_FEATURES]
    geo_idx = [columns.index(p) for p in GEO_FEATURES]
    add_idx = [columns.index(p) for p in ADDED_FEATURES]

    idx = target_idx + comp_idx + met_idx + geo_idx + add_idx

    stations = list(set([f.split('_')[0] for f in os.listdir(sequence_data)]))
    random.seed(1234)
    random.shuffle(stations)
    t_stations = stations[:int(len(stations) * 0.7)]
    v_stations = stations[int(len(stations) * 0.7):]

    t_files = [os.path.join(sequence_data, f) for f in os.listdir(sequence_data) if f.split('_')[0] in t_stations]
    v_files = [os.path.join(sequence_data, f) for f in os.listdir(sequence_data) if f.split('_')[0] in v_stations]

    train_dataset = PTHLSTMDataset(file_paths=t_files,
                                   station_names=t_stations,
                                   col_ind=idx,
                                   chunk_size=chunk_size,
                                   strided=strided,
                                   stats=metadata)

    if strided:
        print(f'\nTrain dataset: {len(train_dataset)} {chunk_size} x {len(columns)} strided samples')
    else:
        print(f'\nTrain dataset: {len(train_dataset)} {chunk_size} x {len(columns)} non-overlapping samples')

    print(f'Batch size: {batch_size}, Sequence Length: {chunk_size}, GPU: {device_name}')

    train_dataset.save_scaler(os.path.join(dirpath, 'scaler.json'))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers,
                                  collate_fn=lambda batch: [x for x in batch if x is not None])

    val_dataset = PTHLSTMDataset(file_paths=v_files,
                                 station_names=v_stations,
                                 col_ind=idx,
                                 chunk_size=chunk_size,
                                 stats=metadata)

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

    zoran = '/data/ssd2/training/simple_lstm'
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
    out_csv = os.path.join(training, 'pth')

    with open(os.path.join(training, 'pth_columns.json'), 'r') as fp:
        cols = json.load(fp)['columns']

    train_model(chk, sequences, metadata_, target_var_, cols, batch_size=512, learning_rate=0.01, n_workers=workers,
                logging_csv=logger_csv)
# ========================= EOF ====================================================================
