import os
import json
import resource
from datetime import datetime

from models.scalers import MinMaxScaler

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from models.lstm.lstm import LSTMPredictor

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


class PTHLSTMDataset(Dataset):
    def __init__(self, file_paths, col_index, expected_width, chunk_size, transform=None, return_station_name=False,
                 scaler_json=None):
        self.chunk_size = chunk_size
        self.transform = transform
        self.col_index = col_index
        self.return_station_name = return_station_name

        all_data = []
        self.station_names = []

        for file_path in file_paths:
            station_name = os.path.splitext(os.path.basename(file_path))[0]

            data = torch.load(file_path, weights_only=True)
            if data.shape[2] != expected_width:
                print(f"Skipping {file_path},shape mismatch. Expected {expected_width} columns, got {data.shape[2]}")
                continue

            for _ in range(len(data)):
                self.station_names.append(station_name)

            all_data.append(data)

        self.data = torch.cat(all_data, dim=0)

        if scaler_json:
            with open(scaler_json, 'r') as f:
                dct = json.load(f)
            self.scaler = MinMaxScaler(out_range=(0, 1.0), axis=0)
            self.scaler.bias = torch.tensor(dct['bias'], dtype=torch.float32).reshape(1, -1, expected_width)
            self.scaler.scale = torch.tensor(dct['scale'], dtype=torch.float32).reshape(1, -1, expected_width)
            self.data = self.scaler.transform(self.data)

        else:
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.data)
            self.data = self.scaler.transform(self.data)

    def __len__(self):
        return len(self.data)

    def save_scaler(self, scaler_path):

        bias_ = self.scaler.bias.flatten().tolist()
        scale_ = self.scaler.scale.flatten().tolist()
        dct = {'bias': bias_, 'scale': scale_}
        with open(scaler_path, 'w') as fp:
            json.dump(dct, fp, indent=4)
        print(f"Scaler saved to {scaler_path}")

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

        if self.return_station_name:
            station_name = self.station_names[idx]
            return y, gm, lf, hf, station_name
        else:
            return y, gm, lf, hf


def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return default_collate(batch)


def train_model(dirpath, pth, metadata, batch_size=1, learning_rate=0.01, n_workers=1, logging_csv=None):
    """"""

    with open(metadata, 'r') as f:
        meta = json.load(f)

    data_frequency = meta['data_frequency']
    # idxs: obs, gm, start_daily: end_daily, start_hourly:
    hf_idx = data_frequency.index('hf')
    idxs = (0, 1, 2, hf_idx)
    hf_bands = data_frequency.count('hf')
    lf_bands = data_frequency.count('lf') - 2
    tensor_width = data_frequency.count('lf') + data_frequency.count('hf')
    print('tensor cols: {}'.format(tensor_width))
    chunk_size = meta['chunk_size']

    tdir = os.path.join(pth, 'train')
    t_files = [os.path.join(tdir, f) for f in os.listdir(tdir)]
    train_dataset = PTHLSTMDataset(file_paths=t_files,
                                   col_index=idxs,
                                   expected_width=tensor_width,
                                   chunk_size=chunk_size)

    train_dataset.save_scaler(os.path.join(dirpath, 'scaler.json'))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers,
                                  collate_fn=lambda batch: [x for x in batch if x is not None])

    vdir = os.path.join(pth, 'val')
    v_files = [os.path.join(vdir, f) for f in os.listdir(vdir)]
    val_dataset = PTHLSTMDataset(file_paths=v_files,
                                 col_index=idxs,
                                 expected_width=tensor_width,
                                 chunk_size=chunk_size)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers,
                                collate_fn=custom_collate)

    model = LSTMPredictor(num_bands_lf=lf_bands,
                          num_bands_hf=hf_bands,
                          learning_rate=learning_rate,
                          expansion_factor=2,
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

    target_var = 'mean_temp'

    if device_name == 'NVIDIA GeForce RTX 2080':
        workers = 6
    elif device_name == 'NVIDIA RTX A6000':
        workers = 12
    else:
        raise NotImplementedError('Specify the machine this is running on')

    zoran = '/home/dgketchum/training/lstm'
    nvm = '/media/nvm/training/lstm'
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
    pth_ = os.path.join(param_dir, 'pth')
    metadata_ = os.path.join(param_dir, 'training_metadata.json')

    now = datetime.now().strftime('%m%d%H%M')
    chk = os.path.join(param_dir, 'checkpoints', now)
    os.mkdir(chk)
    logger_csv = os.path.join(chk, 'training_{}.csv'.format(now))
    # logger_csv = None

    train_model(chk, pth_, metadata_, batch_size=64, learning_rate=0.001, n_workers=workers, logging_csv=logger_csv)
# ========================= EOF ====================================================================
