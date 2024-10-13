import json
import os
import shutil
import resource
import pickle

import pytorch_lightning as pl
import torch
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

from models.scalers import MinMaxScaler
from models.autoencoder.weather_encoder import WeatherAutoencoder

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


class WeatherDataset(Dataset):
    def __init__(self, file_paths, expected_width, data_width, chunk_size, transform=None):
        self.chunk_size = chunk_size
        self.transform = transform
        self.data_width = data_width

        all_data = []
        for file_path in file_paths:
            data = torch.load(file_path, weights_only=True)
            if data.shape[2] != expected_width:
                print(f"Skipping {file_path},shape mismatch. Expected {expected_width} columns, got {data.shape[2]}")
                continue
            all_data.append(data)

        self.data = torch.cat(all_data, dim=0)

        self.scaler = MinMaxScaler()
        self.scaler.fit(self.get_valid_data_for_scaling())

    def scale_chunk(self, chunk):
        chunk_np = chunk.numpy()
        reshaped_chunk = chunk_np.reshape(-1, chunk_np.shape[-1])
        scaled_chunk = self.scaler.transform(reshaped_chunk)
        return torch.from_numpy(scaled_chunk.reshape(chunk_np.shape))

    def get_valid_data_for_scaling(self):
        valid_data = []
        for chunk in self.data:
            chunk_without_pe = chunk[:, :self.data_width]
            valid_rows = chunk_without_pe[~torch.isnan(chunk_without_pe).any(dim=1)]
            valid_data.append(valid_rows)
        return torch.cat(valid_data, dim=0).numpy()

    def save_scaler(self, scaler_path):

        bias_ = self.scaler.bias.flatten().tolist()
        scale_ = self.scaler.scale.flatten().tolist()
        dct = {'bias': bias_, 'scale': scale_}
        with open(scaler_path, 'w') as fp:
            json.dump(dct, fp, indent=4)
        print(f"Scaler saved to {scaler_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        chunk = self.data[idx]
        chunk[:, :self.data_width] = self.scale_chunk(chunk[:, :self.data_width])

        # pytorch has counterintuitive mask logic (opposite)
        # i.e., False where there is valid data
        mask = ~torch.isnan(chunk[:, 0])

        return chunk, mask


def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return default_collate(batch)


def train_model(dirpath, pth, metadata, target, batch_size=64, learning_rate=0.01,
                n_workers=1, logging_csv=None):
    """"""

    with open(metadata, 'r') as f:
        meta = json.load(f)

    chunk_size = meta['chunk_size']
    tensor_width = len(meta['column_order'])
    data_width = len(meta['data_columns'])

    tdir = os.path.join(pth, 'train')
    t_files = [os.path.join(tdir, f) for f in os.listdir(tdir)]
    train_dataset = WeatherDataset(file_paths=t_files,
                                   expected_width=tensor_width,
                                   data_width=data_width,
                                   chunk_size=chunk_size)

    train_dataset.save_scaler(os.path.join(dirpath, 'scaler.json'))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers,
                                  collate_fn=lambda batch: [x for x in batch if x is not None])

    vdir = os.path.join(pth, 'val')
    v_files = [os.path.join(vdir, f) for f in os.listdir(vdir)]
    val_dataset = WeatherDataset(file_paths=v_files,
                                 expected_width=tensor_width,
                                 data_width=data_width,
                                 chunk_size=chunk_size)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=n_workers,
                                collate_fn=lambda batch: [x for x in batch if x is not None])

    model = WeatherAutoencoder(input_dim=tensor_width,
                               latent_size=1,
                               dropout=0.1,
                               learning_rate=learning_rate,
                               log_csv=logging_csv,
                               scaler=val_dataset.scaler,
                               **meta)

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

    if device_name == 'NVIDIA GeForce RTX 2080':
        workers = 6
    elif device_name == 'NVIDIA RTX A6000':
        workers = 6
    else:
        raise NotImplementedError('Specify the machine this is running on')

    variable = 'mean_temp'

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

    print(f'========================== training autoencoder {variable} ==========================')

    param_dir = os.path.join(training, 'autoencoder')
    pth_ = os.path.join(param_dir, 'pth')
    metadata_ = os.path.join(param_dir, 'training_metadata.json')

    now = datetime.now().strftime('%m%d%H%M')
    chk = os.path.join(param_dir, 'checkpoints', now)
    if os.path.isdir(chk):
        shutil.rmtree(chk)
    os.mkdir(chk)
    logger_csv = os.path.join(chk, 'training_{}.csv'.format(now))

    train_model(chk, pth_, metadata_, target=variable, batch_size=512, learning_rate=0.0001,
                n_workers=workers, logging_csv=logger_csv)
# ========================= EOF ====================================================================
