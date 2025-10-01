import json
import os
import resource
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from models.autoencoder.dataset import WeatherDataset
from models.autoencoder.weather_encoder import WeatherAutoencoder

from prep.columns_desc import GEO_FEATURES
import pandas as pd

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


def train_model(dirpath, parquet, target_var, columns, chunk_size, meta_path,
                batch_size=64, learning_rate=0.001, n_workers=1, device='gpu',
                logging_csv=None):
    """"""

    target_idx = [columns.index(target_var)]

    # simple split by presence in provided mapping file removed; use filename hashing for split
    # If a station list exists elsewhere, integrate here.
    train_feats = {}

    file_map = {'train': [], 'val': []}
    for filename in os.listdir(parquet):
        if not filename.endswith('.parquet'):
            continue
        # naive split: alternate files between train/val
        (file_map['train'] if len(file_map['train']) <= len(file_map['val'])
         else file_map['val']).append(os.path.join(parquet, filename))

    sample = file_map['train'][0] if file_map['train'] else (file_map['val'][0] if file_map['val'] else None)
    if sample is None:
        raise FileNotFoundError(f'No parquet files found in {parquet}')

    actual_cols = pd.read_parquet(sample).columns.tolist()
    missing = [c for c in columns if c not in actual_cols]
    if missing:
        raise ValueError(f'Missing expected columns in parquet: {missing}')

    selected_indices = [actual_cols.index(c) for c in columns]

    meta = {
        'chunk_size': chunk_size,
        'data_columns': columns,
        'column_order': columns,
        'selected_indices': selected_indices,
        'expected_width': len(actual_cols),
        'actual_columns': actual_cols,
        'target_var': target_var,
        'input_dim': len(columns),
        'output_dim': len(target_idx),
        'latent_size': 64,
        'hidden_size': 32,
        'dropout': 0.1,
        'learning_rate': learning_rate,
        'sequence_length': chunk_size,
        'margin': 1.0,
        'encoder_heads': 1,
        'encoder_layers': 2,
        'decoder_heads': 1,
        'decoder_layers': 2,
    }
    with open(meta_path, 'w') as fp:
        json.dump(meta, fp, indent=4)

    # file_map['train'] = file_map['train'][:1000]
    # file_map['val'] = file_map['val'][:500]

    train_dataset = WeatherDataset(file_paths=file_map['train'],
                                   expected_width=len(actual_cols),
                                   col_indices=len(selected_indices),
                                   chunk_size=chunk_size,
                                   target_indices=target_idx,
                                   expected_columns=actual_cols,
                                   selected_indices=selected_indices,
                                   num_workers=12)

    if logging_csv:
        train_dataset.save_scaler(os.path.join(dirpath, 'scaler.json'))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers,
                                  persistent_workers=True, pin_memory=True, prefetch_factor=2,
                                  collate_fn=lambda batch: [x for x in batch if x is not None])

    val_dataset = WeatherDataset(file_paths=file_map['val'],
                                 expected_width=len(actual_cols),
                                 col_indices=len(selected_indices),
                                 chunk_size=chunk_size,
                                 target_indices=target_idx,
                                 expected_columns=actual_cols,
                                 selected_indices=selected_indices,
                                 num_workers=12)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=n_workers,
                                persistent_workers=True, pin_memory=True, prefetch_factor=2,
                                collate_fn=lambda batch: [x for x in batch if x is not None])

    model = WeatherAutoencoder(input_dim=len(columns),
                               output_dim=len(target_idx),
                               latent_size=64,
                               hidden_size=32,
                               dropout=0.1,
                               learning_rate=learning_rate,
                               log_csv=logging_csv,
                               scaler=val_dataset.scaler,
                               sequence_length=chunk_size)

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
                         accelerator=device, devices=1)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    variable_ = 'tmax'
    target_var_ = f'{variable_}_obs'

    if device_name == 'NVIDIA GeForce RTX 2080':
        workers = 4
    elif device_name == 'NVIDIA RTX A6000':
        workers = 8
    else:
        raise NotImplementedError('Specify the machine this is running on')

    zoran = '/data/ssd2/dads/training'
    nvm = '/media/nvm/training'
    if os.path.exists(zoran):
        print('Modeling with data from Zoran')
        training = zoran
    elif os.path.exists(nvm):
        print('Modeling with data from NVM drive')
        training = nvm
    else:
        raise NotImplementedError

    param_dir = os.path.join(training, 'autoencoder')
    parq_ = os.path.join(training, 'parquet', target_var_)

    # graph
    now = datetime.now().strftime('%m%d%H%M')
    chk = os.path.join(param_dir, 'checkpoints', now)
    metadata_ = os.path.join(chk, 'training_metadata.json')

    os.mkdir(chk)
    print(f'mkdir: {chk}')
    logger_csv = os.path.join(chk, 'training_{}.csv'.format(now))
    # logger_csv = None
    device_ = 'gpu'

    cols = [target_var_] + GEO_FEATURES

    train_model(chk, parq_, target_var=target_var_, columns=cols, chunk_size=365,
                batch_size=128, learning_rate=0.001, meta_path=metadata_,
                n_workers=workers, logging_csv=logger_csv, device=device_)
# ========================= EOF ====================================================================
