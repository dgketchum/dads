import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import resource
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from models.lstm.lstm import LSTMPredictor
from models.lstm.dataset import LSTMDataset
from models.scalers import MinMaxScaler
from prep.build_variable_scaler import build_variable_scaler, load_variable_scaler

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

WORKERS = 16


def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return default_collate(batch)


def train_model(dirpath, sequence_data, target_var, train_features, now_str, batch_size=1, learning_rate=0.01,
                n_workers=1, chunk_size=16, strided=False, logging_csv=None, debug_size=0, scaler_path=None):
    """"""
    target = f'{target_var}_obs'
    data_dir = os.path.join(sequence_data, target)

    with open(train_features, 'r') as f:
        train_feats = json.load(f)

    file_map = {'train_files': [], 'val_files': [],
                'train_names': [], 'val_names': []}

    for filename in os.listdir(data_dir):
        if not filename.endswith('.parquet'):
            continue

        station = os.path.splitext(filename)[0]
        full_path = os.path.join(data_dir, filename)

        if station in train_feats:
            file_map['train_files'].append(full_path)
            file_map['train_names'].append(station)
        else:
            file_map['val_files'].append(full_path)
            file_map['val_names'].append(station)

        if 0 < debug_size <= len(file_map['train_files']) + len(file_map['val_files']):
            break

    assert len(set(file_map['train_files']).intersection(set(file_map['val_files']))) == 0

    if not file_map['train_files'] and not file_map['val_files']:
        print(f"No data files found in {data_dir}")
        return

    first_file = file_map['train_files'][0] if file_map['train_files'] else file_map['val_files'][0]
    feature_names = pd.read_parquet(first_file).columns.tolist()
    num_features = len(feature_names)
    assert num_features >= 2, "expected [obs, gm, ...] feature layout"
    assert feature_names[0].endswith('_obs'), "first column must be target _obs"
    assert not feature_names[1].endswith('_obs'), "second column must be gridded match, not _obs"

    meta_path = os.path.join(dirpath, 'training_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump({'num_bands': num_features - 2, 'expansion_factor': 2}, f)

    if not scaler_path:
        scaler, scaler_path, _ = load_variable_scaler(sequence_data, target_var, feature_names)
    else:
        with open(scaler_path, 'r') as f:
            scaler_params = json.load(f)
        scaler = MinMaxScaler()
        scaler.bias = np.array(scaler_params['bias']).reshape(1, -1)
        scaler.scale = np.array(scaler_params['scale']).reshape(1, -1)
        if 'feature_names' in scaler_params:
            assert scaler_params['feature_names'] == feature_names, "scaler feature_names mismatch"

    train_dataset = LSTMDataset(file_paths=file_map['train_files'], station_names=file_map['train_names'],
                                feature_names=feature_names,
                                sample_dimensions=(chunk_size, num_features),
                                n_workers=n_workers,
                                scaler=scaler,
                                strided=strided)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=int(n_workers / 2),
                                  collate_fn=custom_collate,
                                  pin_memory=True)
    if strided:
        print(f'\nTrain dataset: {len(train_dataset)} {chunk_size} x {num_features} strided samples')
    else:
        print(f'\nTrain dataset: {len(train_dataset)} {chunk_size} x {num_features} non-overlapping samples')

    print(f'Batch size: {batch_size}, Sequence Length: {chunk_size}, GPU: {device_name}')

    val_dataset = LSTMDataset(file_paths=file_map['val_files'], station_names=file_map['val_names'],
                              feature_names=feature_names,
                              sample_dimensions=(chunk_size, num_features),
                              scaler=scaler)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=int(n_workers / 2),
                                collate_fn=custom_collate,
                                pin_memory=True)

    model = LSTMPredictor(num_bands=num_features - 2,
                          learning_rate=learning_rate,
                          dropout_rate=0.1,
                          expansion_factor=2,  # align with GNN loader
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

    trainer = pl.Trainer(max_epochs=1000,
                         callbacks=[checkpoint_callback, early_stop_callback],
                         accelerator='gpu', devices=1,
                         precision='16-mixed',
                         gradient_clip_val=1.0)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':

    d = '/home/dgketchum/data/IrrigationGIS/dads'

    target_var_ = 'tmax'

    training_ = '/data/ssd2/dads/training'
    sequences_ = os.path.join(training_, 'parquet')

    now_ = datetime.now().strftime('%Y%m%d_%H%M')
    chk_ = os.path.join(training_, 'lstm', 'checkpoints', f'{target_var_}_{now_}')
    os.makedirs(chk_, exist_ok=True)
    logger_csv_ = os.path.join(chk_, 'training_{}.csv'.format(now_))

    scaler_ = os.path.join(training_, 'scalers', f'{target_var_}.json')

    train_edges_ = os.path.join(training_, 'graph', 'train_edge_attr.json')

    train_model(chk_, sequences_, target_var_, train_edges_, now_,
                scaler_path=scaler_,
                batch_size=1056,
                learning_rate=0.001,
                n_workers=16,
                logging_csv=logger_csv_,
                chunk_size=12,
                debug_size=0)

# ========================= EOF ==============================================================================
