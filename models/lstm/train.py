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
from models.lstm.sampler import FileBatchSampler
from models.scalers import MinMaxScaler
from prep.build_variable_scaler import build_variable_scaler, load_variable_scaler
from prep.columns_desc import CDR_FEATURES

device_name = None
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f'Using GPU: {device_name}')
else:
    print('CUDA is not available. PyTorch will use the CPU.')

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

WORKERS = 16


def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return default_collate(batch)

def print_lstm_training_overview(train_files, val_files, feature_names, exog_names, num_bands, chunk_size):
    """Print a concise overview of LSTM inputs and simple stats before training."""
    print("\n[LSTM] Training overview")
    print(f"- Train files: {len(train_files)}  Val files: {len(val_files)}")
    print(f"- Sequence length: {chunk_size}")
    print(f"- Feature columns in Parquet (first file): {feature_names}")
    exog_present = [n for n in exog_names if n in feature_names]
    channels = ['lag(' + feature_names[0] + ')'] + exog_present
    print(f"- Input channels (num_bands={num_bands}): {channels}")
    if train_files:
        try:
            first = train_files[0]
            cols = [feature_names[0]] + exog_present
            df = pd.read_parquet(first, columns=cols)
            desc = df.describe().T
            nn = df.notna().mean().rename('non_null_frac')
            print("- Summary (first train file, selected columns):")
            for col in cols:
                if col in desc.index:
                    row = desc.loc[col]
                    nnv = float(nn.get(col, float('nan')))
                    print(f"  {col}: count={int(row['count'])}, non_null={nnv:.2f}, min={row['min']:.3f}, max={row['max']:.3f}")
        except Exception:
            pass


def train_model(dirpath, sequence_data, target_var, training_stations, batch_size=1, learning_rate=0.01, n_workers=1,
                chunk_size=16, strided=False, logging_csv=None, debug_size=0, scaler_path=None, cache_size_files=32,
                prefetch_train=8, prefetch_val=4, hidden_size=256, num_layers=3, expansion_factor=8, use_compile=False):
    """Train the LSTM with [lag(target) + exog] inputs.

    Notes:
    - Exogenous channels are ['rsun'] + NOAA CDR bands when present.
    - Variable scaler is built from graph/train_ids.json (train-only) to avoid leakage and
      is required (no global-all-stations fallback).
    """
    target = f'{target_var}_obs'
    data_dir = os.path.join(sequence_data, target)

    with open(training_stations, 'r') as f:
        train_fids = json.load(f)

    file_map = {'train_files': [], 'val_files': [],
                'train_names': [], 'val_names': []}

    for filename in os.listdir(data_dir):
        if not filename.endswith('.parquet'):
            continue

        station = os.path.splitext(filename)[0]
        full_path = os.path.join(data_dir, filename)

        if station in train_fids:
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
    assert num_features >= 1, "expected at least the target _obs column"
    assert feature_names[0].endswith('_obs'), "first column must be target _obs"

    # determine multivariate input width: lagged target + rsun + CDR when available
    exog_names = ['rsun'] + list(CDR_FEATURES)
    exog_idx = [feature_names.index(n) for n in exog_names if n in feature_names]
    num_bands = 1 + len(exog_idx)

    meta_path = os.path.join(dirpath, 'training_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump({'num_bands': num_bands, 'expansion_factor': 2}, f)

    # Use scaler from graph (single-writer policy); do not rebuild here
    assert scaler_path is not None and os.path.exists(scaler_path), "scaler_json required and must exist (from graph)"
    with open(scaler_path, 'r') as f:
        sp = json.load(f)
    scaler = MinMaxScaler()
    scaler.bias = np.array(sp['bias']).reshape(1, -1)
    scaler.scale = np.array(sp['scale']).reshape(1, -1)

    # print overview before model instantiation
    print_lstm_training_overview(file_map['train_files'], file_map['val_files'], feature_names, exog_names, num_bands, chunk_size)

    train_dataset = LSTMDataset(file_paths=file_map['train_files'], station_names=file_map['train_names'],
                                feature_names=feature_names,
                                sample_dimensions=(chunk_size, num_features),
                                n_workers=n_workers * 2,
                                scaler=scaler,
                                strided=strided,
                                cache_size_files=cache_size_files)

    train_batch_sampler = FileBatchSampler(train_dataset,
                                           batch_size=batch_size,
                                           drop_last=False,
                                           shuffle_files=True,
                                           shuffle_within=True)

    train_dataloader = DataLoader(train_dataset,
                                  batch_sampler=train_batch_sampler,
                                  num_workers=int(n_workers),
                                  collate_fn=custom_collate,
                                  pin_memory=True,
                                  persistent_workers=False,  # persistent_workers=True can accumulate per-worker caches and pinned memory
                                  prefetch_factor=prefetch_train)
    input_dim = num_bands
    if strided:
        print(f'\nTrain dataset: {len(train_dataset)} {chunk_size} x {input_dim} strided samples')
    else:
        print(f'\nTrain dataset: {len(train_dataset)} {chunk_size} x {input_dim} non-overlapping samples')

    print(f'Batch size: {batch_size}, Sequence Length: {chunk_size}, GPU: {device_name}')

    val_dataset = LSTMDataset(file_paths=file_map['val_files'], station_names=file_map['val_names'],
                              feature_names=feature_names, n_workers=n_workers * 2,
                              sample_dimensions=(chunk_size, num_features),
                              scaler=scaler,
                              cache_size_files=cache_size_files)

    val_batch_sampler = FileBatchSampler(val_dataset,
                                         batch_size=batch_size,
                                         drop_last=False,
                                         shuffle_files=False,
                                         shuffle_within=False)
    val_dataloader = DataLoader(val_dataset,
                                batch_sampler=val_batch_sampler,
                                num_workers=max(1, int(n_workers / 2)),
                                collate_fn=custom_collate,
                                pin_memory=True,
                                persistent_workers=False,  # see note above
                                prefetch_factor=prefetch_val)

    model = LSTMPredictor(num_bands=num_bands,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          learning_rate=learning_rate,
                          dropout_rate=0.1,
                          expansion_factor=expansion_factor,  # align with GNN loader when desired
                          log_csv=logging_csv,
                          scaler=train_dataset.scaler)

    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='max-autotune')
        except Exception:
            pass

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

    graph_dir_ = os.path.join(training_, 'graph', f'{target_var_}_obs')
    train_split_ = os.path.join(graph_dir_, 'train_ids.json')

    train_model(chk_, sequences_, target_var_, train_split_,
                batch_size=4096, learning_rate=0.001, n_workers=32,
                chunk_size=12, logging_csv=logger_csv_,
                debug_size=0,
                scaler_path=scaler_, cache_size_files=128,
                prefetch_train=12, prefetch_val=12, hidden_size=256,
                num_layers=2, expansion_factor=2,
                use_compile=False)

# ========================= EOF ==============================================================================
