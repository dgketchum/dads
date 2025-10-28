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
from models.scalers import MinMaxScaler

from prep.columns_desc import GEO_FEATURES, RS_MISS_FEATURES
import pandas as pd
import numpy as np

device_name = None
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f'Using GPU: {device_name}')
else:
    print('CUDA is not available. PyTorch will use the CPU.')

torch.set_float32_matmul_precision('medium')
torch.cuda.get_device_name(torch.cuda.current_device())  # likely error if CUDA not available
torch.backends.cudnn.benchmark = True

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def train_model(dirpath, parquet, target_var, columns, chunk_size, meta_path,
                batch_size=64, learning_rate=0.001, n_workers=1, device='gpu',
                logging_csv=None, scaler_json=None):
    """Train the autoencoder on yearly 365-day chunks with a unified schema.

    Intent
    - Inputs: columns = [target_var] + GEO_FEATURES + RS_MISS_FEATURES.
      RS_MISS_FEATURES are binary flags (1=missing) so the encoder is mask-aware
      of RS availability over time and across stations.
    - Strict schema: all station Parquets must have identical columns/order
      (enforced via metadata.expected_columns). Missing RS values remain NaN in
      Parquet but their flags indicate absence; the dataset scales and nan-to-num
      is applied inside the model to stabilize training.
    - Outputs: latent embeddings (per-station, averaged over available yearly chunks)
      are later consumed by the DADS GNN.
    - Scaling uses a MinMax scaler fit on graph train_ids only (no leakage).
    - By default, enables `zero_target_in_encoder=True` so embeddings depend on
      exogenous inputs rather than shortcutting via the target channel.
    """

    target_idx = [columns.index(target_var)]

    def print_autoencoder_training_overview(parquet_dir, cols_cfg, actual_cols, selected_idx):
        """Print a concise overview of AE inputs and simple stats before training."""
        files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
        print("\n[AE] Training overview")
        print(f"- Parquet files: {len(files)}  Sequence length: {chunk_size}")
        print(f"- Configured columns ({len(cols_cfg)}): {cols_cfg}")
        exog = [c for c in cols_cfg if c != target_var and c in GEO_FEATURES]
        print(f"- Exog (GEO_FEATURES present): {exog}")
        # summary from first file over target+exog subset
        try:
            sample_file = os.path.join(parquet_dir, files[0])
            subset = [target_var] + exog
            df = pd.read_parquet(sample_file, columns=[c for c in subset if c in actual_cols])
            desc = df.describe().T
            nn = df.notna().mean().rename('non_null_frac')
            print("- Summary (first file, target+exog):")
            for col in df.columns.tolist():
                row = desc.loc[col]
                nnv = float(nn.get(col, float('nan')))
                print(f"  {col}: count={int(row['count'])}, non_null={nnv:.2f}, min={row['min']:.3f}, max={row['max']:.3f}")
        except Exception:
            pass

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

    # graph-dependent split (overrides naive) using prepared graph station IDs
    parquet_root_ = os.path.dirname(parquet)
    training_root_ = os.path.dirname(parquet_root_)
    graph_dir_ = os.path.join(training_root_, 'graph', target_var)
    train_ids_path_ = os.path.join(graph_dir_, 'train_ids.json')
    val_ids_path_ = os.path.join(graph_dir_, 'val_ids.json')
    if os.path.exists(train_ids_path_) and os.path.exists(val_ids_path_):
        try:
            with open(train_ids_path_, 'r') as fp:
                tr_ids = set(str(s) for s in json.load(fp))
            with open(val_ids_path_, 'r') as fp:
                va_ids = set(str(s) for s in json.load(fp))
            all_paths_ = [os.path.join(parquet, f) for f in os.listdir(parquet) if f.endswith('.parquet')]
            id_to_path_ = {os.path.splitext(os.path.basename(p))[0]: p for p in all_paths_}
            file_map = {
                'train': [id_to_path_[s] for s in tr_ids if s in id_to_path_],
                'val': [id_to_path_[s] for s in va_ids if s in id_to_path_],
            }
        except Exception:
            pass  # likely error if split files malformed

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

    assert scaler_json is not None and os.path.exists(scaler_json), "scaler_json required and must exist (from graph)"
    with open(scaler_json, 'r') as f:
        sp = json.load(f)
    scaler_obj = MinMaxScaler()
    scaler_obj.bias = np.array(sp['bias']).reshape(1, -1)
    scaler_obj.scale = np.array(sp['scale']).reshape(1, -1)

    train_dataset = WeatherDataset(file_paths=file_map['train'],
                                   expected_width=len(actual_cols),
                                   col_indices=len(selected_indices),
                                   chunk_size=chunk_size,
                                   target_indices=target_idx,
                                   expected_columns=actual_cols,
                                   selected_indices=selected_indices,
                                   num_workers=12,
                                   scaler=scaler_obj)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers,
                                  persistent_workers=False, pin_memory=True, prefetch_factor=2,
                                  collate_fn=lambda batch: [x for x in batch if x is not None])

    val_dataset = WeatherDataset(file_paths=file_map['val'],
                                 expected_width=len(actual_cols),
                                 col_indices=len(selected_indices),
                                 chunk_size=chunk_size,
                                 target_indices=target_idx,
                                 expected_columns=actual_cols,
                                 selected_indices=selected_indices,
                                 num_workers=12,
                                 scaler=scaler_obj)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=n_workers,
                                persistent_workers=False, pin_memory=True, prefetch_factor=2,
                                collate_fn=lambda batch: [x for x in batch if x is not None])

    # print overview before model instantiation
    print_autoencoder_training_overview(parquet, columns, actual_cols, selected_indices)

    model = WeatherAutoencoder(input_dim=len(columns),
                               output_dim=len(target_idx),
                               latent_size=64,
                               hidden_size=32,
                               dropout=0.1,
                               learning_rate=learning_rate,
                               log_csv=logging_csv,
                               scaler=val_dataset.scaler,
                               sequence_length=chunk_size,
                               zero_target_in_encoder=True,
                               target_input_idx=0)

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
                         accelerator=device, devices=1,
                         precision='16-mixed',
                         gradient_clip_val=1.0)
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

    os.makedirs(chk, exist_ok=True)
    print(f'mkdir: {chk}')
    logger_csv = os.path.join(chk, 'training_{}.csv'.format(now))
    # logger_csv = None
    device_ = 'gpu'

    # Include RS missingness flags to make embeddings mask-aware of RS availability
    cols = [target_var_] + GEO_FEATURES + RS_MISS_FEATURES

    train_model(chk, parq_, target_var=target_var_, columns=cols, chunk_size=365,
                batch_size=128, learning_rate=0.001, meta_path=metadata_,
                n_workers=workers, logging_csv=logger_csv, device=device_,
                scaler_json=os.path.join(training, 'scalers', f"{variable_}.json"))
# ========================= EOF ====================================================================
