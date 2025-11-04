import json
import os
import resource
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from models.autoencoder.dataset import WeatherDataset, WeatherIterableDataset
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
                logging_csv=None, scaler_json=None, window_stride=None,
                debug_n_samples=None, debug_max_files=None,
                debug_max_train_files=None, debug_max_val_files=None,
                use_compile=False, max_epochs=None, autotune_warmup=False):
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

    # Debug: optionally limit number of files to speed up dataset construction
    if (debug_max_files is not None) or (debug_max_train_files is not None) or (debug_max_val_files is not None):
        # prefer explicit train/val limits when provided; fall back to single value heuristic
        if debug_max_train_files is not None:
            keep_tr = max(1, int(debug_max_train_files))
        elif debug_max_files is not None:
            keep_tr = max(1, int(debug_max_files))
        else:
            keep_tr = len(file_map['train'])

        if debug_max_val_files is not None:
            keep_va = max(1, int(debug_max_val_files))
        elif debug_max_files is not None:
            keep_va = max(1, int(max(1, debug_max_files // 4)))
        else:
            keep_va = len(file_map['val'])

        file_map['train'] = file_map['train'][:keep_tr]
        file_map['val'] = file_map['val'][:keep_va]
        print(f"Debug mode: limiting files -> train: {len(file_map['train'])}, val: {len(file_map['val'])}")

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
        'window_stride': int(window_stride) if window_stride is not None else int(chunk_size),
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

    # Stream windows on-the-fly to reduce memory footprint
    train_dataset = WeatherIterableDataset(file_paths=file_map['train'],
                                           expected_width=len(actual_cols),
                                           col_indices=len(selected_indices),
                                           chunk_size=chunk_size,
                                           target_indices=target_idx,
                                           expected_columns=actual_cols,
                                           selected_indices=selected_indices,
                                           num_workers=n_workers,
                                           scaler=scaler_obj,
                                           window_stride=int(window_stride) if window_stride is not None else int(chunk_size),
                                           triplet_sampling=False,
                                           max_samples=int(debug_n_samples) if debug_n_samples is not None else None,
                                           max_files=(int(debug_max_train_files) if debug_max_train_files is not None else None),
                                           split_name='train',
                                           shuffle_files=True)

    def _collate(batch):
        batch = [b for b in batch if b is not None]
        x_list, y_list, m_list, p_list, n_list = zip(*batch)
        p_list = [torch.full_like(xi, torch.nan) if pi is None else pi for xi, pi in zip(x_list, p_list)]
        n_list = [torch.full_like(xi, torch.nan) if ni is None else ni for xi, ni in zip(x_list, n_list)]
        x = torch.stack(x_list, dim=0)
        y = torch.stack(y_list, dim=0)
        m = torch.stack(m_list, dim=0)
        p = torch.stack(p_list, dim=0)
        n = torch.stack(n_list, dim=0)
        return x, y, m, p, n

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=n_workers,
                                  persistent_workers=False, pin_memory=True, prefetch_factor=2,
                                  collate_fn=_collate)

    val_dataset = WeatherIterableDataset(file_paths=file_map['val'],
                                         expected_width=len(actual_cols),
                                         col_indices=len(selected_indices),
                                         chunk_size=chunk_size,
                                         target_indices=target_idx,
                                         expected_columns=actual_cols,
                                         selected_indices=selected_indices,
                                         num_workers=n_workers,
                                         scaler=scaler_obj,
                                         window_stride=int(window_stride) if window_stride is not None else int(chunk_size),
                                         triplet_sampling=False,
                                         max_samples=(
                                             max(1, int(debug_n_samples // 4)) if isinstance(debug_n_samples, int) and debug_n_samples > 0 else None
                                         ),
                                         max_files=(int(debug_max_val_files) if debug_max_val_files is not None else None),
                                         split_name='val',
                                         shuffle_files=False)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=n_workers,
                                persistent_workers=False, pin_memory=True, prefetch_factor=2,
                                collate_fn=_collate)

    # print overview before model instantiation
    print_autoencoder_training_overview(parquet, columns, actual_cols, selected_indices)

    # Prepare scaler subset to match selected input feature order for on-device scaling
    bias_sel = scaler_obj.bias[:, selected_indices]
    scale_sel = scaler_obj.scale[:, selected_indices]

    model = WeatherAutoencoder(input_dim=len(columns),
                               output_dim=len(target_idx),
                               latent_size=64,
                               hidden_size=32,
                               dropout=0.1,
                               learning_rate=learning_rate,
                               log_csv=logging_csv,
                               scaler=val_dataset.scaler,
                               scaler_bias=bias_sel.reshape(-1).tolist(),
                               scaler_scale=scale_sel.reshape(-1).tolist(),
                               sequence_length=chunk_size,
                               zero_target_in_encoder=True,
                               target_input_idx=0,
                               )

    # Try to compile heavy submodules only to avoid capturing Lightning internals.
    _short_debug = any(v is not None for v in (debug_n_samples, debug_max_files))
    if use_compile and not _short_debug:
        try:
            # Best-effort disable cudagraphs in Inductor to improve stability
            try:
                import torch as _t
                _t._inductor.config.triton.cudagraphs = False
            except Exception:
                pass
            # Optionally emphasize autotune during warmup runs; leave default otherwise
            try:
                import torch as _t
                if autotune_warmup:
                    _t._inductor.config.triton.autotune = True
            except Exception:
                pass
            model.encoder = torch.compile(model.encoder, mode='max-autotune')
            model.decoder = torch.compile(model.decoder, mode='max-autotune')
            model.attn_pool = torch.compile(model.attn_pool, mode='max-autotune')
            print('Enabled torch.compile for encoder/decoder/attn_pool.')
        except Exception as e:
            print(f'torch.compile submodules unavailable or failed: {e}')
    elif _short_debug:
        print('Debug mode: skipping torch.compile to avoid potential cudagraph/dynamo issues.')

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="best_model",
        dirpath=dirpath,
        save_top_k=1,
        mode="min",
        save_last=True,
        check_on_train_epoch_end=False,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=100,
        verbose=False,
        mode="min",
        check_finite=True,
        strict=False,
    )

    # Trainer configuration
    trainer_kwargs = dict(
        max_epochs=(int(max_epochs) if max_epochs is not None else 1000),
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator=device,
        devices=1,
        precision='16-mixed',
        # Disable PL gradient clipping to avoid fused AdamW + AMP incompatibility
        gradient_clip_val=0.0,
        accumulate_grad_batches=2,
        num_sanity_val_steps=0,
    )

    # If short debug (sample-capped), drastically shorten the run
    if _short_debug:
        trainer_kwargs.update(dict(
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=1,
        ))
        print('Debug mode: using max_epochs=1, limit_train_batches=2, limit_val_batches=1')

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_dataloader, val_dataloader)


def debug_sample(dirpath, parquet, target_var, columns, chunk_size, meta_path,
                 n_samples=2048, n_files=4, n_train_files=None, n_val_files=None, **kwargs):
    """Run a very small debug training to validate the loop quickly.

    Parameters
    - n_samples: cap the number of training samples (windows) loaded.
    - n_files: limit the number of parquet files considered to speed up dataset build.
    Other training kwargs like batch_size/learning_rate can be passed via **kwargs.
    """
    return train_model(dirpath, parquet, target_var, columns, chunk_size, meta_path,
                       debug_n_samples=int(n_samples) if n_samples is not None else None,
                       debug_max_files=int(n_files) if n_files is not None else None,
                       debug_max_train_files=(int(n_train_files) if n_train_files is not None else None),
                       debug_max_val_files=(int(n_val_files) if n_val_files is not None else None),
                       **kwargs)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    variable_ = 'tmax'
    target_var_ = f'{variable_}_obs'

    if device_name == 'NVIDIA GeForce RTX 2080':
        workers = 4
    elif device_name == 'NVIDIA RTX A6000':
        workers = 16
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

    # Configure persistent compiler caches to reuse autotune across runs
    cache_root = os.path.join(training, 'cache')
    inductor_cache = os.path.join(cache_root, 'inductor')
    triton_cache = os.path.join(cache_root, 'triton')
    os.makedirs(inductor_cache, exist_ok=True)
    os.makedirs(triton_cache, exist_ok=True)
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', inductor_cache)
    os.environ.setdefault('TRITON_CACHE_DIR', triton_cache)
    print(f"Inductor cache dir: {os.environ.get('TORCHINDUCTOR_CACHE_DIR')}")
    print(f"Triton cache dir:   {os.environ.get('TRITON_CACHE_DIR')}")

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

    cols = [target_var_] + GEO_FEATURES + RS_MISS_FEATURES

    seq_len_ = 120
    stride_ = 30
    
    # Keep batch size consistent across debug and full runs to maximize
    # kernel/cache reuse from autotune.
    batch_size_ = 2048
    
    # Full run (no debug file limits)
    train_model(chk, parq_, target_var=target_var_, columns=cols, chunk_size=seq_len_,
                batch_size=batch_size_, learning_rate=0.001, meta_path=metadata_,
                n_workers=workers, logging_csv=logger_csv, device=device_,
                scaler_json=os.path.join(training, 'scalers', f"{variable_}.json"),
                window_stride=stride_,
                use_compile=True, autotune_warmup=False)

    # Debug sample: same setup, limit training files to 1000
    # debug_sample(dirpath=chk, parquet=parq_, target_var=target_var_, columns=cols, chunk_size=seq_len_,
    #              n_samples=None, n_files=None, n_val_files=500, n_train_files=1000,
    #              batch_size=batch_size_, learning_rate=0.001, meta_path=metadata_,
    #              n_workers=workers, logging_csv=logger_csv, device=device_,
    #              scaler_json=os.path.join(training, 'scalers', f"{variable_}.json"),
    #              window_stride=stride_,
    #              use_compile=True, max_epochs=5, autotune_warmup=True)
# ========================= EOF ====================================================================
