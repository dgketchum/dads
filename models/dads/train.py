import json
import os
from datetime import datetime

import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.loader import DataLoader
import torch  # set matmul precision for Tensor Cores
import torch.multiprocessing as mp  # control sharing strategy for large batches

from models.dads.dads_gnn import DadsMetGNN
from models.dads.dataset import DadsDataset
from prep.build_variable_scaler import load_variable_scaler


def train_model(dirpath, parquet_dir, embeddings, edge_info, scaler_json, nodes=5, batch_size=1,
                dropout=0.2, learning_rate=0.01, n_workers=1, logging_csv=None, device='gpu',
                n_samples=None, debug=False, windows_per_station=None,
                tcn_out_dim=256, tcn_channels=128, tcn_dilations=(1, 2, 4, 8), tcn_kernel=3, tcn_dropout=0.1):
    """Train the DADS GNN with prebuilt embeddings and edges, computing neighbor contexts via TCN.

    Prints a brief overview of inputs (exog, embeddings, edge attributes, nodes) before
    model instantiation.
    """

    # Sanity check: graph artifacts must exist (three-tier subgraph expected)
    if not os.path.isdir(edge_info):
        raise FileNotFoundError('edge_info directory not found: {}'.format(edge_info))
    _req = [
        'train_edge_index.json', 'train_edge_attr.json',
        'val_edge_index.json', 'val_edge_attr.json',
    ]
    for _f in _req:
        _p = os.path.join(edge_info, _f)
        if not os.path.exists(_p):
            raise FileNotFoundError('required graph file missing: {}'.format(_p))

    # Debug/sample handling: subset stations to a small sample for quick checks
    if n_samples is not None:
        sample = (int(n_samples), int(n_samples))
    else:
        sample = (None, None)

    # Minimal sequence metadata
    meta = {'chunk_size': 12}

    # Station-based split from graph prep
    all_files = {os.path.splitext(f)[0]: os.path.join(parquet_dir, f)
                 for f in os.listdir(parquet_dir) if f.endswith('.parquet')}
    train_edges = os.path.join(edge_info, 'train_edge_index.json')
    train_attr = os.path.join(edge_info, 'train_edge_attr.json')
    with open(train_attr, 'r') as f:
        train_attr_map = json.load(f)
    val_attr = os.path.join(edge_info, 'val_edge_attr.json')
    with open(val_attr, 'r') as f:
        val_attr_map = json.load(f)
    train_stations = set(train_attr_map.keys())
    val_stations = set(val_attr_map.keys()) - train_stations

    t_files = [all_files[s] for s in train_stations if s in all_files]
    v_files = [all_files[s] for s in val_stations if s in all_files]
    if debug and n_samples is not None:
        # Limit actual file lists as an extra safeguard in debug
        t_files = t_files[:n_samples]
        v_files = v_files[:max(1, min(n_samples, len(v_files)))]

    # Ensure variable-specific scaler is built from train-only stations to avoid leakage
    var_name = os.path.basename(parquet_dir).replace('_obs', '')
    parquet_root = os.path.dirname(parquet_dir)
    scaler_obj_loaded, scaler_json_path, _ = load_variable_scaler(
        parquet_root, var_name, station_ids=sorted(list(train_stations)), rebuild=True
    )

    train_dataset = DadsDataset(nodes, t_files, meta, embeddings, train_edges, train_attr,
                                scaler=scaler_json_path, sample=sample[0], lstm_workers=n_workers,
                                normalize_keys=train_stations,
                                windows_per_station=(64 if debug and n_samples is not None else windows_per_station),
                                neighbor_file_map=all_files)

    def _collate(batch):
        import torch
        from torch_geometric.data import Batch
        # graph, y, neighbor_seq, neighbor_mask, target_seq
        graphs, y_list, seq_list, mask_list, tgt_seq_list = zip(*batch)
        b_graph = Batch.from_data_list(list(graphs))
        y = torch.stack(list(y_list), dim=0)
        neighbor_seq = torch.stack(seq_list, dim=0)  # [B, n_nodes, T, C]
        neighbor_mask = torch.stack(mask_list, dim=0)  # [B, n_nodes]
        target_seq = torch.stack(tgt_seq_list, dim=0)  # [B, T, C]
        return b_graph, y, neighbor_seq, neighbor_mask, target_seq

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        # Prefetch fewer batches to reduce pressure on pin-memory thread
        prefetch_factor=(2 if n_workers > 0 else None),
        collate_fn=_collate,
        persistent_workers=bool(n_workers > 0),
        # Disable pin_memory to avoid 'received 0 items of ancdata' on large batches
        pin_memory=False,
    )

    val_edges = os.path.join(edge_info, 'val_edge_index.json')
    val_attr = os.path.join(edge_info, 'val_edge_attr.json')

    val_dataset = DadsDataset(nodes, v_files, meta, embeddings, val_edges, val_attr,
                              scaler=scaler_json_path, sample=sample[1], lstm_workers=n_workers,
                              normalize_keys=train_stations,
                              windows_per_station=(16 if debug and n_samples is not None else windows_per_station),
                              neighbor_file_map=all_files)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        prefetch_factor=(2 if n_workers > 0 else None),
        collate_fn=_collate,
        persistent_workers=bool(n_workers > 0),
        pin_memory=False,
    )

    def print_dads_training_overview(tr_ds, va_ds, n_nodes):
        print("\n[DADS] Training overview")
        print(f"- Train stations: {len(tr_ds.file_paths)}  Val stations: {len(va_ds.file_paths)}")
        print(f"- Nodes per sample: {n_nodes}")
        print(f"- Embedding dim: {tr_ds.emb_dim}  Exog dim: {tr_ds.exog_dim}")
        has_bear = bool(getattr(tr_ds, '_bearing_map', None))
        has_dist = bool(getattr(tr_ds, '_distance_map', None))
        print(f"- Edge attr dim: {tr_ds.edge_dim}  (bearing={'on' if has_bear else 'off'}, distance={'on' if has_dist else 'off'})")
        print(f"- Target exog columns: {tr_ds.exog_cols}")
        # simple distance summary (first two targets)
        if has_dist and isinstance(tr_ds._distance_map, dict):
            try:
                ks = list(tr_ds._distance_map.keys())[:2]
                vals = []
                for k in ks:
                    vals.extend(list(tr_ds._distance_map[k].values()))
                if vals:
                    arr = np.asarray(vals, dtype=float)
                    print(f"- Distance (km) summary over sample edges: min={arr.min():.1f}, median={np.median(arr):.1f}, max={arr.max():.1f}")
            except Exception:
                pass

    # print overview before model instantiation
    print_dads_training_overview(train_dataset, val_dataset, nodes)

    column_indices = train_dataset.column_indices
    scaler_obj = train_dataset.scaler
    exog_dim = getattr(train_dataset, 'exog_dim', 0)
    emb_dim = getattr(train_dataset, 'emb_dim', None)
    edge_dim = int(getattr(train_dataset, 'edge_dim', next(iter(train_dataset.edge_attr.values())).shape[-1]))
    tcn_in_channels = int(train_dataset.seq_in_channels)
    model = DadsMetGNN(output_dim=1, n_nodes=nodes, hidden_dim=256,
                       edge_attr_dim=edge_dim, dropout=dropout, learning_rate=learning_rate,
                       log_csv=logging_csv, use_target_exog_branch=True,
                       emb_dim=emb_dim, exog_dim=exog_dim, scaler=scaler_obj, column_indices=column_indices,
                       tcn_in_channels=tcn_in_channels, tcn_out_dim=tcn_out_dim,
                       tcn_channels=tcn_channels, tcn_dilations=tcn_dilations,
                       tcn_kernel=tcn_kernel, tcn_dropout=tcn_dropout)
    callbacks = []
    max_epochs = 100
    if not debug:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            filename='best_model',
            dirpath=dirpath,
            save_top_k=1,
            mode='min'
        )
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=100,
            verbose=False,
            mode='min',
            check_finite=True,
        )
        callbacks = [checkpoint_callback, early_stop_callback]
    else:
        max_epochs = 1  # quick functional check

    trainer = pl.Trainer(max_epochs=max_epochs,
                         callbacks=callbacks,
                         accelerator=device, devices=1,
                         # Start in full precision for numerical stability
                         precision='32-true',
                         # Clip gradients to avoid explosive updates early in training
                         gradient_clip_val=1.0,
                         enable_checkpointing=not debug)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':

    target_var = 'tmax'

    d = '/home/dgketchum/data/IrrigationGIS/dads'
    training = '/data/ssd2/dads/training'

    # Improve Tensor Core matmul stability/perf under mixed precision
    torch.set_float32_matmul_precision('medium')
    # Avoid FD passing issues in DataLoader on large, frequent batch transfers
    mp.set_sharing_strategy('file_system')
    print('========================== modeling {} =========================='.format(target_var))

    # data
    parquet_dir_ = os.path.join(training, 'parquet', f'{target_var}_obs')

    # graph (per target)
    # edges = os.path.join(training, 'graph', f'{target_var}_obs')
    edges = os.path.join(training, 'subgraph', f'{target_var}_obs')

    # climate embedding
    encoder_dir = os.path.join(training, 'autoencoder', 'checkpoints', '10211323')

    # Toggle explicit debug mode here
    DEBUG = False
    N_SAMPLES = 1500  # number of stations to sample in debug
    WINDOWS_PER_STATION = 256  # cap indexed windows per station (None uses all)

    if DEBUG:
        chk = None
        logger_csv = None
    else:
        now = datetime.now().strftime('%m%d%H%M')
        chk = os.path.join(training, 'gnn', 'checkpoints', now)
        os.makedirs(chk, exist_ok=True)
        logger_csv = os.path.join(chk, 'training_{}.csv'.format(now))

    workers = 40
    device_ = 'gpu'

    scaler_json_ = os.path.join(training, 'scalers', f'{target_var}.json')

    train_model(chk, parquet_dir_, encoder_dir, edges, scaler_json_, batch_size=512, nodes=10, dropout=0.1,
                learning_rate=0.001, n_workers=workers, logging_csv=logger_csv, device=device_,
                n_samples=(N_SAMPLES if DEBUG else None), debug=DEBUG,
                windows_per_station=WINDOWS_PER_STATION)
# ========================= EOF ====================================================================
