import json
import os
from datetime import datetime

import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.loader import DataLoader

from models.dads.dads_gnn import DadsMetGNN
from models.dads.dataset import DadsDataset
from prep.build_variable_scaler import load_variable_scaler


def train_model(dirpath, parquet_dir, lstm_model, embeddings, edge_info, nodes=5, batch_size=1,
                dropout=0.2, learning_rate=0.01, n_workers=1, logging_csv=None, device='gpu', sample=None,
                node_ctx_dir=None):
    """Train the DADS GNN with prebuilt embeddings, edges, and node contexts.

    Prints a brief overview of inputs (exog, embeddings, edge attributes, nodes) before
    model instantiation.
    """

    if sample is None:
        sample = None, None

    metadata_ = os.path.join(lstm_model, 'training_metadata.json')
    with open(metadata_, 'r') as f:
        meta = json.load(f)

    meta['model'] = 'lstm'
    # Prefer variable-specific scaler under training/scalers/{var}.json
    var_dir = os.path.basename(parquet_dir)
    var_name = var_dir.replace('_obs', '')
    parquet_root = os.path.dirname(parquet_dir)
    _scaler_obj, scaler, _ = load_variable_scaler(parquet_root, var_name, split_ids_path=os.path.join(edge_info, 'train_ids.json'))

    # Require node contexts
    assert node_ctx_dir is not None and os.path.isdir(node_ctx_dir), "node_ctx_dir required and must exist"

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

    t_files = [all_files[s] for s in train_stations if s in all_files][:100]
    v_files = [all_files[s] for s in val_stations if s in all_files][:100]

    train_dataset = DadsDataset(nodes, t_files, meta, embeddings, train_edges, train_attr,
                                scaler=scaler, sample=sample[0], node_ctx_dir=node_ctx_dir,
                                lstm_workers=n_workers)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers,
                                  persistent_workers=bool(n_workers > 0),
                                  pin_memory=(device == 'gpu'))

    val_edges = os.path.join(edge_info, 'val_edge_index.json')
    val_attr = os.path.join(edge_info, 'val_edge_attr.json')

    val_dataset = DadsDataset(nodes, v_files, meta, embeddings, val_edges, val_attr,
                              scaler=scaler, sample=sample[1], node_ctx_dir=node_ctx_dir,
                              lstm_workers=n_workers)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=n_workers,
                                persistent_workers=bool(n_workers > 0),
                                pin_memory=(device == 'gpu'))

    def print_dads_training_overview(tr_ds, va_ds, n_nodes):
        print("\n[DADS] Training overview")
        print(f"- Train stations: {len(tr_ds.lstm_dataset.file_paths)}  Val stations: {len(va_ds.lstm_dataset.file_paths)}")
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

    meta['column_indices'] = train_dataset.column_indices
    meta['tensor_width'] = train_dataset.tensor_width

    meta['scaler'] = train_dataset.lstm_dataset.scaler
    meta['exog_dim'] = getattr(train_dataset, 'exog_dim', 0)
    meta['emb_dim'] = getattr(train_dataset, 'emb_dim', None)
    edge_dim = int(getattr(train_dataset, 'edge_dim', next(iter(train_dataset.edge_attr.values())).shape[-1]))
    model = DadsMetGNN(lstm_model, output_dim=1, n_nodes=nodes, hidden_dim=1024,
                       edge_attr_dim=edge_dim, dropout=dropout, learning_rate=learning_rate,
                       log_csv=logging_csv, use_target_exog_branch=True, **meta)

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

    trainer = pl.Trainer(max_epochs=100, callbacks=[checkpoint_callback, early_stop_callback],
                         accelerator=device, devices=1)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':

    target_var = 'tmax'

    d = '/home/dgketchum/data/IrrigationGIS/dads'
    training = '/data/ssd2/dads/training'

    print('========================== modeling {} =========================='.format(target_var))

    # lstm model
    lstm_model_ = os.path.join(training, 'lstm', 'checkpoints', 'tmax_20251004_1650')

    # data
    parquet_dir_ = os.path.join(training, 'parquet', f'{target_var}_obs')

    # graph (per target)
    edges = os.path.join(training, 'graph', f'{target_var}_obs')

    # climate embedding
    encoder_dir = os.path.join(training, 'autoencoder', 'checkpoints', '10011529')

    now = datetime.now().strftime('%m%d%H%M')
    chk = os.path.join(training, 'gnn', 'checkpoints', now)
    os.makedirs(chk, exist_ok=True)
    logger_csv = os.path.join(chk, 'training_{}.csv'.format(now))

    workers = 12
    device_ = 'gpu'

    node_ctx_dir = os.path.join(training, 'node_ctx')
    train_model(chk, parquet_dir_, lstm_model_, encoder_dir, edges, batch_size=256, nodes=5, dropout=0.5,
                learning_rate=0.001, n_workers=workers, logging_csv=logger_csv, device=device_, sample=None,
                node_ctx_dir=node_ctx_dir)
# ========================= EOF ====================================================================
