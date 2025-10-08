import json
import os
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.loader import DataLoader

from models.dads.dads_gnn import DadsMetGNN
from models.dads.dataset import DadsDataset
from prep.build_variable_scaler import load_variable_scaler


def train_model(dirpath, parquet_dir, lstm_model, embeddings, edge_info, nodes=5, batch_size=1,
                dropout=0.2, learning_rate=0.01, n_workers=1, logging_csv=None, device='gpu', sample=None,
                node_ctx_dir=None):
    """"""

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
    _scaler_obj, scaler, _ = load_variable_scaler(parquet_root, var_name)

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

    meta['column_indices'] = train_dataset.column_indices
    meta['tensor_width'] = train_dataset.tensor_width

    meta['scaler'] = train_dataset.lstm_dataset.scaler
    edge_dim = next(iter(train_dataset.edge_attr.values())).shape[-1]  # infer edge attribute width
    model = DadsMetGNN(lstm_model, output_dim=1, n_nodes=nodes, hidden_dim=1024,
                       edge_attr_dim=edge_dim, dropout=dropout, learning_rate=learning_rate,
                       log_csv=logging_csv, **meta)

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
