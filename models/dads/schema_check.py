import os
import json

import pandas as pd


def validate_training_schema(sample_file, scaler_json, lstm_model_dir, edge_attr_file, embedding_dir,
                             edge_map_file=None, node_ctx_dir=None):
    df = pd.read_parquet(sample_file)
    cols = df.columns.tolist()

    with open(scaler_json, 'r') as f:
        sp = json.load(f)
    feat_names = sp.get('feature_names')
    if feat_names is not None:
        assert cols == feat_names, "parquet columns != scaler feature_names"

    assert cols[0].endswith('_obs'), "first column must be target _obs"
    base = cols[0][:-4]
    assert cols[1].startswith(base) and not cols[1].endswith('_obs'), "second column should be gridded match for target"
    assert 'day_int' not in cols and 'day_diff' not in cols, "window bookkeeping leaked into table"

    meta_path = os.path.join(lstm_model_dir, 'training_metadata.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    num_bands = int(meta['num_bands'])
    assert num_bands == len(cols) - 2, "num_bands must equal feature_count - 2"

    with open(edge_attr_file, 'r') as f:
        edge_attr = json.load(f)
    edge_lens = {len(v) for v in edge_attr.values()}
    assert len(edge_lens) == 1, "edge_attr vectors must be uniform length"
    edge_dim = next(iter(edge_lens))

    emb_path = os.path.join(embedding_dir, 'embeddings.json')
    with open(emb_path, 'r') as f:
        embeddings = json.load(f)
    emb_lens = {len(v) for v in embeddings.values()}
    assert len(emb_lens) == 1, "embedding vectors must be uniform length"
    emb_dim = next(iter(emb_lens))

    missing_emb = 0
    missing_attr = 0
    if edge_map_file is not None:
        with open(edge_map_file, 'r') as f:
            edge_map = json.load(f)
        for k, nbrs in edge_map.items():
            if k not in embeddings or k not in edge_attr:
                missing_emb += int(k not in embeddings)
                missing_attr += int(k not in edge_attr)
            for n in nbrs:
                if n not in embeddings:
                    missing_emb += 1
                if n not in edge_attr:
                    missing_attr += 1

    ctx_dim = None
    if node_ctx_dir is not None:
        stn = os.path.splitext(os.path.basename(sample_file))[0]
        stn_dir = os.path.join(node_ctx_dir, stn)
        if os.path.isdir(stn_dir):
            for name in os.listdir(stn_dir):
                if name.endswith('.npy'):
                    import numpy as np
                    arr = np.load(os.path.join(stn_dir, name))
                    assert arr.ndim in (1, 2), "node context must be 1D or batched 2D"
                    ctx_dim = int(arr.shape[-1])
                    break

    report = {
        'target_col': cols[0],
        'gm_col': cols[1],
        'num_features': len(cols),
        'num_bands': num_bands,
        'edge_dim': edge_dim,
        'emb_dim': emb_dim,
        'missing_embeddings': missing_emb,
        'missing_edge_attr': missing_attr,
        'ctx_dim': ctx_dim,
    }
    return report


if __name__ == '__main__':
    import json
    import sys

    # Default preflight for tmax_obs using conventional training layout.
    target_var = 'tmax_obs'
    var = target_var.replace('_obs', '')

    # Locate training root
    training = '/data/ssd2/dads/training'
    lstm_ckpt = 'tmax_20251004_1650'

    parquet_dir = os.path.join(training, 'parquet', target_var)
    graph_dir = os.path.join(training, 'graph', target_var)
    node_ctx_dir = os.path.join(training, 'node_ctx')

    # Pick latest autoencoder checkpoint with embeddings.json
    ae_root = os.path.join(training, 'autoencoder', 'checkpoints')
    if not os.path.isdir(ae_root):
        raise FileNotFoundError(f'Missing autoencoder checkpoints dir: {ae_root}')
    ae_runs = [os.path.join(ae_root, d) for d in os.listdir(ae_root)
               if os.path.isdir(os.path.join(ae_root, d))]
    ae_runs.sort(key=lambda p: os.path.getmtime(p))
    embedding_dir = None
    for run in reversed(ae_runs):
        if os.path.exists(os.path.join(run, 'embeddings.json')):
            embedding_dir = run
            break
    if embedding_dir is None:
        raise FileNotFoundError('No autoencoder run with embeddings.json found.')

    lstm_model_dir = os.path.join(training, 'lstm','checkpoints', lstm_ckpt)
    if not os.path.isdir(lstm_model_dir):
        raise FileNotFoundError(f'Missing LSTM checkpoints dir: {lstm_model_dir}')

    # Graph files
    train_edge_attr = os.path.join(graph_dir, 'train_edge_attr.json')
    train_edge_index = os.path.join(graph_dir, 'train_edge_index.json')
    val_edge_attr = os.path.join(graph_dir, 'val_edge_attr.json')
    val_edge_index = os.path.join(graph_dir, 'val_edge_index.json')
    if not os.path.exists(train_edge_attr) or not os.path.exists(train_edge_index):
        raise FileNotFoundError(f'Missing graph files in {graph_dir}')

    # Scaler
    scaler_json = os.path.join(training, 'scalers', f'{var}.json')
    if not os.path.exists(scaler_json):
        raise FileNotFoundError(f'Missing scaler json: {scaler_json}')

    # Choose a sample station from training edge_attr that has a parquet file
    with open(train_edge_attr, 'r') as f:
        train_attr = json.load(f)
    station = None
    for st in train_attr.keys():
        p = os.path.join(parquet_dir, f'{st}.parquet')
        if os.path.exists(p):
            station = st
            sample_file = p
            break
    if station is None:
        # fallback: any parquet file
        files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
        if not files:
            raise FileNotFoundError(f'No parquet files in {parquet_dir}')
        sample_file = os.path.join(parquet_dir, files[0])

    report = validate_training_schema(sample_file,
                                      scaler_json,
                                      lstm_model_dir,
                                      train_edge_attr,
                                      embedding_dir,
                                      edge_map_file=train_edge_index,
                                      node_ctx_dir=node_ctx_dir)
    print('Schema check report:')
    for k, v in report.items():
        print(f'- {k}: {v}')
    print('\nOK: schema looks consistent for', target_var)

# ========================= EOF ====================================================================
