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

# ========================= EOF ====================================================================
