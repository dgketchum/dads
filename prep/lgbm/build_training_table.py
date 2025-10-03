import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def _load_json(p):
    with open(p, 'r') as fp:
        d = json.load(fp)
    d = {str(k): v for k, v in d.items()}
    return d


def _flat(v):
    a = np.asarray(v)
    a = a.reshape(-1)
    return a


def build_table(parquet_dir: str,
                edge_index: Dict[str, List[str]],
                edge_attr: Dict[str, List[float]],
                embeddings: Dict[str, List[float]],
                out_dir: Optional[str] = None,
                split: Optional[str] = None,
                rows_per_file: int = 5_000_000,
                file_prefix: str = 'part',
                obs_col: Optional[str] = None):

    keys = list(edge_index.keys())
    k0 = str(keys[0])
    p0 = os.path.join(parquet_dir, f'{k0}.parquet')
    base = pd.read_parquet(p0).sort_index()
    base_cols = list(base.columns)

    emb0 = _flat(embeddings[k0])
    attr0 = _flat(edge_attr[k0])
    n_nbr_all = len(edge_index[k0])
    nbr0_all = [str(nb) for nb in edge_index[k0]]
    nbr0 = [nb for nb in nbr0_all if nb != k0]  # drop self neighbor
    n_nbr = len(nbr0)

    cols = []
    cols.extend(base_cols)
    cols.append('station')
    cols.extend([f'emb_{i}' for i in range(len(emb0))])
    cols.extend([f'attr_{i}' for i in range(len(attr0))])
    for n_i in range(n_nbr):
        cols.extend([f'nbr{n_i}_emb_{i}' for i in range(len(emb0))])
        cols.extend([f'nbr{n_i}_attr_{i}' for i in range(len(attr0))])
        if obs_col is not None:
            cols.append(f'nbr{n_i}_{obs_col}')

    dct = {c: [] for c in cols}
    idx = []
    part = 0

    def _flush_chunk():
        nonlocal dct, idx, part
        if out_dir is None:
            return None
        if not idx:
            return None
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        block = 100_000
        total = len(idx)
        start = 0
        while start < total:
            end = start + block
            if end > total:
                end = total
            df_chunk = pd.DataFrame({k: v[start:end] for k, v in dct.items()})
            df_chunk.index = pd.to_datetime(idx[start:end])
            pth = os.path.join(out_dir, f"{file_prefix}_{part:05d}.parquet")
            df_chunk.to_parquet(pth)
            part += 1
            start = end
        dct = {c: [] for c in cols}
        idx = []
        return None

    pbar = tqdm(edge_index.items(), total=len(edge_index))
    for staid, neigh in pbar:
        staid = str(staid)
        pth = os.path.join(parquet_dir, f'{staid}.parquet')
        df = pd.read_parquet(pth).sort_index()

        n = len(df)

        rec = {c: [] for c in cols}
        add_ok = True

        for c in base_cols:
            rec[c].extend(df[c].tolist())

        rec['station'].extend([staid] * n)

        if staid in embeddings and staid in edge_attr:
            emb = _flat(embeddings[staid])
            if len(emb) == len(emb0):
                for i, v in enumerate(emb):
                    rec[f'emb_{i}'].extend([v] * n)
            else:
                add_ok = False
            attr = _flat(edge_attr[staid])
            if len(attr) == len(attr0):
                for i, v in enumerate(attr):
                    rec[f'attr_{i}'].extend([v] * n)
            else:
                add_ok = False
        else:
            add_ok = False

        if add_ok:
            out_i = 0
            for nb_raw in neigh:
                nb = str(nb_raw)
                if nb == staid:
                    continue  # drop self neighbor features
                if nb not in embeddings or nb not in edge_attr:
                    add_ok = False
                    break
                nb_emb = _flat(embeddings[nb])
                if len(nb_emb) != len(emb0):
                    add_ok = False
                    break
                for i, v in enumerate(nb_emb):
                    rec[f'nbr{out_i}_emb_{i}'].extend([v] * n)
                nb_attr = _flat(edge_attr[nb])
                if len(nb_attr) != len(attr0):
                    add_ok = False
                    break
                for i, v in enumerate(nb_attr):
                    rec[f'nbr{out_i}_attr_{i}'].extend([v] * n)
                if obs_col is not None:
                    nb_pth = os.path.join(parquet_dir, f'{nb}.parquet')
                    nb_df = pd.read_parquet(nb_pth).sort_index()
                    if obs_col in nb_df.columns:
                        nb_vals = nb_df[obs_col].reindex(df.index)
                    else:
                        nb_vals = pd.Series(np.nan, index=df.index)  # fill mandatory neighbor obs with NaN
                    rec[f'nbr{out_i}_{obs_col}'].extend(nb_vals.tolist())
                out_i += 1
            if out_i != n_nbr:
                add_ok = False
        else:
            add_ok = False

        if add_ok:
            idx.extend(df.index.tolist())
            for k in cols:
                dct[k].extend(rec[k])
            pbar.set_postfix({'obs': len(idx)})
            if out_dir is not None and len(idx) >= rows_per_file:
                _flush_chunk()

    if out_dir is not None:
        _flush_chunk()
        return None
    out = pd.DataFrame(dct)
    out.index = pd.to_datetime(idx)
    return out


if __name__ == '__main__':

    variable_ = 'tmax'
    target_var_ = f'{variable_}_obs'

    zoran = '/data/ssd2/dads/training'
    nvm = '/media/nvm/training'
    if os.path.exists(zoran):
        training = zoran
    elif os.path.exists(nvm):
        training = nvm
    else:
        raise NotImplementedError

    parq_ = os.path.join(training, 'parquet', target_var_)
    graph_dir = os.path.join(training, 'graph')

    # set to a specific run directory containing embeddings.json
    ae_dir = os.path.join(training, 'autoencoder', 'checkpoints', '10011529')
    embeddings_ = os.path.join(ae_dir, 'embeddings.json')

    emb = _load_json(embeddings_)

    tr_idx = _load_json(os.path.join(graph_dir, 'train_edge_index.json'))
    tr_attr = _load_json(os.path.join(graph_dir, 'train_edge_attr.json'))
    va_idx = _load_json(os.path.join(graph_dir, 'val_edge_index.json'))
    va_attr = _load_json(os.path.join(graph_dir, 'val_edge_attr.json'))

    out_dir = os.path.join(training, 'lgbm', target_var_)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_dir_ = os.path.join(out_dir, 'train')
    val_dir_ = os.path.join(out_dir, 'val')

    build_table(parq_, tr_idx, tr_attr, emb, out_dir=train_dir_, split='train',
                rows_per_file=1_000_000, file_prefix=f'{variable_}_train_part', obs_col=target_var_)
    build_table(parq_, va_idx, va_attr, emb, out_dir=val_dir_, split='val',
                rows_per_file=1_000_000, file_prefix=f'{variable_}_val_part', obs_col=target_var_)

# ========================= EOF ===============================================================================
