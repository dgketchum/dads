import json
import os
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from pandas.api.types import is_numeric_dtype


def _load_parquet_dir(d, limit_files=0):
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.parquet')]
    files.sort()
    if limit_files and limit_files > 0:
        files = files[:limit_files]
    frames = []
    for p in files:
        df = pd.read_parquet(p)
        frames.append(df)
    if frames:
        out = pd.concat(frames, axis=0, ignore_index=False)
    else:
        out = pd.DataFrame()
    return out


def _list_parquet_files(d, limit_files=0):
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.parquet')]
    files.sort()
    if limit_files and limit_files > 0:
        files = files[:limit_files]
    return files


def _iter_parquet_batches(path, x_cols, target_var, batch_rows):
    cols = [target_var] + x_cols
    pf = pq.ParquetFile(path)
    for rb in pf.iter_batches(columns=cols, batch_size=batch_rows):
        df = rb.to_pandas()  # likely error: mixed dtypes inflate memory
        y = df[target_var]
        x = df.drop(columns=[target_var])
        yield x, y


def _load_val_sample(val_dir, x_cols, target_var, drop_cols, limit_rows=500000, limit_files=0):
    val_files = _list_parquet_files(val_dir, limit_files=limit_files)
    xs, ys = [], []
    need = limit_rows if limit_rows and limit_rows > 0 else None
    for p in val_files:
        pf = pq.ParquetFile(p)
        cols = [target_var] + x_cols
        for rb in pf.iter_batches(columns=cols, batch_size=50000):
            df = rb.to_pandas()
            y = df[target_var]
            x = df.drop(columns=[target_var])
            xs.append(x)
            ys.append(y)
            if need is not None:
                have = sum(len(a) for a in ys)
                if have >= need:
                    x_out = pd.concat(xs, axis=0, ignore_index=False).iloc[:need]
                    y_out = pd.concat(ys, axis=0, ignore_index=False).iloc[:need]
                    return x_out, y_out
    if xs:
        x_out = pd.concat(xs, axis=0, ignore_index=False)
        y_out = pd.concat(ys, axis=0, ignore_index=False)
        return x_out, y_out
    return pd.DataFrame(columns=x_cols), pd.Series(dtype=float)


def train_model(dirpath, lgbm_root, target_var, now_str, params=None, debug_files=0, num_boost_round=5000,
                early_stopping_rounds=100, rows_per_batch=25000, val_limit_rows=500000, num_threads=None):
    train_dir = os.path.join(lgbm_root, target_var, 'train')
    val_dir = os.path.join(lgbm_root, target_var, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f'Missing train/val directories for {target_var} under {lgbm_root}')

    train_files = _list_parquet_files(train_dir, limit_files=debug_files)
    if not train_files:
        return None  # likely error: no data in lgbm tables

    # derive feature column order from schema (no full load)
    pf0 = pq.ParquetFile(train_files[0])
    all_cols0 = pf0.schema.names
    if target_var not in all_cols0:
        return None  # likely error: target column missing

    drop_cols = ['station']
    drop_cols = [c for c in drop_cols if c in all_cols0]

    x_cols = [c for c in all_cols0 if c not in ([target_var] + drop_cols)]

    # drop reserved index-like columns serialized by pandas parquet
    dropped_reserved = [c for c in x_cols if c.startswith('__index_level_') or c == 'index' or c.startswith('Unnamed:')]
    if dropped_reserved:
        x_cols = [c for c in x_cols if c not in dropped_reserved]

    # filter out non-numeric and missing columns based on a small sample to avoid feature_name mismatch
    sample_df = None
    for _rb in pq.ParquetFile(train_files[0]).iter_batches(columns=[target_var] + x_cols, batch_size=1000):
        sample_df = _rb.to_pandas()
        break
    dropped_non_numeric = []
    dropped_missing_train = []
    if sample_df is not None:
        present = set(sample_df.columns)
        keep = []
        for c in x_cols:
            if c not in present:
                dropped_missing_train.append(c)
                continue
            if not is_numeric_dtype(sample_df[c]):
                dropped_non_numeric.append(c)
                continue
            keep.append(c)
        x_cols = keep

    x_va, y_va = _load_val_sample(val_dir, x_cols, target_var, drop_cols, limit_rows=val_limit_rows,
                                  limit_files=debug_files)
    # align features present in validation as well
    dropped_missing_val = [c for c in x_cols if c not in x_va.columns]
    if dropped_missing_val:
        x_cols = [c for c in x_cols if c in x_va.columns]
        x_va = x_va[x_cols]
    x_va = x_va.astype(np.float32)
    y_va = y_va.astype(np.float32)
    mask_va = np.isfinite(y_va.values)
    x_va = x_va.loc[mask_va]
    y_va = y_va.loc[mask_va]

    if params is None:
        params = {
            'objective': 'regression',
            'metric': ['l1', 'l2'],
            'learning_rate': 0.05,
            'num_leaves': 128,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'min_data_in_leaf': 1000,
            'verbosity': -1,
            'device_type': 'gpu',
            'gpu_device_id': 0,
            'max_bin': 63,
        }
    if num_threads is not None:
        params['num_threads'] = int(num_threads)

    booster = None
    total_rows = 0
    total_batches = 0
    if rows_per_batch and rows_per_batch > 0:
        for p in train_files:
            m = pq.ParquetFile(p).metadata
            total_rows += m.num_rows
        total_batches = max(1, int(np.ceil(total_rows / rows_per_batch)))
    else:
        total_batches = len(train_files)

    rounds_per_batch = max(1, num_boost_round // total_batches)
    seen_batches = 0

    pbar = tqdm(total=total_batches, desc='LightGBM training')
    for p in train_files:
        if rows_per_batch and rows_per_batch > 0:
            for x_tr, y_tr in _iter_parquet_batches(p, x_cols, target_var, rows_per_batch):
                x_tr = x_tr.astype(np.float32)
                y_tr = y_tr.astype(np.float32)
                mask_tr = np.isfinite(y_tr.values)
                x_tr = x_tr.loc[mask_tr]
                y_tr = y_tr.loc[mask_tr]
                if x_tr.empty:
                    continue
                dtrain = lgb.Dataset(x_tr, label=y_tr, feature_name=x_cols, free_raw_data=True)
                dvalid = lgb.Dataset(x_va, label=y_va, reference=dtrain, feature_name=x_cols, free_raw_data=True)

                add_rounds = rounds_per_batch
                seen_batches += 1
                if seen_batches == total_batches:
                    rem = num_boost_round - rounds_per_batch * (total_batches - 1)
                    add_rounds = max(add_rounds, rem)

                booster = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=add_rounds,
                    valid_sets=[dtrain, dvalid],
                    valid_names=['train', 'val'],
                    init_model=booster,
                    keep_training_booster=True,
                )
                pbar.update(1)
        else:
            df_tr = pd.read_parquet(p)
            y_tr = df_tr[target_var]
            x_tr = df_tr.drop(columns=[target_var] + drop_cols, errors='ignore')
            x_tr = x_tr.reindex(columns=x_cols)
            x_tr = x_tr.astype(np.float32)
            y_tr = y_tr.astype(np.float32)
            mask_tr = np.isfinite(y_tr.values)
            x_tr = x_tr.loc[mask_tr]
            y_tr = y_tr.loc[mask_tr]
            if x_tr.empty:
                continue
            dtrain = lgb.Dataset(x_tr, label=y_tr, feature_name=x_cols, free_raw_data=True)
            dvalid = lgb.Dataset(x_va, label=y_va, reference=dtrain, feature_name=x_cols, free_raw_data=True)

            add_rounds = rounds_per_batch
            seen_batches += 1
            if seen_batches == total_batches:
                rem = num_boost_round - rounds_per_batch * (total_batches - 1)
                add_rounds = max(add_rounds, rem)

            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=add_rounds,
                valid_sets=[dtrain, dvalid],
                valid_names=['train', 'val'],
                init_model=booster,
                keep_training_booster=True,
            )
            pbar.update(1)
    pbar.close()

    model_path = os.path.join(dirpath, 'best_model.txt')
    booster.save_model(model_path, num_iteration=booster.best_iteration)

    meta = {
        'target_var': target_var,
        'best_iteration': int(booster.best_iteration),
        'feature_names': list(x_tr.columns),
        'params': params,
        'now': now_str,
        'dropped_non_numeric': dropped_non_numeric,
        'dropped_reserved': dropped_reserved,
        'dropped_missing_train': dropped_missing_train,
        'dropped_missing_val': dropped_missing_val,
    }
    with open(os.path.join(dirpath, 'training_metadata.json'), 'w') as fp:
        json.dump(meta, fp, indent=2)

    return model_path


if __name__ == '__main__':

    zoran = '/data/ssd2/dads/training'
    nvm = '/media/nvm/training'
    if os.path.exists(zoran):
        training = zoran
    elif os.path.exists(nvm):
        training = nvm
    else:
        raise NotImplementedError

    variable_ = 'tmax'
    target_var_ = f'{variable_}_obs'

    now_ = datetime.now().strftime('%Y%m%d_%H%M')
    chk_ = os.path.join(training, 'lgbm', 'checkpoints', f'{variable_}_{now_}')
    os.makedirs(chk_, exist_ok=True)

    lgbm_root_ = os.path.join(training, 'lgbm')

    params_ = {
        'objective': 'regression',
        'metric': ['l1', 'l2'],
        'learning_rate': 0.05,
        'num_leaves': 128,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_data_in_leaf': 1000,
        'verbosity': -1,
        'device_type': 'cpu',
        'gpu_device_id': 0,
        'max_bin': 63,
    }

    train_model(chk_, lgbm_root_, target_var_, now_, params=params_, debug_files=0, num_boost_round=5000,
                early_stopping_rounds=200, rows_per_batch=25000, val_limit_rows=500000, num_threads=50)

# ========================= EOF ====================================================================
