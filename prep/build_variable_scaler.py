import json
import os
import numpy as np

import pandas as pd
import torch
from tqdm import tqdm

from models.scalers import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor


def _read_parquet_features(arg):
    file_path, feature_names = arg
    try:
        df = pd.read_parquet(file_path)[feature_names]
        assert df.columns.tolist() == feature_names, "feature_names mismatch"
        df.dropna(inplace=True)
        if not df.empty:
            a = df.values.astype(np.float32)
            return a
    except Exception:
        return None
    return None


def fit_and_save_scaler(file_paths, feature_names, scaler_path, num_workers=None):
    """Fit MinMax scaler on provided files and persist to JSON.

    Expects files to already be filtered to train-only stations to avoid leakage.
    """
    all_data_chunks = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        args = [(p, feature_names) for p in file_paths]
        for chunk in tqdm(ex.map(_read_parquet_features, args),
                          total=len(file_paths),
                          desc="Reading files for scaler"):
            if chunk is not None:
                all_data_chunks.append(chunk)

    assert all_data_chunks, "no data available to fit scaler"

    full_dataset = np.vstack(all_data_chunks)
    scaler = MinMaxScaler(axis=0)
    scaler.fit(full_dataset)

    bias_ = scaler.bias.flatten().tolist()
    scale_ = scaler.scale.flatten().tolist()
    dct = {'bias': bias_, 'scale': scale_, 'feature_names': feature_names}
    with open(scaler_path, 'w') as fp:
        json.dump(dct, fp, indent=4)
    return scaler


def build_variable_scaler(parquet_root, variable, scaler_dir=None, num_workers=None, station_ids=None, split_ids_path=None):
    """Build scaler for a variable using train-only stations from the graph split.

    Requires either an explicit list of station_ids or a path to graph/train_ids.json.
    """
    target = f"{variable}_obs"
    var_dir = os.path.join(parquet_root, target)
    files = [os.path.join(var_dir, f) for f in os.listdir(var_dir) if f.endswith('.parquet')]
    if split_ids_path is not None and os.path.exists(split_ids_path):
        try:
            with open(split_ids_path, 'r') as fp:
                ids = json.load(fp)
            station_ids = set(str(s) for s in ids) if ids is not None else station_ids
        except Exception:
            pass  # likely error if path invalid
    if station_ids is not None:
        sids = set(str(s) for s in station_ids)
        files = [p for p in files if os.path.splitext(os.path.basename(p))[0] in sids]
    assert files, "no parquet files found for variable"

    feature_names = pd.read_parquet(files[0]).columns.tolist()
    assert len(feature_names) >= 2 and feature_names[0].endswith('_obs') and not feature_names[1].endswith('_obs')

    if scaler_dir is None:
        training_root = os.path.dirname(parquet_root)
        scaler_dir = os.path.join(training_root, 'scalers')
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_path = os.path.join(scaler_dir, f"{variable}.json")

    _ = fit_and_save_scaler(files, feature_names, scaler_path, num_workers)
    return scaler_path


def get_variable_scaler_path(parquet_root, variable, ensure_dir=True):
    training_root = os.path.dirname(parquet_root)
    scaler_dir = os.path.join(training_root, 'scalers')
    if ensure_dir:
        os.makedirs(scaler_dir, exist_ok=True)
    return os.path.join(scaler_dir, f"{variable}.json")


def load_variable_scaler(parquet_root, variable, feature_names=None, station_ids=None, split_ids_path=None):
    """Always (re)build and load a train-only scaler to prevent information leakage.

    The graph's train_ids define the station subset used for fitting.
    """
    assert (station_ids is not None) or (split_ids_path is not None), "train_ids required for scaler"
    # always (re)build from the provided split to avoid leakage
    path = build_variable_scaler(parquet_root, variable, station_ids=station_ids, split_ids_path=split_ids_path)
    with open(path, 'r') as f:
        params = json.load(f)
    scaler = MinMaxScaler()
    scaler.bias = torch.as_tensor(params['bias']).reshape(1, -1).numpy()
    scaler.scale = torch.as_tensor(params['scale']).reshape(1, -1).numpy()
    if 'feature_names' in params and feature_names is not None:
        assert params['feature_names'] == feature_names, "scaler feature_names mismatch"
    return scaler, path, params.get('feature_names')


if __name__ == '__main__':
    """"""
    variable_ = 'tmax'

    parquet_root_ = '/data/ssd2/dads/training/parquet'
    scaler_dir_ = '/data/ssd2/dads/training/scalers'

    path_ = build_variable_scaler(parquet_root_, variable_, scaler_dir_)

# ========================= EOF ====================================================================
