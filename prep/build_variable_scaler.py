import json
import os

import pandas as pd
import torch
from tqdm import tqdm

from models.scalers import MinMaxScaler


def fit_and_save_scaler(file_paths, feature_names, scaler_path):
    all_data_chunks = []
    for file_path in tqdm(file_paths, desc="Reading files for scaler"):
        try:
            df = pd.read_parquet(file_path)[feature_names]
            assert df.columns.tolist() == feature_names, "feature_names mismatch"
            df.dropna(inplace=True)
            if not df.empty:
                all_data_chunks.append(torch.tensor(df.values, dtype=torch.float32))
        except Exception:
            continue

    assert all_data_chunks, "no data available to fit scaler"

    full_dataset = torch.cat(all_data_chunks, dim=0)
    scaler = MinMaxScaler(axis=0)
    scaler.fit(full_dataset.numpy())

    bias_ = scaler.bias.flatten().tolist()
    scale_ = scaler.scale.flatten().tolist()
    dct = {'bias': bias_, 'scale': scale_, 'feature_names': feature_names}
    with open(scaler_path, 'w') as fp:
        json.dump(dct, fp, indent=4)
    return scaler


def build_variable_scaler(parquet_root, variable, scaler_dir=None):
    target = f"{variable}_obs"
    var_dir = os.path.join(parquet_root, target)
    files = [os.path.join(var_dir, f) for f in os.listdir(var_dir) if f.endswith('.parquet')]
    assert files, "no parquet files found for variable"

    feature_names = pd.read_parquet(files[0]).columns.tolist()
    assert len(feature_names) >= 2 and feature_names[0].endswith('_obs') and not feature_names[1].endswith('_obs')

    if scaler_dir is None:
        training_root = os.path.dirname(parquet_root)
        scaler_dir = os.path.join(training_root, 'scalers')
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_path = os.path.join(scaler_dir, f"{variable}.json")

    _ = fit_and_save_scaler(files, feature_names, scaler_path)
    return scaler_path


def get_variable_scaler_path(parquet_root, variable, ensure_dir=True):
    training_root = os.path.dirname(parquet_root)
    scaler_dir = os.path.join(training_root, 'scalers')
    if ensure_dir:
        os.makedirs(scaler_dir, exist_ok=True)
    return os.path.join(scaler_dir, f"{variable}.json")


def load_variable_scaler(parquet_root, variable, feature_names=None):
    path = get_variable_scaler_path(parquet_root, variable, ensure_dir=False)
    if not os.path.exists(path):
        path = build_variable_scaler(parquet_root, variable)
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
