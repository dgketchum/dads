import json
import os
import numpy as np

import pandas as pd
import torch
from tqdm import tqdm

from models.scalers import MinMaxScaler
from models.dads.value_limits import FEATURE_LIMITS, TARGET_LIMITS
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


def fit_and_save_scaler(file_paths, feature_names, scaler_path, num_workers=None, variable=None):
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
    mask = np.ones_like(full_dataset, dtype=bool)
    n_rows = full_dataset.shape[0]
    # feature-level limits
    feature_rows_masked = np.zeros(n_rows, dtype=bool)
    feature_invalid_stats = []
    if FEATURE_LIMITS:
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        for fname, lim in FEATURE_LIMITS.items():
            if fname in name_to_idx and lim is not None:
                j = name_to_idx[fname]
                lo, hi = float(lim[0]), float(lim[1])
                xj = full_dataset[:, j]
                bad = (xj < lo) | (xj > hi)
                if bad.any():
                    feature_rows_masked |= bad
                    mask[:, j] &= ~bad
                    cnt = int(bad.sum())
                    mn = float(np.nanmin(xj)) if xj.size else np.nan
                    mx = float(np.nanmax(xj)) if xj.size else np.nan
                    feature_invalid_stats.append((fname, cnt, 100.0 * cnt / n_rows, lo, hi, mn, mx))
    # target limits on first column (variable_obs)
    target_rows_masked = 0
    if variable is not None:
        lim = TARGET_LIMITS.get(variable)
        if lim is None:
            raise ValueError(f"TARGET_LIMITS not set for variable {variable}")  # likely error if unset
        lo, hi = float(lim[0]), float(lim[1])
        x0 = full_dataset[:, 0]
        bad0 = (x0 < lo) | (x0 > hi)
        target_rows_masked = int(bad0.sum())
        mask[:, 0] &= ~bad0
    # concise prints
    if feature_rows_masked.any():
        print('[Scaler] rows masked by FEATURE_LIMITS:', int(feature_rows_masked.sum()), '/', int(n_rows))
        if feature_invalid_stats:
            feature_invalid_stats.sort(key=lambda t: t[1], reverse=True)
            top = feature_invalid_stats[:20]
            for name, cnt, pct, lo, hi, mn, mx in top:
                print('[Scaler][limit]', name, 'invalid=', cnt, f'({pct:.4f}%)', 'limits=(', lo, ',', hi, ')', 'min=', mn, 'max=', mx)
    print('[Scaler] rows masked by TARGET_LIMITS:', int(target_rows_masked), '/', int(n_rows))
    # target distribution quick-look
    x0 = full_dataset[:, 0]
    try:
        q05 = float(np.quantile(x0, 0.05))
        q50 = float(np.quantile(x0, 0.50))
        q95 = float(np.quantile(x0, 0.95))
        mn0 = float(np.min(x0))
        mx0 = float(np.max(x0))
        suspected_kelvin = (q50 > 130.0)
        print('[Scaler][target]', feature_names[0], 'min=', mn0, 'p05/p50/p95=', q05, q50, q95, 'max=', mx0, 'suspected_kelvin=', suspected_kelvin)
    except Exception:
        pass
    # fully valid rows under all applied limits
    fully_valid = int(np.all(mask, axis=1).sum())
    print('[Scaler] fully_valid_rows under limits:', fully_valid, '/', int(n_rows))
    scaler = MinMaxScaler(axis=0)
    scaler.fit(full_dataset, mask=mask)

    bias_ = scaler.bias.flatten().tolist()
    scale_ = scaler.scale.flatten().tolist()
    dct = {'bias': bias_, 'scale': scale_, 'feature_names': feature_names}
    with open(scaler_path, 'w') as fp:
        json.dump(dct, fp, indent=4)
    print(f'[Scaler] wrote new scaler with {len(feature_names)} features: {feature_names}')
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

    _ = fit_and_save_scaler(files, feature_names, scaler_path, num_workers, variable=variable)
    return scaler_path


def get_variable_scaler_path(parquet_root, variable, ensure_dir=True):
    training_root = os.path.dirname(parquet_root)
    scaler_dir = os.path.join(training_root, 'scalers')
    if ensure_dir:
        os.makedirs(scaler_dir, exist_ok=True)
    return os.path.join(scaler_dir, f"{variable}.json")


def load_variable_scaler(parquet_root, variable, feature_names=None, station_ids=None, split_ids_path=None,
                         rebuild=True):
    """Load a variable-specific MinMax scaler; optionally rebuild.

    - When rebuild=True (default), fit using train-only stations (from station_ids or split_ids_path)
      and persist to training/scalers/{variable}.json to prevent leakage.
    - When rebuild=False, reuse an existing scaler JSON if present; if missing, falls back to rebuild.
    """
    path = get_variable_scaler_path(parquet_root, variable, ensure_dir=True)

    if not rebuild and os.path.exists(path):
        try:
            with open(path, 'r') as f:
                params = json.load(f)
            # Validate feature_names if provided
            if 'feature_names' in params and feature_names is not None:
                assert params['feature_names'] == feature_names, "scaler feature_names mismatch"
            scaler = MinMaxScaler()
            scaler.bias = torch.as_tensor(params['bias']).reshape(1, -1).numpy()
            scaler.scale = torch.as_tensor(params['scale']).reshape(1, -1).numpy()
            return scaler, path, params.get('feature_names')
        except Exception:
            # fall through to rebuild on any error
            pass

    # Rebuild path: require train-only ids
    assert (station_ids is not None) or (split_ids_path is not None), "train_ids required for scaler rebuild"
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
    pass

# ========================= EOF ====================================================================
