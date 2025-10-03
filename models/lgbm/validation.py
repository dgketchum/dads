import os
import json
from typing import Dict, List, Optional

from tqdm import tqdm
import lightgbm as lgb
import numpy as np
import pandas as pd


def _list_parquet(dir_path: str) -> List[str]:
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.parquet')]
    files.sort()
    return files


def _station_list_from_val_dir(val_dataset_dir: str) -> List[str]:
    files = _list_parquet(val_dataset_dir)
    stations = [os.path.splitext(os.path.basename(p))[0] for p in files]
    return stations


def build_lgbm_val_predictions(model_dir: str, lgbm_root: str, target_var: str, limit_files: int = 0,
                               val_dataset_dir: Optional[str] = None) -> pd.DataFrame:
    meta_path = os.path.join(model_dir, 'training_metadata.json')
    with open(meta_path, 'r') as fp:
        meta = json.load(fp)
    feature_names = list(meta['feature_names'])
    best_iteration = int(meta['best_iteration']) if 'best_iteration' in meta else None

    model_path = os.path.join(model_dir, 'best_model.txt')
    booster = lgb.Booster(model_file=model_path)

    val_dir = os.path.join(lgbm_root, target_var, 'val')
    files = _list_parquet(val_dir)
    if limit_files and limit_files > 0:
        files = files[:limit_files]

    station_set = None
    if val_dataset_dir is not None:
        station_set = set(_station_list_from_val_dir(val_dataset_dir))

    frames = []
    for p in tqdm(files, total=len(files), desc=f'Inference on {target_var}'):
        df = pd.read_parquet(p)
        if 'station' not in df.columns:
            continue
        if station_set is not None:
            df = df[df['station'].astype(str).isin(station_set)]
            if df.empty:
                continue
        x = df.reindex(columns=feature_names)
        if best_iteration is None:
            yhat = booster.predict(x)
        else:
            yhat = booster.predict(x, num_iteration=best_iteration)
        out = pd.DataFrame({'station': df['station'].astype(str).values,
                            'dt': pd.to_datetime(df.index).strftime('%Y-%m-%d').values,
                            'lgbm': yhat.astype(np.float32)})
        out = out.groupby(['station', 'dt'], as_index=False).mean()
        frames.append(out)

    if frames:
        preds = pd.concat(frames, axis=0, ignore_index=True)
    else:
        preds = pd.DataFrame(columns=['station', 'dt', 'lgbm'])
    return preds


def compare_validation(val_dataset_dir: str, preds: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    files = _list_parquet(val_dataset_dir)
    cols = None
    acc = {}

    for p in files:
        df = pd.read_parquet(p)
        if cols is None:
            cols = [c for c in df.columns if c not in ['station', 'dt', 'obs']]
            for name in cols + ['lgbm']:
                acc[name] = {'n': 0, 'sse': 0.0, 'sae': 0.0, 'ysum': 0.0, 'y2sum': 0.0}

        merged = df.merge(preds, on=['station', 'dt'], how='left')

        for name in cols + ['lgbm']:
            if name not in merged.columns:
                continue
            y = merged['obs'].values
            yhat = merged[name].values
            m = np.isfinite(y) & np.isfinite(yhat)
            if not np.any(m):
                continue
            ye = y[m]
            yh = yhat[m]
            n = ye.size
            sse = np.sum((ye - yh) ** 2)
            sae = np.sum(np.abs(ye - yh))
            ysum = np.sum(ye)
            y2sum = np.sum(ye ** 2)
            acc[name]['n'] += int(n)
            acc[name]['sse'] += float(sse)
            acc[name]['sae'] += float(sae)
            acc[name]['ysum'] += float(ysum)
            acc[name]['y2sum'] += float(y2sum)

    rows = []
    for name, a in acc.items():
        n = a['n']
        if n == 0:
            continue
        mae = a['sae'] / n
        rmse = np.sqrt(a['sse'] / n)
        ymean = a['ysum'] / n
        sst = a['y2sum'] - n * (ymean ** 2)
        r2 = 1.0 - (a['sse'] / sst) if sst != 0.0 else np.nan
        rows.append({'model': name, 'n': n, 'mae': mae, 'rmse': rmse, 'r2': r2})

    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)
    return res


if __name__ == '__main__':

    home = os.path.expanduser('~')
    base = os.path.join('/data', 'ssd2', 'dads', 'training')
    lgbm_root_ = os.path.join(base, 'lgbm')
    var_ = 'tmax'
    target_var_ = f'{var_}_obs'

    model_dir_ = os.path.join(base, 'lgbm', 'checkpoints', 'tmax_20251003_1109')
    val_data_dir_ = os.path.join(base, 'validation_sets', var_)

    preds_df_ = build_lgbm_val_predictions(model_dir_, lgbm_root_, target_var_, limit_files=0,
                                           val_dataset_dir=val_data_dir_)
    out_csv_ = os.path.join(val_data_dir_, f'metrics_{var_}.csv')
    _ = compare_validation(val_data_dir_, preds_df_, out_csv_)
# ========================= EOF ====================================================================
