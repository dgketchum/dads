import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from collections import OrderedDict
from prep.columns_desc import CDR_FEATURES


def _scan_windows_worker(args):
    """Scan a file and return valid consecutive-window start indices and end days.

    Returns (file_index, start_ilocs, end_days) with day step = 1.
    Only requires the target column to compute valid windows.
    """
    file_index, file_path, chunk_size, target_col = args
    try:
        df = pd.read_parquet(file_path, columns=[target_col])
        # Require non-NaN target
        df.dropna(subset=[target_col], inplace=True)
        if len(df) < chunk_size:
            return file_index, [], []
    except Exception:
        return file_index, [], []

    day_ints = df.index.to_julian_date().astype(np.int32).to_numpy()
    diffs = np.diff(day_ints)
    window_size = chunk_size - 1
    if len(diffs) < window_size:
        return file_index, [], []
    is_consecutive = (diffs == 1)
    convolved = np.convolve(is_consecutive, np.ones(window_size, dtype=int), mode='valid')
    valid_start_ilocs = np.where(convolved == window_size)[0]
    if valid_start_ilocs.size == 0:
        return file_index, [], []
    end_days = [int(day_ints[i + chunk_size - 1]) for i in valid_start_ilocs]
    return file_index, valid_start_ilocs.tolist(), end_days


class LSTMDataset(Dataset):
    """Unified dataset for LSTM pretraining and for assembling DADS graph samples.

    Ingests
    - Station Parquets (file_paths) created by the preprocessing pipeline
      (training/parquet/{target_var}_obs/{station}.parquet). Required column order:
      target observation in col 0 ("*_obs"), optional comparator in col 1 (same base
      variable without "_obs"), followed by other features. This dataset selects:
      * target (col 0),
      * optional comparator (col 1 if present),
      * day-of-interest exogenous channels: 'rsun' and NOAA CDR bands defined in
        prep.columns_desc.CDR_FEATURES when present in the Parquet schema.
    - Variable-specific MinMaxScaler (scaler) from prep.build_variable_scaler; fit on
      graph train_ids only to prevent leakage, then used to scale the selected columns
      before batching. Window indexing uses only the target column and requires strictly
      consecutive days.

    Behavior
    - Exogenous availability varies in time; any missing rsun/CDR values are left as NaN
      by Pandas, then set to 0 after scaling when forming features so early-period windows
      (pre-remote sensing) can be used without masking.
    - For each indexed window, the end-of-window day is returned as an integer Julian day
      ("day_int") for downstream alignment (e.g., node contexts).

    Returns
    - Training path (return_station_name=False): (y, comparator, features)
      * y: (T,) target sequence.
      * comparator: (T,) comparator sequence or NaN if unavailable.
      * features: (T, 1 + exog_dim) where channel 0 is the 1-step lag of the target,
        and remaining channels are contemporaneous exogenous inputs (rsun + CDR). NaNs
        in exogenous channels are 0 after scaling.
    - DADS path (return_station_name=True): (y, comparator, empty_seq, station_name, day_int)
      * empty_seq is a placeholder (DADS builds node contexts separately and does not
        consume an input sequence for the target node).
      * station_name and day_int are used to fetch neighbor contexts and construct the graph.
    """
    def __init__(self, file_paths, station_names, feature_names, sample_dimensions, scaler, strided=False,
                 transform=None, return_station_name=False, n_workers=1, cache_size_files=8):

        self.sample_dimensions = sample_dimensions
        self.transform = transform
        self.feature_names = feature_names
        self.return_station_name = return_station_name
        self.strided = strided
        self.scaler = scaler

        # Identify comparator column by base name if present; else mark missing
        base = self.feature_names[0].replace('_obs', '')
        comp_idx = None
        for i, name in enumerate(self.feature_names[1:], start=1):
            if name.startswith(base + '_'):
                comp_idx = i
                break
        self.comp_idx = comp_idx

        # Store file mapping
        self.file_paths = list(file_paths)
        self.file_station_names = list(station_names)

        # Columns we will materialize per-sample: target, optional comparator, RSUN + NOAA CDR
        exog_names = ['rsun'] + list(CDR_FEATURES)
        self.exog_idx = [i for i, n in enumerate(self.feature_names) if n in exog_names]
        self.selected_indices = [0] + ([self.comp_idx] if self.comp_idx is not None else []) + self.exog_idx
        self.selected_columns = [self.feature_names[i] for i in self.selected_indices]

        # Build index of (file_idx, start_iloc, end_day)
        self._build_index(n_workers)

        # Small per-worker LRU cache of loaded, scaled file tensors (selected columns only)
        self._cache = OrderedDict()
        self._cache_max = int(cache_size_files)
        if self._cache_max < 1:
            self._cache_max = 1

    def _get_file_tensor(self, file_idx):
        # LRU cache lookup
        if file_idx in self._cache:
            t = self._cache.pop(file_idx)
            self._cache[file_idx] = t
            return t

        file_path = self.file_paths[file_idx]
        df = pd.read_parquet(file_path, columns=self.selected_columns)
        # Align with indexer: drop rows with NaN target only
        df.dropna(subset=[self.selected_columns[0]], inplace=True)
        arr = torch.tensor(df.values, dtype=torch.float32)

        # Scale selected columns only
        if self.scaler is not None:
            try:
                bias = torch.as_tensor(self.scaler.bias, dtype=torch.float32).view(1, -1)
                scale = torch.as_tensor(self.scaler.scale, dtype=torch.float32).view(1, -1)
                sel = torch.as_tensor(self.selected_indices, dtype=torch.long)
                bias_sel = bias[:, sel]
                scale_sel = scale[:, sel]
                arr = (arr - bias_sel) / scale_sel + 5e-8
            except Exception:
                pass

        # Insert into LRU and evict old if necessary
        self._cache[file_idx] = arr
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return arr

    def _build_index(self, n_workers):
        print("Reading metadata and indexing windows...")
        chunk_size_ = self.sample_dimensions[0]
        target_col = self.feature_names[0]
        tasks_ = [(i, self.file_paths[i], chunk_size_, target_col) for i in range(len(self.file_paths))]

        index_ = []
        if n_workers == 1:
            results_ = []
            for task in tqdm(tasks_, total=len(tasks_), desc="Indexing LSTM windows"):
                r = _scan_windows_worker(task)
                results_.append(r)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                results_ = list(tqdm(ex.map(_scan_windows_worker, tasks_),
                                     total=len(tasks_),
                                     desc="Indexing LSTM windows"))

        for file_index, starts, end_days in results_:
            for s, d in zip(starts, end_days):
                index_.append((file_index, s, d))

        self.index = index_
        print(f"Successfully indexed {len(self.index)} samples across {len(self.file_paths)} files.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """Return a sample for the selected path.

        - Training: (y, comparator, features) where features is (T, 1) lagged target.
        - DADS: (y, comparator, empty_seq, station_name, day_int).
        """
        file_idx, start_iloc, end_day = self.index[idx]
        station_name = self.file_station_names[file_idx] if self.file_station_names else None

        full_tensor = self._get_file_tensor(file_idx)
        chunk = full_tensor[start_iloc: start_iloc + self.sample_dimensions[0], :]

        # Target and comparator
        y = chunk[:, 0]
        if self.comp_idx is not None and len(self.selected_indices) > 1:
            comparator = chunk[:, 1]
        else:
            comparator = torch.full_like(y, float('nan'))

        if self.return_station_name:
            # Dads path: omit target-node sequence to drop dependency on target meteorology
            seq = torch.empty(0)
            day_int = end_day
            return y, comparator, seq, station_name, day_int
        else:
            # LSTM training path: multivariate (lagged target + RSUN/CDR exogenous)
            obs_feat = y.unsqueeze(-1)
            obs_shift = obs_feat.clone()
            if obs_shift.shape[0] > 1:
                obs_shift[1:, 0] = obs_feat[:-1, 0]
            obs_shift[0, 0] = obs_feat[0, 0]
            start_exog = 1 + (1 if self.comp_idx is not None else 0)
            exog = chunk[:, start_exog:]
            if exog.numel() > 0:
                exog = torch.nan_to_num(exog, nan=0.0, posinf=0.0, neginf=0.0)
                features = torch.cat([obs_shift, exog], dim=1)
            else:
                features = obs_shift
            return y, comparator, features


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
