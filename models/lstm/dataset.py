import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def _read_and_chunk_worker(args):
    """Read one station file and produce consecutive daily chunks.

    - Requires non-NaN target; comparator optional; RS/GEO may be NaN.
    - Builds windows of length `chunk_size` where `day_diff == 1` for all steps.
    - Returns lists of (chunk arrays, station names, end-of-window day_int).
    """
    file_path, station_name, chunk_size, feature_names = args
    try:
        df = pd.read_parquet(file_path)[feature_names]
        # Require target present; comparator optional for training/contexts
        df.dropna(subset=[feature_names[0]], inplace=True)
        if len(df) < chunk_size:
            return [], [], []
    except Exception:
        return [], [], []

    df['day_int'] = df.index.to_julian_date().astype(np.int32)
    df['day_diff'] = df['day_int'].diff()
    data_np = df.to_numpy(dtype=np.float32)

    diff_col_idx = -1
    is_consecutive = (data_np[1:, diff_col_idx] == 1.0)

    window_size = chunk_size - 1
    if len(is_consecutive) < window_size:
        return [], [], []

    convolved = np.convolve(is_consecutive, np.ones(window_size, dtype=int), mode='valid')
    valid_start_ilocs = np.where(convolved == window_size)[0]

    if valid_start_ilocs.size == 0:
        return [], [], []

    num_features = len(feature_names)
    chunks = [data_np[i: i + chunk_size, :num_features] for i in valid_start_ilocs]
    chunk_station_names = [station_name] * len(chunks)
    end_days = [int(data_np[i + chunk_size - 1, -2]) for i in valid_start_ilocs]

    return chunks, chunk_station_names, end_days


class LSTMDataset(Dataset):
    """Unified dataset for LSTM pretraining and DADS graph assembly.

    LSTM Training Path (return_station_name=False):
    - __getitem__ returns (y, comparator, features)
      * y: target sequence (T,)
      * comparator: comparator sequence (T,) if available, otherwise NaN (optional)
      * features: single-channel lagged target (T, 1); univariate input to LSTM.
    - GEO/RS columns in Parquet are ignored as inputs.

    Dads Path (return_station_name=True):
    - __getitem__ returns (y, comparator, empty_seq, station_name, day_int)
      * empty_seq is a placeholder; DADS does not consume a target-node sequence.
      * station_name and day_int are used to fetch node contexts/edges.
    - Comparator optional in this path as well (comparator may be NaN).
    """
    def __init__(self, file_paths, station_names, feature_names, sample_dimensions, scaler, strided=False,
                 transform=None, return_station_name=False, n_workers=1):

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

        self._preload_data(file_paths, station_names, n_workers)

    def _preload_data(self, file_paths, station_names, n_workers):
        print("Reading and chunking data in parallel...")
        chunk_size_ = self.sample_dimensions[0]
        tasks_ = [(file_paths[i], station_names[i], chunk_size_, self.feature_names) for i in range(len(file_paths))]

        all_chunks_ = []
        new_station_names_ = []
        end_days_ = []

        if n_workers == 1:
            results_ = []
            for task in tasks_:
                r = _read_and_chunk_worker(task)
                results_.append(r)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results_ = list(tqdm(executor.map(_read_and_chunk_worker, tasks_),
                                     total=len(tasks_),
                                     desc="Chunking LSTM Inputs"))

        for chunks_, names_, days_ in results_:
            all_chunks_.extend(chunks_)
            new_station_names_.extend(names_)
            end_days_.extend(days_)

        self.preloaded_data = all_chunks_
        self.station_names = new_station_names_
        self.day_ints = end_days_
        print(f"Successfully preloaded and chunked {len(self.preloaded_data)} samples into memory.")

    def __len__(self):
        return len(self.preloaded_data)

    def __getitem__(self, idx):
        """Return a sample for the selected path.

        - Training: (y, comparator, features) where features is (T, 1) lagged target.
        - DADS: (y, comparator, empty_seq, station_name, day_int).
        """
        while True:
            chunk_np = self.preloaded_data[idx]
            station_name = self.station_names[idx]
            chunk = torch.tensor(chunk_np, dtype=torch.float32)
            if not torch.any(torch.isnan(chunk)):
                break
            idx = random.randint(0, len(self.preloaded_data) - 1)

        if self.scaler:
            chunk = torch.tensor(self.scaler.transform(chunk.numpy()), dtype=torch.float32)

        y = chunk[:, 0]
        if self.comp_idx is not None:
            comparator = chunk[:, self.comp_idx]
        else:
            comparator = torch.full_like(y, float('nan'))
        if self.return_station_name:
            # Dads path: omit target-node sequence to drop dependency on target meteorology
            seq = torch.empty(0)
            day_int = self.day_ints[idx]
            return y, comparator, seq, station_name, day_int  # Dads: no target sequence
        else:
            # LSTM training path: strictly univariate (observed target only)
            obs_feat = y.unsqueeze(-1)
            obs_shift = obs_feat.clone()
            if obs_shift.shape[0] > 1:
                obs_shift[1:, 0] = obs_feat[:-1, 0]
            obs_shift[0, 0] = obs_feat[0, 0]
            features = obs_shift  # single-channel lagged target
            return y, comparator, features


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
