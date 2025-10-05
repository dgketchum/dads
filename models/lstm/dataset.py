import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def _read_and_chunk_worker(args):
    file_path, station_name, chunk_size, feature_names = args
    try:
        df = pd.read_parquet(file_path)[feature_names]
        df.dropna(inplace=True)
        if len(df) < chunk_size:
            return [], []
    except Exception:
        return [], []

    df['day_int'] = df.index.to_julian_date().astype(np.int32)
    df['day_diff'] = df['day_int'].diff()
    data_np = df.to_numpy(dtype=np.float32)

    diff_col_idx = -1
    is_consecutive = (data_np[1:, diff_col_idx] == 1.0)

    window_size = chunk_size - 1
    if len(is_consecutive) < window_size:
        return [], []

    convolved = np.convolve(is_consecutive, np.ones(window_size, dtype=int), mode='valid')
    valid_start_ilocs = np.where(convolved == window_size)[0]

    if valid_start_ilocs.size == 0:
        return [], []

    num_features = len(feature_names)
    chunks = [data_np[i: i + chunk_size, :num_features] for i in valid_start_ilocs]
    chunk_station_names = [station_name] * len(chunks)
    end_days = [int(data_np[i + chunk_size - 1, -2]) for i in valid_start_ilocs]

    return chunks, chunk_station_names, end_days


class LSTMDataset(Dataset):
    def __init__(self, file_paths, station_names, feature_names, sample_dimensions, scaler, strided=False,
                 transform=None, return_station_name=False, n_workers=1):

        self.sample_dimensions = sample_dimensions
        self.transform = transform
        self.feature_names = feature_names
        self.return_station_name = return_station_name
        self.strided = strided
        self.scaler = scaler
        self._preload_data(file_paths, station_names, n_workers)

    def _preload_data(self, file_paths, station_names, n_workers):
        print("Reading and chunking data in parallel...")
        chunk_size_ = self.sample_dimensions[0]
        tasks_ = [(file_paths[i], station_names[i], chunk_size_, self.feature_names) for i in range(len(file_paths))]

        all_chunks_ = []
        new_station_names_ = []
        end_days_ = []

        if n_workers == 1:
            results = []
            for task in tasks_:
                r = _read_and_chunk_worker(task)
                results.append(r)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results_ = list(tqdm(executor.map(_read_and_chunk_worker, tasks_),
                                 total=len(tasks_),
                                 desc="Processing files"))

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
        gm = chunk[:, 1]
        # Use only past observations as features; ignore reanalysis columns
        obs_feat = y.unsqueeze(-1)
        obs_shift = obs_feat.clone()
        obs_shift[1:, 0] = obs_feat[:-1, 0]
        obs_shift[0, 0] = obs_feat[0, 0]
        input_width = chunk.shape[1] - 2
        lf = obs_shift.repeat(1, input_width)

        if self.return_station_name:
            day_int = self.day_ints[idx]
            return y, gm, lf, station_name, day_int
        else:
            return y, gm, lf


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
