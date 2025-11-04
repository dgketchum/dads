import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data._utils.collate import default_collate

from models.scalers import MinMaxScaler


class WeatherDataset(Dataset):
    """Yearly or sliding-window dataset for autoencoder training.

    Notes on performance
    - This dataset now avoids per-sample NumPy scaling to reduce CPU overhead.
      Scaling is performed on-device inside the model using preloaded scaler
      parameters, which significantly reduces Python/Numpy round-trips and
      speeds up training.
    - Optional contrastive triplet sampling can be disabled to avoid extra
      encoder passes when speed is preferred over representation shaping.
    """
    def __init__(self, file_paths, expected_width, col_indices, chunk_size,
                 scaler,
                 target_indices=None, transform=None, expected_columns=None, selected_indices=None,
                 num_workers=12, window_stride=None, triplet_sampling=True, max_samples=None,
                 max_files=None, split_name=None, tqdm_desc=None):

        self.chunk_size = chunk_size
        self.transform = transform
        self.col_indices = col_indices
        self.target_indices = target_indices
        self.expected_columns = expected_columns
        self.selected_indices = selected_indices
        self.num_workers = num_workers
        self.window_stride = int(window_stride) if window_stride is not None else int(chunk_size)
        self.triplet_sampling = bool(triplet_sampling)
        self._max_samples = int(max_samples) if max_samples is not None else None
        self._max_files = int(max_files) if max_files is not None else None
        self._split_name = split_name

        self.station_idx = []

        self.stations = []

        # Optionally restrict number of files processed (debug/fast-start)
        if self._max_files is not None:
            original_len = len(file_paths)
            file_paths = file_paths[: self._max_files]
            print(f"Debug mode: limiting {self._split_name or 'dataset'} files to first {len(file_paths)} of {original_len}")

        def load_one(file_path):
            staid = os.path.basename(file_path).replace('.parquet', '')
            df = pd.read_parquet(file_path)

            if self.expected_columns is not None:
                if not list(df.columns) == list(self.expected_columns):
                    print(f"Skipping {file_path}, columns mismatch. Expected {expected_width} columns, got {df.shape[1]}")
                    return staid, None, 0, 0

            assert isinstance(df.index, pd.DatetimeIndex)
            df = df.sort_index()
            # drop leap day to enforce 365-day alignment
            df = df[~((df.index.month == 2) & (df.index.day == 29))]
            if df.shape[1] != expected_width:
                print(f"Skipping {file_path}, shape mismatch. Expected {expected_width} columns, got {df.shape[1]}")
                return staid, None, 0, 0
            # build continuous daily index across the full record (Feb 29 removed)
            start = pd.Timestamp(df.index.min().year, df.index.min().month, df.index.min().day)
            end = pd.Timestamp(df.index.max().year, df.index.max().month, df.index.max().day)
            idx = pd.date_range(start, end, freq='D')
            idx = idx[~((idx.month == 2) & (idx.day == 29))]
            sub = df.reindex(idx)

            chunks = []
            dropped, kept = 0, 0

            # resolve target column index in the full frame order
            if self.selected_indices is not None:
                target_col_idx = self.selected_indices[self.target_indices[0]] if isinstance(self.target_indices, list) else self.selected_indices[0]
            else:
                target_col_idx = self.target_indices[0] if isinstance(self.target_indices, list) else 0

            # sliding windows over the entire record
            if len(sub) >= self.chunk_size:
                for start_i in range(0, len(sub) - self.chunk_size + 1, self.window_stride):
                    win = sub.iloc[start_i:start_i + self.chunk_size]
                    # require target coverage only (relax overall valid fraction)
                    t_vals = win.iloc[:, target_col_idx].to_numpy()
                    t_total = t_vals.size
                    t_valid = np.isfinite(t_vals).sum()
                    t_frac = t_valid / t_total if t_total > 0 else 0.0
                    if t_frac < 0.10:
                        dropped += 1
                        continue
                    arr = torch.as_tensor(win.values, dtype=torch.float32)
                    chunks.append(arr.unsqueeze(0))
                    kept += 1
            if not chunks:
                return staid, None, dropped, kept
            data = torch.cat(chunks, dim=0)
            return staid, data, dropped, kept

        all_data = []
        total_dropped, total_kept = 0, 0
        desc = tqdm_desc or f"AE[{self._split_name or 'data'}] files={len(file_paths)} T={self.chunk_size} stride={self.window_stride}"
        with ThreadPoolExecutor(max_workers=min(self.num_workers, len(file_paths) or 1)) as ex:
            results = list(tqdm(ex.map(load_one, file_paths), total=len(file_paths), desc=desc, unit='file'))

        stations_seen = set()
        for staid, data, dropped, kept in results:
            total_dropped += dropped
            total_kept += kept
            if data is None:
                continue
            stations_seen.add(staid)
            if staid not in self.stations:
                self.stations.append(staid)
            all_data.append(data)
            self.station_idx.extend([self.stations.index(staid)] * len(data))

        if not all_data:
            print('No valid chunks found after filtering. Dropped: {}, Kept: {}, Stations: {}'.format(total_dropped, total_kept, len(stations_seen)))
            raise ValueError('No valid chunks to train on.')
        self.data = torch.cat(all_data, dim=0)

        # Optionally truncate to a small number of samples for fast debug runs
        if self._max_samples is not None and self._max_samples < len(self.data):
            keep = max(1, self._max_samples)
            self.data = self.data[:keep]
            self.station_idx = self.station_idx[:keep]
            print(f"Debug mode: limiting dataset to first {keep} samples")

        self.scaler = scaler

        # map from station id to sample indices to enable in-station positive sampling
        self.station_to_indices = {}
        for i, s_idx in enumerate(self.station_idx):
            self.station_to_indices.setdefault(s_idx, []).append(i)

        print('Loaded stations: {}, valid chunks: {}, dropped chunks (<30% valid): {}'.format(len(stations_seen), total_kept, total_dropped))

    def get_positive_pair(self, idx):
        s_idx = self.station_idx[idx]
        candidates = self.station_to_indices.get(s_idx, [])
        candidates = [j for j in candidates if j != idx]
        if not candidates:
            return None
        j = random.choice(candidates)
        sample = self.data[j]
        return sample

    def get_negative_pair(self, idx):
        s_idx = self.station_idx[idx]
        all_indices = list(range(len(self.data)))
        candidates = [j for j in all_indices if self.station_idx[j] != s_idx]
        if not candidates:
            return None
        j = random.choice(candidates)
        sample = self.data[j]
        return sample

    def scale_chunk(self, chunk):
        # Deprecated: scaling now happens on-GPU in the model. Keeping this
        # method for backward compatibility, but it is no longer used.
        chunk_np = chunk.numpy()
        reshaped_chunk = chunk_np.reshape(-1, chunk_np.shape[-1])
        scaled_chunk = self.scaler.transform(reshaped_chunk)
        return torch.from_numpy(scaled_chunk.reshape(chunk_np.shape))

    def get_valid_data_for_scaling(self):
        valid_data = []
        for chunk in self.data:
            # align scaler fit columns with selected training inputs
            if self.selected_indices is not None:
                cols = self.selected_indices
                chunk_cols = chunk[:, cols]
            else:
                chunk_cols = chunk[:, :self.col_indices]
            valid_rows = chunk_cols[~torch.isnan(chunk_cols).any(dim=1)]
            valid_data.append(valid_rows)
        return torch.cat(valid_data, dim=0).numpy()

    def save_scaler(self, scaler_path):

        bias_ = self.scaler.bias.flatten().tolist()
        scale_ = self.scaler.scale.flatten().tolist()
        dct = {'bias': bias_, 'scale': scale_}
        with open(scaler_path, 'w') as fp:
            json.dump(dct, fp, indent=4)
        print(f"Scaler saved to {scaler_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return scaled inputs, masked targets, and optional contrastive pairs.

        Returns
        - chunk: (T, C_in) inputs for the autoencoder (selected indices only)
        - target: (T, C_tgt) columns to reconstruct (by target_indices)
        - mask: (T, C_tgt) boolean mask for valid target elements
        - positive_chunk / negative_chunk: same shape as chunk or None
        """

        # Return raw (unscaled) data; scaling occurs in the model on device.
        full = self.data[idx]
        if self.selected_indices is not None:
            chunk = full[:, self.selected_indices].clone()
        else:
            chunk = full[:, :self.col_indices].clone()

        if isinstance(self.target_indices[0], tuple):
            target = diff_pairs(chunk, self.target_indices)
        else:
            target = chunk[:, self.target_indices]

        # derive mask per target, not just first column
        if isinstance(self.target_indices[0], tuple):
            masks = []
            for i, j in self.target_indices:
                m = (~torch.isnan(chunk[:, i])) & (~torch.isnan(chunk[:, j]))
                masks.append(m.unsqueeze(1))
            mask = torch.cat(masks, dim=1)
        else:
            idxs = self.target_indices if isinstance(self.target_indices, list) else [self.target_indices]
            mask = ~torch.isnan(chunk[:, idxs])

        positive_chunk = self.get_positive_pair(idx) if self.triplet_sampling else None
        negative_chunk = self.get_negative_pair(idx) if self.triplet_sampling else None

        if positive_chunk is not None:
            if self.selected_indices is not None:
                positive_chunk = positive_chunk[:, self.selected_indices].clone()
            else:
                positive_chunk = positive_chunk[:, :self.col_indices].clone()

        if negative_chunk is not None:
            if self.selected_indices is not None:
                negative_chunk = negative_chunk[:, self.selected_indices].clone()
            else:
                negative_chunk = negative_chunk[:, :self.col_indices].clone()

        return chunk, target, mask, positive_chunk, negative_chunk


def diff_pairs(tensor, pairs):
    """Compute pairwise differences for tuple index pairs within a tensor."""
    diffs = []
    for i, j in pairs:
        diff = tensor[:, i] - tensor[:, j]
        diffs.append(diff.unsqueeze(-1))
    return torch.cat(diffs, dim=-1)


def custom_collate(batch):
    """Drop Nones yielded by workers (e.g., invalid files) before collation."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return default_collate(batch)


class WeatherIterableDataset(IterableDataset):
    """Streaming version of the autoencoder dataset.

    - Iterates files and yields sliding windows on-the-fly to avoid holding all
      windows in memory.
    - Only reads the selected input columns to reduce I/O.
    - Provides an estimated length for logging/progress via a cheap pass that
      only inspects index bounds from a single column per file.
    """

    def __init__(self, file_paths, expected_width, col_indices, chunk_size,
                 scaler,
                 target_indices=None, transform=None, expected_columns=None, selected_indices=None,
                 num_workers=12, window_stride=None, triplet_sampling=False, max_samples=None,
                 max_files=None, split_name=None, shuffle_files=True, tqdm_desc=None):

        super().__init__()
        self.file_paths = list(file_paths)
        self.expected_width = expected_width
        self.col_indices = col_indices
        self.chunk_size = int(chunk_size)
        self.transform = transform
        self.target_indices = target_indices if target_indices is not None else [0]
        self.expected_columns = expected_columns
        self.selected_indices = selected_indices
        self.num_workers = int(num_workers)
        self.window_stride = int(window_stride) if window_stride is not None else int(chunk_size)
        self.triplet_sampling = bool(triplet_sampling)
        self._max_samples = int(max_samples) if max_samples is not None else None
        self._max_files = int(max_files) if max_files is not None else None
        self._split_name = split_name or 'data'
        self._shuffle_files = bool(shuffle_files)
        self._tqdm_desc = tqdm_desc
        self.scaler = scaler

        # Effective file list (respect max_files if provided)
        if self._max_files is not None:
            original_len = len(self.file_paths)
            self.file_paths = self.file_paths[: self._max_files]
            print(f"Debug mode: limiting {self._split_name} files to first {len(self.file_paths)} of {original_len}")

        # Pre-compute selected column names to request minimal data from Parquet
        if self.expected_columns is not None and self.selected_indices is not None:
            self._selected_names = [self.expected_columns[i] for i in self.selected_indices]
        else:
            self._selected_names = None

        # Cached approximate length
        self._approx_len = None

    def _worker_file_slice(self, all_paths):
        info = torch.utils.data.get_worker_info()
        if info is None or info.num_workers is None or info.num_workers <= 1:
            return all_paths
        # split contiguous chunks across workers
        per_worker = int((len(all_paths) + info.num_workers - 1) / info.num_workers)
        start = info.id * per_worker
        end = min(start + per_worker, len(all_paths))
        return all_paths[start:end]

    def _iter_file(self, file_path):
        def _synthesize_missing_flags(df_local):
            # Ensure any *_miss columns exist. If base present, flag = isna(base)
            # If base missing, set flag=1.0 (conservative).
            if not self._selected_names:
                return df_local
            for name in self._selected_names:
                if name.endswith('_miss'):
                    base = name[:-5]
                    if name not in df_local.columns:
                        if base in df_local.columns:
                            df_local[name] = df_local[base].isna().astype(np.float32)
                        else:
                            df_local[name] = 1.0
            return df_local

        # Read only needed columns to reduce I/O; on failure, fall back to
        # reading minimal bases and synthesizing *_miss flags.
        try:
            df = pd.read_parquet(file_path, columns=self._selected_names)
        except Exception:
            # Build a minimal set of columns to request
            needed = set()
            if self._selected_names:
                for n in self._selected_names:
                    if n.endswith('_miss'):
                        needed.add(n[:-5])  # base column to derive miss flag
                    else:
                        needed.add(n)
            try:
                df = pd.read_parquet(file_path, columns=list(needed) if needed else None)
            except Exception as e2:
                # As a last resort, read all columns; if this fails, skip file
                try:
                    df = pd.read_parquet(file_path)
                except Exception:
                    print(f"Skipping {file_path}, failed to read selected columns: {e2}")
                    return
            # Synthesize missingness flags and any missing selected columns
            df = _synthesize_missing_flags(df)
            # Create any other missing selected columns as NaN
            if self._selected_names:
                for n in self._selected_names:
                    if n not in df.columns:
                        df[n] = np.nan
            # Reorder to selected order
            if self._selected_names:
                df = df[self._selected_names]

        # Basic schema sanity: ensure all requested columns arrived
        if self._selected_names is not None and list(df.columns) != list(self._selected_names):
            # Try to recover by synthesizing missingness flags if only those are absent
            df = _synthesize_missing_flags(df)
            for n in self._selected_names:
                if n not in df.columns:
                    # create placeholder column for stability
                    df[n] = np.nan
            try:
                df = df[self._selected_names]
            except Exception:
                print(f"Skipping {file_path}, columns mismatch for selected inputs.")
                return

        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"Skipping {file_path}, index is not DatetimeIndex")
            return

        df = df.sort_index()
        # drop leap day to enforce 365-day alignment
        df = df[~((df.index.month == 2) & (df.index.day == 29))]

        # build continuous daily index across the full record (Feb 29 removed)
        start = pd.Timestamp(df.index.min().year, df.index.min().month, df.index.min().day)
        end = pd.Timestamp(df.index.max().year, df.index.max().month, df.index.max().day)
        idx = pd.date_range(start, end, freq='D')
        idx = idx[~((idx.month == 2) & (idx.day == 29))]
        sub = df.reindex(idx)

        # Sanity on shape
        if self._selected_names is not None and sub.shape[1] != len(self._selected_names):
            return

        # Resolve target column indices within the selected feature order
        if isinstance(self.target_indices, list) and len(self.target_indices) > 0:
            tgt_idxs = self.target_indices
        else:
            tgt_idxs = [0]

        # Sliding window yield
        total = len(sub)
        if total < self.chunk_size:
            return

        # target coverage threshold
        min_frac = 0.10
        values = sub.values  # (T, C_in)
        for start_i in range(0, total - self.chunk_size + 1, self.window_stride):
            win = values[start_i:start_i + self.chunk_size]
            # Compute target valid fraction across the first target channel
            t_vals = win[:, tgt_idxs[0]]
            t_total = t_vals.size
            t_valid = np.isfinite(t_vals).sum()
            t_frac = t_valid / t_total if t_total > 0 else 0.0
            if t_frac < min_frac:
                continue

            chunk = torch.as_tensor(win, dtype=torch.float32)
            if isinstance(self.target_indices[0], tuple):
                target = diff_pairs(chunk, self.target_indices)
                # mask: valid only when both elements in each pair are finite
                masks = []
                for i, j in self.target_indices:
                    m = (~torch.isnan(chunk[:, i])) & (~torch.isnan(chunk[:, j]))
                    masks.append(m.unsqueeze(1))
                mask = torch.cat(masks, dim=1)
            else:
                mask = ~torch.isnan(chunk[:, tgt_idxs])
                target = chunk[:, tgt_idxs]

            yield chunk, target, mask, None, None

    def __iter__(self):
        # Choose and optionally shuffle the file sequence for this iterator
        paths = list(self.file_paths)
        if self._shuffle_files:
            random.Random().shuffle(paths)
        paths = self._worker_file_slice(paths)

        produced = 0
        for fp in paths:
            for sample in self._iter_file(fp):
                yield sample
                produced += 1
                if self._max_samples is not None and produced >= self._max_samples:
                    return

    def __len__(self):
        # Cheap approximate length: assume continuous daily coverage between
        # first and last timestamp; ignore target coverage filtering.
        if self._approx_len is not None:
            return self._approx_len

        total_windows = 0
        # pick a minimal column to read (first selected)
        col_name = None
        if self._selected_names:
            col_name = self._selected_names[0]
        elif self.expected_columns:
            col_name = self.expected_columns[0]

        for fp in self.file_paths:
            try:
                df = pd.read_parquet(fp, columns=[col_name] if col_name is not None else None)
                if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
                    continue
                start = df.index.min()
                end = df.index.max()
                # inclusive day count; skip Feb 29 approximately by ignoring
                days = (end.normalize() - start.normalize()).days + 1
                if days >= self.chunk_size:
                    n_win = 1 + (days - self.chunk_size) // self.window_stride
                    if n_win > 0:
                        total_windows += int(n_win)
            except Exception:
                continue

        # Respect max_samples cap if provided
        if self._max_samples is not None:
            total_windows = min(total_windows, self._max_samples)

        self._approx_len = int(total_windows)
        return self._approx_len


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
