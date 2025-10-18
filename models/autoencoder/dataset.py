import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from models.scalers import MinMaxScaler


class WeatherDataset(Dataset):
    """Yearly-chunk dataset for autoencoder training.

    Intent
    - Enforce a unified column schema across all stations/years ("expected_columns").
    - Build 365-day chunks per year (Feb 29 removed) to learn seasonal/annual structure.
    - Inputs are a selected subset of the Parquet columns ("selected_indices").
      These typically include the observed target, GEO features, and RS missingness flags
      so the encoder is mask-aware of RS availability.
    - Targets are one or more columns specified by "target_indices" (by position in the
      selected input order); reconstruction is computed only where masks are valid.

    Notes
    - Scaling is fit/applied on the selected input columns.
    - Chunks with <30% valid target coverage are dropped (other channels may be sparse).
    - Positive/negative pairs are optional and sampled within/between stations for the
      triplet loss used during training.
    """
    def __init__(self, file_paths, expected_width, col_indices, chunk_size,
                 scaler,
                 target_indices=None, transform=None, expected_columns=None, selected_indices=None,
                 num_workers=12):

        self.chunk_size = chunk_size
        self.transform = transform
        self.col_indices = col_indices
        self.target_indices = target_indices
        self.expected_columns = expected_columns
        self.selected_indices = selected_indices
        self.num_workers = num_workers

        self.station_idx = []

        self.stations = []

        def load_one(file_path):
            staid = os.path.basename(file_path).replace('.parquet', '')
            df = pd.read_parquet(file_path)
            if self.expected_columns is not None:
                assert list(df.columns) == list(self.expected_columns)
            assert isinstance(df.index, pd.DatetimeIndex)
            df = df.sort_index()
            # drop leap day to enforce 365-day alignment
            df = df[~((df.index.month == 2) & (df.index.day == 29))]
            if df.shape[1] != expected_width:
                print(f"Skipping {file_path}, shape mismatch. Expected {expected_width} columns, got {df.shape[1]}")
                return staid, None, 0, 0
            years = range(df.index.min().year, df.index.max().year + 1)
            chunks = []
            dropped, kept = 0, 0
            for y in years:
                start = pd.Timestamp(y, 1, 1)
                end = pd.Timestamp(y, 12, 31)
                idx = pd.date_range(start, end, freq='D')
                idx = idx[~((idx.month == 2) & (idx.day == 29))]
                sub = df.reindex(idx)
                if len(sub) != self.chunk_size:
                    continue
                if self.selected_indices is not None:
                    sub_vals = sub.iloc[:, self.selected_indices].to_numpy()
                    target_col_idx = self.selected_indices[self.target_indices[0]] if isinstance(self.target_indices, list) else self.selected_indices[0]
                else:
                    sub_vals = sub.iloc[:, :self.col_indices].to_numpy()
                    target_col_idx = self.target_indices[0] if isinstance(self.target_indices, list) else 0
                total = sub_vals.size
                valid = np.isfinite(sub_vals).sum()
                frac = valid / total if total > 0 else 0.0
                # require target coverage as well
                t_vals = sub.iloc[:, target_col_idx].to_numpy()
                t_total = t_vals.size
                t_valid = np.isfinite(t_vals).sum()
                t_frac = t_valid / t_total if t_total > 0 else 0.0
                if frac < 0.30 or t_frac < 0.30:
                    dropped += 1
                    continue
                arr = torch.as_tensor(sub.values, dtype=torch.float32)
                chunks.append(arr.unsqueeze(0))
                kept += 1
            if not chunks:
                return staid, None, dropped, kept
            data = torch.cat(chunks, dim=0)
            return staid, data, dropped, kept

        all_data = []
        total_dropped, total_kept = 0, 0
        with ThreadPoolExecutor(max_workers=min(self.num_workers, len(file_paths) or 1)) as ex:
            results = list(tqdm(ex.map(load_one, file_paths), total=len(file_paths)))

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

        full = self.data[idx].clone()
        full[:, :] = self.scale_chunk(full)
        if self.selected_indices is not None:
            chunk = full[:, self.selected_indices]
        else:
            chunk = full[:, :self.col_indices]

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

        positive_chunk = self.get_positive_pair(idx)
        negative_chunk = self.get_negative_pair(idx)

        if positive_chunk is not None:
            pos_full = positive_chunk.clone()
            pos_full[:, :] = self.scale_chunk(pos_full)
            if self.selected_indices is not None:
                positive_chunk = pos_full[:, self.selected_indices]
            else:
                positive_chunk = pos_full[:, :self.col_indices]

        if negative_chunk is not None:
            neg_full = negative_chunk.clone()
            neg_full[:, :] = self.scale_chunk(neg_full)
            if self.selected_indices is not None:
                negative_chunk = neg_full[:, self.selected_indices]
            else:
                negative_chunk = neg_full[:, :self.col_indices]

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


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
