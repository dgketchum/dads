"""
ZarrWeatherDataset — autoencoder dataset that reads yearly 365-day windows
from stations.zarr instead of per-station parquets.

Target obs come from stations.zarr.  Exog features still come from per-station
parquets (LRU cached) until they migrate to cube.zarr.

Train/val split uses graph.zarr/is_train (consistent with GNN pipeline).
"""

import os
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset


class ZarrWeatherDataset(Dataset):
    """Autoencoder dataset backed by stations.zarr + per-station parquets.

    Returns
    -------
    (chunk, target, mask, positive_chunk, negative_chunk, station_idx)
        - chunk:  (T, C_in) float32 — raw unscaled
        - target: (T, 1) float32 — column 0 of chunk
        - mask:   (T, 1) bool — True where target is finite
        - positive_chunk / negative_chunk: (T, C_in) or None
        - station_idx: int
    """

    def __init__(
        self,
        stations_zarr,
        graph_zarr,
        target_variable,
        parquet_root,
        columns,
        chunk_size=365,
        split="train",
        window_stride=None,
        min_target_frac=0.8,
        triplet_sampling=False,
        windows_per_station=None,
        max_stations=None,
        num_workers=4,
    ):
        self.chunk_size = int(chunk_size)
        self.window_stride = (
            int(window_stride) if window_stride is not None else self.chunk_size
        )
        self.min_target_frac = float(min_target_frac)
        self.triplet_sampling = bool(triplet_sampling)
        self.max_windows_per_station = (
            None if windows_per_station is None else int(windows_per_station)
        )
        self._target_variable = target_variable
        self._columns = list(columns)  # [target_var, exog1, exog2, ...]

        # Compatibility: train.py expects dataset.scaler
        self.scaler = None

        # ---- Open zarr stores (read-only) ----
        self._sz = zarr.open(str(stations_zarr), mode="r")
        self._gz = zarr.open(str(graph_zarr), mode="r")

        # ---- Station metadata ----
        gz_ids = list(self._gz["station_id"][:])
        sz_ids = list(self._sz["station_id"][:])
        self._gz_ids = gz_ids
        self._sz_ids = sz_ids
        self._sz_sid_to_idx = {s: i for i, s in enumerate(sz_ids)}

        is_train = self._gz["is_train"][:]

        # ---- Time axis ----
        self._time_days = self._sz["time"][:]  # int64, days since epoch
        self._n_times = len(self._time_days)

        # Precompute filtered time indices (exclude Feb 29)
        epoch_dates = pd.to_datetime(self._time_days, unit="D", origin="unix")
        not_feb29 = ~((epoch_dates.month == 2) & (epoch_dates.day == 29))
        self._filtered_tidx = np.where(not_feb29)[0]  # indices into zarr time axis
        self._n_filtered = len(self._filtered_tidx)

        # Map filtered position -> epoch day for parquet alignment
        self._filtered_epoch_days = self._time_days[self._filtered_tidx]

        # ---- Train/val split ----
        if split == "train":
            split_mask = is_train
        elif split == "val":
            split_mask = ~is_train
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        # Map graph station IDs to stations.zarr indices
        split_gz_indices = np.where(split_mask)[0]
        split_station_indices = []
        for gi in split_gz_indices:
            sid = gz_ids[gi]
            if sid in self._sz_sid_to_idx:
                split_station_indices.append(self._sz_sid_to_idx[sid])
        split_station_indices = np.array(split_station_indices, dtype=np.int64)

        if max_stations is not None:
            split_station_indices = split_station_indices[: int(max_stations)]

        self._split_station_indices = split_station_indices

        # ---- Parquet exog setup ----
        self._parquet_root = parquet_root
        self._file_map = {}
        target_dir = os.path.join(parquet_root, target_variable)
        if os.path.isdir(target_dir):
            for fname in os.listdir(target_dir):
                if fname.endswith(".parquet"):
                    fid = os.path.splitext(fname)[0]
                    self._file_map[fid] = os.path.join(target_dir, fname)

        self._exog_cache = OrderedDict()
        self._exog_cache_max = max(256, min(4096, len(self._file_map) or 256))

        # ---- Build index ----
        self._build_index()

        # Expose station list for compatibility
        self.stations = [self._sz_ids[si] for si in split_station_indices]

        # Station-to-sample-indices map for triplet sampling
        self._station_to_indices = {}
        for i, (si, _) in enumerate(self.index):
            self._station_to_indices.setdefault(si, []).append(i)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self):
        """Scan stations.zarr target column for windows meeting min_target_frac."""
        target_arr = self._sz[self._target_variable]  # (T, S) zarr array
        rng = np.random.default_rng(42)

        index_ = []
        for station_idx in self._split_station_indices:
            station_idx = int(station_idx)
            # Read target values at filtered time positions
            col_full = target_arr[:, station_idx]
            col = col_full[self._filtered_tidx]  # values at non-Feb29 positions

            if len(col) < self.chunk_size:
                continue

            station_windows = []
            for start_pos in range(
                0, len(col) - self.chunk_size + 1, self.window_stride
            ):
                window = col[start_pos : start_pos + self.chunk_size]
                n_finite = np.isfinite(window).sum()
                frac = n_finite / self.chunk_size
                if frac >= self.min_target_frac:
                    station_windows.append((station_idx, start_pos))

            if (
                self.max_windows_per_station is not None
                and len(station_windows) > self.max_windows_per_station
            ):
                sel = rng.choice(
                    len(station_windows),
                    size=self.max_windows_per_station,
                    replace=False,
                )
                station_windows = [station_windows[int(i)] for i in sel]

            index_.extend(station_windows)

        self.index = index_

    # ------------------------------------------------------------------
    # Exog helpers
    # ------------------------------------------------------------------

    def _get_exog_cached(self, station_id):
        """Return {epoch_day: row_array} dict for exog columns, LRU cached."""
        if station_id in self._exog_cache:
            d = self._exog_cache.pop(station_id)
            self._exog_cache[station_id] = d
            return d

        exog_cols = self._columns[1:]  # everything except target
        fp = self._file_map.get(station_id)
        result = {}

        if fp is not None and os.path.exists(fp):
            try:
                # Read available columns, synthesize missing ones
                df = pd.read_parquet(fp)
                if isinstance(df.index, pd.DatetimeIndex):
                    # Drop Feb 29
                    df = df[~((df.index.month == 2) & (df.index.day == 29))]
                    # Synthesize missing flags
                    df = self._synthesize_missing_flags(df, exog_cols)
                    # Ensure all requested columns exist
                    for c in exog_cols:
                        if c not in df.columns:
                            df[c] = np.nan
                    sub = df[exog_cols]
                    # Build epoch_day -> row dict
                    epoch_days = (
                        (df.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1D")
                    ).astype(np.int64)
                    values = sub.values.astype(np.float32)
                    for i, ed in enumerate(epoch_days):
                        result[int(ed)] = values[i]
            except Exception:
                pass

        self._exog_cache[station_id] = result
        if len(self._exog_cache) > self._exog_cache_max:
            self._exog_cache.popitem(last=False)
        return result

    @staticmethod
    def _synthesize_missing_flags(df, needed_cols):
        """Synthesize *_miss columns if absent, following WeatherIterableDataset."""
        for name in needed_cols:
            if name.endswith("_miss") and name not in df.columns:
                base = name[:-5]
                if base in df.columns:
                    df[name] = df[base].isna().astype(np.float32)
                else:
                    df[name] = np.float32(1.0)
        return df

    # ------------------------------------------------------------------
    # Chunk assembly
    # ------------------------------------------------------------------

    def _get_item_no_triplet(self, idx):
        """Assemble a single (chunk, target, mask, station_idx) without triplets."""
        station_idx, start_pos = self.index[idx]
        station_id = self._sz_ids[station_idx]

        # ---- Target from zarr ----
        zarr_tidx = self._filtered_tidx[start_pos : start_pos + self.chunk_size]
        target_raw = self._sz[self._target_variable][
            int(zarr_tidx[0]) : int(zarr_tidx[-1]) + 1, station_idx
        ]
        # Sub-select only non-Feb29 positions (handle leap years in the span)
        offsets = zarr_tidx - int(zarr_tidx[0])
        target_col = target_raw[offsets].astype(np.float32)

        # ---- Exog from parquet ----
        exog_cols = self._columns[1:]
        n_exog = len(exog_cols)
        exog_data = np.full((self.chunk_size, n_exog), np.nan, dtype=np.float32)

        exog_dict = self._get_exog_cached(station_id)
        epoch_days = self._filtered_epoch_days[start_pos : start_pos + self.chunk_size]
        for t, ed in enumerate(epoch_days):
            row = exog_dict.get(int(ed))
            if row is not None:
                exog_data[t] = row

        # ---- Assemble chunk: [target, exog...] ----
        chunk_np = np.empty((self.chunk_size, 1 + n_exog), dtype=np.float32)
        chunk_np[:, 0] = target_col
        chunk_np[:, 1:] = exog_data

        chunk = torch.from_numpy(chunk_np)
        target = chunk[:, 0:1]
        mask = torch.isfinite(target)

        return chunk, target, mask, station_idx

    # ------------------------------------------------------------------
    # Triplet helpers
    # ------------------------------------------------------------------

    def _get_positive(self, idx):
        """Return a chunk from the same station (different window)."""
        station_idx = self.index[idx][0]
        candidates = self._station_to_indices.get(station_idx, [])
        candidates = [j for j in candidates if j != idx]
        if not candidates:
            return None
        j = random.choice(candidates)
        chunk, _, _, _ = self._get_item_no_triplet(j)
        return chunk

    def _get_negative(self, idx):
        """Return a chunk from a different station."""
        station_idx = self.index[idx][0]
        all_stations = [si for si in self._station_to_indices if si != station_idx]
        if not all_stations:
            return None
        neg_station = random.choice(all_stations)
        j = random.choice(self._station_to_indices[neg_station])
        chunk, _, _, _ = self._get_item_no_triplet(j)
        return chunk

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        chunk, target, mask, station_idx = self._get_item_no_triplet(idx)

        if self.triplet_sampling:
            positive_chunk = self._get_positive(idx)
            negative_chunk = self._get_negative(idx)
        else:
            positive_chunk = None
            negative_chunk = None

        return chunk, target, mask, positive_chunk, negative_chunk, station_idx
