"""
ZarrStationDataset — reads obs from stations.zarr, graph from graph.zarr,
exog features from per-station parquets, and emits the identical batch
contract as DadsDataset: (graph, y, neighbor_seq, neighbor_mask, target_seq).

Embeddings are optional: when unavailable, emb_dim=0 and neighbor node
features become [exog_today, obs_today].
"""

import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import zarr
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from models.components.scalers import MinMaxScaler
from prep.columns_desc import CDR_FEATURES, CDR_MISS_FEATURES, TERRAIN_FEATURES

# Julian Day epoch offset: JD of 1970-01-01
_JD_EPOCH = 2440588


class ZarrStationDataset(Dataset):
    """Station-level dataset backed by zarr stores.

    Data sources
    ------------
    - Target obs (y): ``stations.zarr/{target_variable}[time, station]``
    - Neighbor obs:   ``stations.zarr/{target_variable}[time, neighbor]``
    - Graph:          ``graph.zarr`` arrays (neighbor_idx, edge_attr, etc.)
    - Train/val:      ``graph.zarr/is_train``
    - Embeddings:     Optional ``embeddings.json``
    - Scaler:         Variable scaler JSON (MinMaxScaler params)
    - Exog features:  Per-station parquets (LRU cached)

    Returns
    -------
    (graph, y, neighbor_seq, neighbor_mask, target_seq)
        Same contract as ``DadsDataset`` for ``DadsMetGNN``.
    """

    def __init__(
        self,
        stations_zarr,
        graph_zarr,
        target_variable,
        scaler_path,
        parquet_root,
        n_nodes=10,
        seq_len=12,
        split="train",
        embedding_path=None,
        windows_per_station=None,
        num_workers=4,
    ):
        self.n_nodes = n_nodes
        self.seq_len = seq_len

        # ---- Open zarr stores (read-only) ----
        self._sz = zarr.open(str(stations_zarr), mode="r")
        self._gz = zarr.open(str(graph_zarr), mode="r")

        # ---- Station / graph metadata ----
        station_ids = self._gz["station_id"][:]
        self._station_ids = list(station_ids)
        self._sid_to_idx = {s: i for i, s in enumerate(self._station_ids)}
        is_train = self._gz["is_train"][:]

        # Time axis from stations.zarr (days since epoch)
        self._time_days = self._sz["time"][:]  # int64
        self._n_times = len(self._time_days)
        # Build reverse lookup: day -> time index
        self._day_to_tidx = {int(d): i for i, d in enumerate(self._time_days)}

        # ---- Graph arrays (load fully — they're small) ----
        self._neighbor_idx = self._gz["neighbor_idx"][:]  # (S, K)
        self._node_attr = self._gz["node_attr"][:]  # (S, F_node)
        self._edge_attr = self._gz["edge_attr"][:]  # (S, K, F_edge)
        self._distance_km = self._gz["distance_km"][:]  # (S, K)
        self._bearing_sin = self._gz["bearing_sin"][:]  # (S, K)
        self._bearing_cos = self._gz["bearing_cos"][:]  # (S, K)

        # ---- Train/val split ----
        if split == "train":
            self._station_mask = is_train
        elif split == "val":
            self._station_mask = ~is_train
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        self._split_indices = np.where(self._station_mask)[0]

        # ---- Scaler ----
        with open(scaler_path, "r") as f:
            scaler_params = json.load(f)
        self._scaler_bias = np.array(scaler_params["bias"], dtype=np.float32).reshape(
            -1
        )
        self._scaler_scale = np.array(scaler_params["scale"], dtype=np.float32).reshape(
            -1
        )
        self.feature_names = scaler_params["feature_names"]

        # Full MinMaxScaler object for DadsMetGNN.inverse_transform()
        scaler_obj = MinMaxScaler()
        scaler_obj.bias = np.array(scaler_params["bias"]).reshape(1, -1)
        scaler_obj.scale = np.array(scaler_params["scale"]).reshape(1, -1)
        self.scaler = scaler_obj

        # Target variable scaling (column 0 of the scaler)
        assert self.feature_names[0] == target_variable, (
            f"scaler feature_names[0]={self.feature_names[0]} != {target_variable}"
        )
        self._y_bias = float(self._scaler_bias[0])
        self._y_scale = float(self._scaler_scale[0])
        self._target_variable = target_variable

        # ---- Embeddings (optional) ----
        if embedding_path is not None and os.path.exists(embedding_path):
            with open(embedding_path, "r") as f:
                emb_raw = json.load(f)
            raw = {}
            for k, v in emb_raw.items():
                t = torch.tensor(v)
                if t.dim() == 2 and t.shape[-1] == 1:
                    t = t.squeeze(-1)
                elif t.dim() > 1:
                    t = t.view(-1)
                raw[k] = t
            # MinMax normalize across all stations
            all_tensors = torch.stack(list(raw.values()), dim=0)
            emin = all_tensors.min(dim=0).values
            emax = all_tensors.max(dim=0).values
            denom = emax - emin
            denom[denom == 0] = 1.0
            self._embeddings = {k: (t - emin) / denom for k, t in raw.items()}
            self.emb_dim = int(next(iter(self._embeddings.values())).shape[-1])
        else:
            self._embeddings = None
            self.emb_dim = 0

        # ---- Sequence channel setup ----
        # TCN channels: [target_obs(lagged), rsun, CDR(SR1-3,BT1-3), CDR_miss, doy_sin, doy_cos]
        seq_names = (
            ["rsun"]
            + list(CDR_FEATURES)
            + list(CDR_MISS_FEATURES)
            + ["doy_sin", "doy_cos"]
        )
        self._seq_exog_idx = [
            i for i, n in enumerate(self.feature_names) if n in seq_names
        ]
        self._seq_selected_indices = [0] + self._seq_exog_idx
        self._seq_selected_columns = [
            self.feature_names[i] for i in self._seq_selected_indices
        ]
        self._seq_bias = self._scaler_bias[self._seq_selected_indices]
        self._seq_scale = self._scaler_scale[self._seq_selected_indices]
        self.seq_in_channels = len(self._seq_selected_indices)

        # ---- Node exog setup ----
        allowed_exog = (
            ["rsun"]
            + list(CDR_FEATURES)
            + list(CDR_MISS_FEATURES)
            + list(TERRAIN_FEATURES)
            + ["doy_sin", "doy_cos"]
        )
        self._exog_cols = [c for c in allowed_exog if c in self.feature_names]
        self._exog_idx = [self.feature_names.index(c) for c in self._exog_cols]
        self.exog_dim = len(self._exog_cols)
        if self._exog_cols:
            self._exog_bias = self._scaler_bias[self._exog_idx]
            self._exog_scale = self._scaler_scale[self._exog_idx]
        else:
            self._exog_bias = np.empty((0,), dtype=np.float32)
            self._exog_scale = np.ones((0,), dtype=np.float32)

        # ---- Edge dimension ----
        base_edge_dim = self._edge_attr.shape[-1]
        # +2 bearing (sin/cos), +1 distance, + exog_dim dynamic deltas
        self.edge_dim = base_edge_dim + 2 + 1 + self.exog_dim

        # ---- column_indices for model compatibility ----
        num_features = len(self.feature_names)
        self.column_indices = (0, 1, 2, num_features)

        # ---- Parquet file map for exog reads ----
        self._parquet_root = parquet_root
        self._file_map = {}
        target_dir = os.path.join(parquet_root, target_variable)
        if os.path.isdir(target_dir):
            for fname in os.listdir(target_dir):
                if fname.endswith(".parquet"):
                    fid = os.path.splitext(fname)[0]
                    self._file_map[fid] = os.path.join(target_dir, fname)

        # ---- Probe parquet schema to find available columns ----
        self._pq_columns = None
        if self._file_map:
            sample_fp = next(iter(self._file_map.values()))
            schema = pq.read_schema(sample_fp)
            self._pq_columns = set(schema.names)

        # Build column presence masks for seq and exog reads
        if self._pq_columns is not None:
            self._seq_present = [
                c in self._pq_columns for c in self._seq_selected_columns
            ]
            self._seq_read_cols = [
                c for c in self._seq_selected_columns if c in self._pq_columns
            ]
            self._exog_present = [c in self._pq_columns for c in self._exog_cols]
            self._exog_read_cols = [c for c in self._exog_cols if c in self._pq_columns]
        else:
            self._seq_present = [True] * len(self._seq_selected_columns)
            self._seq_read_cols = list(self._seq_selected_columns)
            self._exog_present = [True] * len(self._exog_cols)
            self._exog_read_cols = list(self._exog_cols)

        # ---- LRU caches ----
        self._seq_cache = OrderedDict()
        self._seq_cache_max = max(256, min(4096, len(self._file_map)))
        self._exog_cache = OrderedDict()
        self._exog_cache_max = max(256, min(4096, len(self._file_map)))

        # ---- Build index ----
        self.max_windows_per_station = (
            None if windows_per_station is None else int(windows_per_station)
        )
        self._build_index(num_workers)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self, n_workers):
        """Scan stations.zarr target column for contiguous non-NaN windows."""
        target_arr = self._sz[self._target_variable]  # (T, S) zarr array

        tasks = []
        for station_idx in self._split_indices:
            tasks.append((int(station_idx), target_arr, self.seq_len))

        results = []
        if n_workers <= 1:
            for t in tqdm(tasks, desc="Indexing windows"):
                results.append(self._scan_station_windows(t))
        else:
            # zarr arrays aren't pickle-safe for multiprocess, use threads
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                results = list(
                    tqdm(
                        ex.map(self._scan_station_windows, tasks),
                        total=len(tasks),
                        desc="Indexing windows",
                    )
                )

        rng = np.random.default_rng(42)
        index_ = []
        for station_idx, end_time_indices in results:
            if not end_time_indices:
                continue
            if (
                self.max_windows_per_station is not None
                and len(end_time_indices) > self.max_windows_per_station
            ):
                sel = rng.choice(
                    len(end_time_indices),
                    size=self.max_windows_per_station,
                    replace=False,
                )
                end_time_indices = [end_time_indices[int(i)] for i in sel]
            for end_tidx in end_time_indices:
                index_.append((station_idx, end_tidx))

        self.index = index_

    def _scan_station_windows(self, task):
        """Find contiguous non-NaN runs of length seq_len for one station."""
        station_idx, target_arr, seq_len = task

        # Read full time series for this station
        col = target_arr[:, station_idx]
        valid = ~np.isnan(col)

        if valid.sum() < seq_len:
            return station_idx, []

        # Convert to consecutive-day check using time axis
        valid_idx = np.where(valid)[0]
        days = self._time_days[valid_idx]  # epoch days for valid entries
        # Check consecutive days
        is_consec = np.diff(days) == 1
        win = seq_len - 1
        if len(is_consec) < win:
            return station_idx, []

        conv = np.convolve(is_consec, np.ones(win, dtype=int), mode="valid")
        starts = np.where(conv == win)[0]
        # Map back to time axis indices
        end_time_indices = [int(valid_idx[s + win]) for s in starts]
        return station_idx, end_time_indices

    # ------------------------------------------------------------------
    # Caching helpers (parquet exog reads)
    # ------------------------------------------------------------------

    def _get_seq_cached(self, fp):
        """Return (arr[T, C_seq], jd_idx[T]) for sequence columns, scaled."""
        if fp in self._seq_cache:
            arr, jd_idx = self._seq_cache.pop(fp)
            self._seq_cache[fp] = (arr, jd_idx)
            return arr, jd_idx
        n_cols = len(self._seq_selected_columns)
        df = pd.read_parquet(fp, columns=self._seq_read_cols)
        # Target column is always first; drop rows where target is NaN
        if self._seq_read_cols:
            df.dropna(subset=[self._seq_read_cols[0]], inplace=True)
        if df.empty:
            arr = np.zeros((0, n_cols), dtype=np.float32)
            jd_idx = np.zeros((0,), dtype=np.int32)
        else:
            jd_idx = df.index.to_julian_date().astype(np.int32).to_numpy()
            n_rows = len(df)
            if all(self._seq_present):
                arr = df.to_numpy(dtype=np.float32)
            else:
                arr = np.zeros((n_rows, n_cols), dtype=np.float32)
                read_idx = 0
                for i, present in enumerate(self._seq_present):
                    if present:
                        arr[:, i] = df.iloc[:, read_idx].to_numpy(dtype=np.float32)
                        read_idx += 1
            arr = (arr - self._seq_bias) / self._seq_scale + 5e-8
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        self._seq_cache[fp] = (arr, jd_idx)
        if len(self._seq_cache) > self._seq_cache_max:
            self._seq_cache.popitem(last=False)
        return arr, jd_idx

    def _get_exog_cached(self, fp):
        """Return (arr[T, exog_dim], jd_idx[T]) for exog columns, scaled."""
        if self.exog_dim == 0:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        if fp in self._exog_cache:
            arr, jd_idx = self._exog_cache.pop(fp)
            self._exog_cache[fp] = (arr, jd_idx)
            return arr, jd_idx
        df = pd.read_parquet(fp, columns=self._exog_read_cols)
        if df.empty:
            arr = np.zeros((0, self.exog_dim), dtype=np.float32)
            jd_idx = np.zeros((0,), dtype=np.int32)
        else:
            jd_idx = df.index.to_julian_date().astype(np.int32).to_numpy()
            n_rows = len(df)
            if all(self._exog_present):
                arr = df.to_numpy(dtype=np.float32)
            else:
                arr = np.zeros((n_rows, self.exog_dim), dtype=np.float32)
                read_idx = 0
                for i, present in enumerate(self._exog_present):
                    if present:
                        arr[:, i] = df.iloc[:, read_idx].to_numpy(dtype=np.float32)
                        read_idx += 1
            arr = (arr - self._exog_bias) / self._exog_scale + 5e-8
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        self._exog_cache[fp] = (arr, jd_idx)
        if len(self._exog_cache) > self._exog_cache_max:
            self._exog_cache.popitem(last=False)
        return arr, jd_idx

    def _epoch_day_to_jd(self, epoch_day):
        """Convert days-since-1970 to Julian Day integer (matching parquet index)."""
        return int(epoch_day) + _JD_EPOCH

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        station_idx, end_tidx = self.index[idx]
        start_tidx = end_tidx - (self.seq_len - 1)
        station_id = self._station_ids[station_idx]

        # ---- y: target obs from stations.zarr, scaled ----
        y_raw = self._sz[self._target_variable][start_tidx : end_tidx + 1, station_idx]
        y = (y_raw.astype(np.float32) - self._y_bias) / self._y_scale + 5e-8
        y = torch.from_numpy(y)

        # Day-of-interest (last day in window) as Julian Day for parquet alignment
        day_epoch = int(self._time_days[end_tidx])
        day_jd = self._epoch_day_to_jd(day_epoch)

        # ---- Neighbors from graph.zarr ----
        all_neighbors = self._neighbor_idx[station_idx]  # (K,)
        chosen = all_neighbors[: self.n_nodes]

        # ---- Embeddings ----
        if self._embeddings is not None:
            source_embeddings = []
            for ni in chosen:
                sid = self._station_ids[ni]
                if sid in self._embeddings:
                    source_embeddings.append(self._embeddings[sid])
                else:
                    source_embeddings.append(
                        torch.zeros(self.emb_dim, dtype=torch.float32)
                    )
            emb_stack = torch.stack(source_embeddings, dim=0)
        else:
            emb_stack = torch.zeros(len(chosen), 0, dtype=torch.float32)

        # ---- Edge attributes ----
        # Base edge attr from graph.zarr (target - neighbor feature deltas)
        edge_attr_base = torch.from_numpy(
            self._edge_attr[station_idx, : self.n_nodes].astype(np.float32)
        )
        # Bearing sin/cos
        b_sin = torch.from_numpy(
            self._bearing_sin[station_idx, : self.n_nodes].astype(np.float32)
        ).unsqueeze(1)
        b_cos = torch.from_numpy(
            self._bearing_cos[station_idx, : self.n_nodes].astype(np.float32)
        ).unsqueeze(1)
        # Distance
        dist = torch.from_numpy(
            self._distance_km[station_idx, : self.n_nodes].astype(np.float32)
        ).unsqueeze(1)
        edge_attr = torch.cat([edge_attr_base, b_sin, b_cos, dist], dim=1)

        # ---- Neighbor sequences + obs + exog ----
        seq_list = []
        mask_list = []
        obs_curr = []
        nbr_exog_today = []

        for ni in chosen:
            sid = self._station_ids[ni]
            fp = self._file_map.get(sid)
            ok = False

            if fp is not None and os.path.exists(fp):
                try:
                    arr, jd_idx = self._get_seq_cached(fp)
                    if arr.shape[0] >= self.seq_len:
                        where = np.where(jd_idx == day_jd)[0]
                        if where.size > 0:
                            end = int(where[0])
                            start = end - (self.seq_len - 1)
                            if start >= 0:
                                window_days = jd_idx[start : end + 1]
                                if np.all(window_days[1:] - window_days[:-1] == 1):
                                    y_seq = arr[start : end + 1, 0:1]
                                    y_shift = y_seq.copy()
                                    if y_shift.shape[0] > 1:
                                        y_shift[1:, 0] = y_seq[:-1, 0]
                                    y_shift[0, 0] = y_seq[0, 0]
                                    exog = arr[start : end + 1, 1:]
                                    feats = np.concatenate([y_shift, exog], axis=1)
                                    seq_list.append(torch.from_numpy(feats))
                                    mask_list.append(True)
                                    obs_curr.append(float(arr[end, 0]))
                                    # Neighbor exog for current day
                                    exog_arr, exog_jd = self._get_exog_cached(fp)
                                    if exog_arr.shape[0] > 0:
                                        w2 = np.where(exog_jd == day_jd)[0]
                                        if w2.size > 0:
                                            nbr_exog_today.append(
                                                torch.from_numpy(
                                                    exog_arr[int(w2[0])].astype(
                                                        np.float32
                                                    )
                                                )
                                            )
                                        else:
                                            nbr_exog_today.append(
                                                torch.zeros(
                                                    self.exog_dim, dtype=torch.float32
                                                )
                                            )
                                    else:
                                        nbr_exog_today.append(
                                            torch.zeros(
                                                self.exog_dim, dtype=torch.float32
                                            )
                                        )
                                    ok = True
                except Exception:
                    pass

            if not ok:
                seq_list.append(
                    torch.zeros(self.seq_len, self.seq_in_channels, dtype=torch.float32)
                )
                mask_list.append(False)
                obs_curr.append(0.0)
                nbr_exog_today.append(torch.zeros(self.exog_dim, dtype=torch.float32))

        neighbor_seq = torch.stack(seq_list, dim=0)  # (k, T, C)
        neighbor_mask = torch.tensor(mask_list, dtype=torch.bool)

        # ---- Target exog for day-of-interest ----
        target_exog = torch.zeros(self.exog_dim, dtype=torch.float32)
        fp_target = self._file_map.get(station_id)
        if fp_target is not None and os.path.exists(fp_target):
            exog_arr, exog_jd = self._get_exog_cached(fp_target)
            if exog_arr.shape[0] > 0:
                w = np.where(exog_jd == day_jd)[0]
                if w.size > 0:
                    target_exog = torch.from_numpy(
                        exog_arr[int(w[0])].astype(np.float32)
                    )

        # ---- Node features ----
        # Target: [zeros(emb_dim), exog_today, 0.0]
        tgt_row = torch.cat(
            [
                torch.zeros(self.emb_dim, dtype=torch.float32),
                target_exog,
                torch.zeros(1, dtype=torch.float32),
            ],
            dim=-1,
        )
        # Neighbors: [embedding, exog_today, obs_current]
        obs_tensor = torch.tensor(obs_curr, dtype=torch.float32).unsqueeze(1)
        nbr_exog_mat = torch.stack(nbr_exog_today, dim=0)
        nbr_rows = torch.cat([emb_stack, nbr_exog_mat, obs_tensor], dim=1)
        x = torch.cat([tgt_row.unsqueeze(0), nbr_rows], dim=0)

        # ---- Edge index: neighbors -> target ----
        source_indices = torch.arange(1, len(chosen) + 1)
        target_index = torch.zeros(len(chosen), dtype=torch.long)
        edge_index = torch.stack([source_indices, target_index], dim=0)

        # ---- Dynamic exog deltas in edge attributes ----
        if self.exog_dim > 0:
            delta = target_exog.unsqueeze(0) - nbr_exog_mat
            edge_attr = torch.cat([edge_attr, delta], dim=1)

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # ---- Target temporal sequence (exog-only, y channel zeroed) ----
        target_seq = torch.zeros(
            self.seq_len, self.seq_in_channels, dtype=torch.float32
        )
        if fp_target is not None and os.path.exists(fp_target):
            try:
                arr_t, jd_idx_t = self._get_seq_cached(fp_target)
                where_t = np.where(jd_idx_t == day_jd)[0]
                if where_t.size > 0:
                    end_t = int(where_t[0])
                    start_t = end_t - (self.seq_len - 1)
                    if start_t >= 0:
                        window_days_t = jd_idx_t[start_t : end_t + 1]
                        if np.all(window_days_t[1:] - window_days_t[:-1] == 1):
                            exog_t = arr_t[start_t : end_t + 1, 1:]
                            if exog_t.shape[0] == self.seq_len:
                                target_seq[:, 1:] = torch.from_numpy(exog_t)
            except Exception:
                pass

        return graph, y, neighbor_seq, neighbor_mask, target_seq
