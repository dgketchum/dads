import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from models.components.scalers import MinMaxScaler
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from prep.columns_desc import CDR_FEATURES, TERRAIN_FEATURES, CDR_MISS_FEATURES


class DadsDataset(Dataset):
    def __init__(
        self,
        n_nodes,
        lstm_input_files,
        seq_meta,
        embedding_dir,
        edge_map_file,
        edge_attr_file,
        scaler,
        sample=None,
        lstm_workers=1,
        record_holders: int = 3,
        normalize_keys=None,
        windows_per_station=None,
        neighbor_file_map=None,
    ):
        """Build per-sample graphs for DADS from prebuilt components.

        Ingests
        - Station sequences (Parquet in lstm_input_files): used to form labels `y` and to index
          contiguous windows; also supplies day-of-interest (`day_int`) used to align contexts.
        - Embeddings (embedding_dir/embeddings.json): station-level vectors produced by
          models.autoencoder.infer.WeatherAutoencoder over yearly windows, min-max normalized
          across stations here and used as neighbor node identity.
        - Edge map and attributes (edge_map_file, edge_attr_file): built by prep.graph.Graph
          (generate_edge_index). `edge_map_file` maps target->candidate neighbors; `edge_attr_file`
          holds per-station scaled attribute vectors; per-edge attributes are computed as
          (target_attr - neighbor_attr). If a sibling '*_edge_bearing.json' is present, per-edge
          sin/cos(bearing) are appended; geodesic distance (km, Haversine) is appended using lat/lon
          read from station Parquets.
        - Variable scaler (scaler): MinMaxScaler json from prep.build_variable_scaler; fit on graph
          train_ids only and used to scale day-of-interest exogenous features at the target node.

        Assembled sample (returned as (graph, y))
        - Nodes: 1 target (index 0) + n_nodes neighbors chosen from edge_map.
        - graph.x: shape (1 + n_nodes, emb_dim + exog_dim)
          * Target row (idx 0): [zeros(emb_dim), exog_vec(day-of-interest)] where exog_vec is the
            scaled slice of GEO_FEATURES present in the station Parquet on that day
            (exog = rsun, NOAA CDR daily bands, and available lat/lon/doy/terrain).
          * Neighbor rows (idx 1..k): [embedding, zeros(exog_dim)].
        - Neighbor sequences (returned alongside graph): shape (n_nodes, T, C)
          * For each neighbor, a 12-day window of [lagged target, exogenous] channels ending
            at day-of-interest if available; otherwise zeros with a mask indicating missing.
        - graph.edge_index: shape (2, n_nodes) with directed edges neighbor->target.
        - graph.edge_attr: shape (n_nodes, edge_dim) with (target_attr - neighbor_attr),
          optionally augmented with sin/cos(bearing) and distance (km).
        - y: shape (T,), observed target sequence aligned so that y[-1] corresponds to `day_int`.
          DadsMetGNN consumes y[-1] during training/validation.

        Builders
        - Embeddings: models/autoencoder/infer.py
        - Graph files: prep/graph.py (train_edge_index.json, train_edge_attr.json, and val analogs)
        - Variable scaler: prep/build_variable_scaler.py
        """

        self.n_nodes = n_nodes
        embedding_dict_file = os.path.join(embedding_dir, "embeddings.json")
        with open(embedding_dict_file, "r") as f:
            embeddings = json.load(f)
        embed_dict = embeddings
        # build raw tensor per station and ensure 1D shape
        raw = {}
        for k, v in embed_dict.items():
            t = torch.tensor(v)
            if t.dim() == 2 and t.shape[-1] == 1:
                t = t.squeeze(-1)
            elif t.dim() > 1:
                t = t.view(-1)
            raw[k] = t

        # verify uniform dimensionality
        lens = {int(t.shape[-1]) for t in raw.values()}
        assert len(lens) == 1, "embedding vectors must be uniform length"

        # choose keys for normalization (train-only if provided)
        if normalize_keys is not None:
            norm_keys = [k for k in raw.keys() if k in normalize_keys]
        else:
            norm_keys = list(raw.keys())
        if not norm_keys:
            norm_keys = list(raw.keys())
        norm_arr = torch.stack([raw[k] for k in norm_keys], dim=0)
        min_ = norm_arr.min(dim=0).values
        max_ = norm_arr.max(dim=0).values
        denom = max_ - min_
        denom[denom == 0] = 1.0

        # apply normalization to all stations
        self.embeddings = {k: (t - min_) / denom for k, t in raw.items()}

        with open(edge_attr_file, "r") as f:
            edge_attr = json.load(f)
        _ea_lens = {len(v) for v in edge_attr.values()}
        assert len(_ea_lens) == 1, "edge_attr vectors must be uniform length"
        self.edge_attr = {k: torch.tensor(v) for k, v in edge_attr.items()}

        with open(edge_map_file, "r") as f:
            edge_map = json.load(f)

        # Optional bearing file (same directory as edge_attr_file)
        bearing_file = edge_attr_file.replace("edge_attr", "edge_bearing")
        bearing_map = None
        if os.path.exists(bearing_file):
            with open(bearing_file, "r") as f:
                bearing_raw = json.load(f)
            # Build neighbor->bearing map per target from original edge_map order
            bearing_map = {}
            for k, nbrs in edge_map.items():
                blist = bearing_raw.get(k, [])
                if not blist:
                    continue
                m = {}
                for nb, b in zip(nbrs, blist):
                    m[nb] = float(b)
                bearing_map[k] = m

        # Optional distance file (km)
        distance_file = edge_attr_file.replace("edge_attr", "edge_distance")
        distance_map = None
        if os.path.exists(distance_file):
            with open(distance_file, "r") as f:
                distance_raw = json.load(f)
            distance_map = {}
            for k, nbrs in edge_map.items():
                dlist = distance_raw.get(k, [])
                if not dlist:
                    continue
                m = {}
                for nb, d in zip(nbrs, dlist):
                    m[nb] = float(d)
                distance_map[k] = m

        node_keys = set(self.embeddings.keys())
        attr_keys = set(self.edge_attr.keys())
        valid_nodes = node_keys & attr_keys

        # Targets: only those with files in this dataset; Neighbors: may come from a larger pool
        file_keys = {os.path.splitext(os.path.basename(f))[0] for f in lstm_input_files}
        if neighbor_file_map is not None:
            neighbor_keys = set(neighbor_file_map.keys())
        else:
            neighbor_keys = file_keys

        attributes = {}
        for k in tqdm(edge_map.keys(), desc="Building Graph Edge Attributes"):
            if k not in valid_nodes or k not in file_keys:
                continue
            try:
                possible_edges = edge_map[k]
            except KeyError:
                continue
            # Keep only valid neighbors that also have files (in neighbor_keys) and are not self
            filtered = [
                e
                for e in possible_edges
                if (e != k and e in valid_nodes and e in neighbor_keys)
            ]
            if len(filtered) >= n_nodes:
                attributes[k] = filtered  # keep all candidates; select n at sample-time

        if sample:
            keys = list(attributes.keys())
            keys = np.random.choice(keys, sample)
            attributes = {k: v for k, v in attributes.items() if k in keys}

        attr_set = set(attributes.keys())
        filter_files = [
            f
            for f in lstm_input_files
            if os.path.splitext(os.path.basename(f))[0] in attr_set
        ]

        self.edge_map = attributes
        self._bearing_map = bearing_map
        self._distance_map = distance_map

        self.record_holders = int(record_holders)
        # Determine feature names and tensor width from first file
        first_file = filter_files[0]
        feature_names = pd.read_parquet(first_file).columns.tolist()
        num_features = len(feature_names)
        assert feature_names[0].endswith("_obs") and not feature_names[1].endswith(
            "_obs"
        ), "unexpected column order"
        # column indices: y=0, comparator=1, features start=2; no hf; set hf idx to tensor_width for downstream math
        self.tensor_width = num_features
        self.column_indices = (0, 1, 2, self.tensor_width)
        chunk_size = int(seq_meta.get("chunk_size", 12))
        self.seq_len = chunk_size

        # load scaler from json path
        with open(scaler, "r") as f:
            scaler_params = json.load(f)
        scaler_obj = MinMaxScaler()
        scaler_obj.bias = np.array(scaler_params["bias"]).reshape(1, -1)
        scaler_obj.scale = np.array(scaler_params["scale"]).reshape(1, -1)
        if "feature_names" in scaler_params:
            assert scaler_params["feature_names"] == feature_names, (
                "scaler feature_names mismatch"
            )

        station_names = [os.path.splitext(os.path.basename(f))[0] for f in filter_files]
        # lightweight in-class window indexer (no LSTM dependency)
        self.file_paths = list(filter_files)
        self.file_station_names = list(station_names)
        self.feature_names = feature_names
        self.sample_dimensions = (chunk_size, num_features)
        self.scaler = scaler_obj
        self.max_windows_per_file = (
            None if windows_per_station is None else int(windows_per_station)
        )
        self._build_index(lstm_workers)
        # sequences for neighbor TCN: select [target, rsun, CDR, CDR_miss, doy_sin, doy_cos], scaled
        seq_names = (
            ["rsun"]
            + list(CDR_FEATURES)
            + list(CDR_MISS_FEATURES)
            + ["doy_sin", "doy_cos"]
        )
        self.seq_exog_idx = [i for i, n in enumerate(feature_names) if n in seq_names]
        self.seq_selected_indices = [0] + self.seq_exog_idx
        self.seq_selected_columns = [
            feature_names[i] for i in self.seq_selected_indices
        ]
        # bias/scale for selected indices only
        bias = np.asarray(scaler_obj.bias).reshape(-1)
        scale = np.asarray(scaler_obj.scale).reshape(-1)
        self.seq_bias = bias[self.seq_selected_indices].astype(np.float32)
        self.seq_scale = scale[self.seq_selected_indices].astype(np.float32)
        self.seq_in_channels = 1 + len(self.seq_exog_idx)
        # map station -> parquet path for fetching day-of-interest exogenous features (targets only)
        self._file_map = {
            os.path.splitext(os.path.basename(f))[0]: f for f in filter_files
        }
        # neighbor file map (may include stations beyond targets)
        if neighbor_file_map is not None:
            self._nbr_file_map = dict(neighbor_file_map)
        else:
            self._nbr_file_map = dict(self._file_map)
        # exogenous feature columns available in parquet and desired for node features (target and neighbors)
        # Use only exogenous inputs: RSUN + NOAA CDR (daily) + CDR missingness flags + TERRAIN + seasonal DOY features.
        # (Landsat bands are not used in node exog.)
        allowed_exog = (
            ["rsun"]
            + list(CDR_FEATURES)
            + list(CDR_MISS_FEATURES)
            + list(TERRAIN_FEATURES)
            + ["doy_sin", "doy_cos"]
        )
        self.exog_cols = [c for c in allowed_exog if c in feature_names]
        self.exog_idx = [feature_names.index(c) for c in self.exog_cols]
        if self.exog_cols:
            _bias = np.asarray(scaler_obj.bias).reshape(-1)
            _scale = np.asarray(scaler_obj.scale).reshape(-1)
            self.exog_bias = _bias[self.exog_idx].astype(np.float32)
            self.exog_scale = _scale[self.exog_idx].astype(np.float32)
        else:
            self.exog_bias = np.empty((0,), dtype=np.float32)
            self.exog_scale = np.ones((0,), dtype=np.float32)
        # dimensions for assembling node features
        self.emb_dim = int(next(iter(self.embeddings.values())).shape[-1])
        self.exog_dim = int(len(self.exog_cols))
        base_edge_dim = int(next(iter(self.edge_attr.values())).shape[-1])
        # Include dynamic (target_exog_today - neighbor_exog_today) deltas in edge attributes at sample-time
        self.edge_dim = (
            base_edge_dim
            + (2 if self._bearing_map is not None else 0)
            + (1 if self._distance_map is not None else 0)
            + int(len(self.exog_cols))
        )
        # small per-worker LRU caches to reduce parquet IO
        self._seq_cache = OrderedDict()  # file -> (arr_selected[T,C], di_idx[T])
        self._seq_cache_max = max(256, min(4096, len(self._file_map)))
        self._exog_cache = OrderedDict()  # file -> (arr_exog[T,exog_dim], di_idx[T])
        self._exog_cache_max = max(256, min(4096, len(self._file_map)))
        # no pre-cached contexts; contexts computed on-the-fly by the model's TCN

    def __len__(self):
        return len(self.index)

    def _scan_windows_worker(self, task):
        file_idx, file_path, chunk_size_, target_col = task
        try:
            df = pd.read_parquet(file_path, columns=[target_col])
            # drop only rows with NaN target
            df.dropna(subset=[target_col], inplace=True)
            if len(df) < chunk_size_:
                return file_idx, [], []
            di = df.index.to_julian_date().astype(np.int32).to_numpy()
            is_consec = (di[1:] - di[:-1]) == 1
            win = chunk_size_ - 1
            if len(is_consec) < win:
                return file_idx, [], []
            conv = np.convolve(is_consec, np.ones(win, dtype=int), mode="valid")
            starts = np.where(conv == win)[0]
            end_days = [int(di[s + win]) for s in starts]
            return file_idx, starts.tolist(), end_days
        except Exception:
            return file_idx, [], []

    def _build_index(self, n_workers):
        chunk_size_ = self.sample_dimensions[0]
        target_col = self.feature_names[0]
        tasks = [
            (i, self.file_paths[i], chunk_size_, target_col)
            for i in range(len(self.file_paths))
        ]
        index_ = []
        if n_workers == 1:
            results = []
            for t in tqdm(tasks, total=len(tasks), desc="Indexing windows"):
                results.append(self._scan_windows_worker(t))
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                results = list(
                    tqdm(
                        ex.map(self._scan_windows_worker, tasks),
                        total=len(tasks),
                        desc="Indexing windows",
                    )
                )
        rng = np.random.default_rng(42)
        for file_index, starts, end_days in results:
            if not starts:
                continue
            if (
                self.max_windows_per_file is not None
                and len(starts) > self.max_windows_per_file
            ):
                sel = rng.choice(
                    len(starts), size=self.max_windows_per_file, replace=False
                )
                starts = [starts[int(i)] for i in sel]
                end_days = [end_days[int(i)] for i in sel]
            for s, d in zip(starts, end_days):
                index_.append((file_index, s, d))
        self.index = index_

    def _get_seq_cached(self, fp):
        # Returns (arr[T,C], di_idx[T]) for seq_selected_columns
        if fp in self._seq_cache:
            arr, di_idx = self._seq_cache.pop(fp)
            self._seq_cache[fp] = (arr, di_idx)
            return arr, di_idx
        df = pd.read_parquet(fp, columns=self.seq_selected_columns)
        df.dropna(subset=[self.seq_selected_columns[0]], inplace=True)
        if df.empty:
            arr = np.zeros((0, len(self.seq_selected_columns)), dtype=np.float32)
            di_idx = np.zeros((0,), dtype=np.int32)
        else:
            arr = df.to_numpy(dtype=np.float32)
            arr = (arr - self.seq_bias) / self.seq_scale + 5e-8
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            di_idx = df.index.to_julian_date().astype(np.int32).to_numpy()
        self._seq_cache[fp] = (arr, di_idx)
        if len(self._seq_cache) > self._seq_cache_max:
            self._seq_cache.popitem(last=False)
        return arr, di_idx

    def _get_exog_cached(self, fp):
        # Returns (arr[T, exog_dim], di_idx[T]) for target exog cols
        if self.exog_dim == 0:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        if fp in self._exog_cache:
            arr, di_idx = self._exog_cache.pop(fp)
            self._exog_cache[fp] = (arr, di_idx)
            return arr, di_idx
        df = pd.read_parquet(fp, columns=self.exog_cols)
        if df.empty:
            arr = np.zeros((0, self.exog_dim), dtype=np.float32)
            di_idx = np.zeros((0,), dtype=np.int32)
        else:
            di_idx = df.index.to_julian_date().astype(np.int32).to_numpy()
            arr = df.to_numpy(dtype=np.float32)
            arr = (arr - self.exog_bias) / self.exog_scale + 5e-8
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        self._exog_cache[fp] = (arr, di_idx)
        if len(self._exog_cache) > self._exog_cache_max:
            self._exog_cache.popitem(last=False)
        return arr, di_idx

    def _get_target_exog(self, station, day_int):
        # fetch scaled exogenous features for the day-of-interest
        if self.exog_dim == 0:
            return torch.empty(0, dtype=torch.float32)
        fp = self._file_map[station]
        arr, di_idx = self._get_exog_cached(fp)
        di = int(day_int)
        where = np.where(di_idx == di)[0]
        assert where.size > 0, "no exogenous row for this day"
        row = arr[where[0]]
        row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(row.astype(np.float32))

    def __getitem__(self, idx):
        # Sample from local index: (file_idx, start_iloc, end_day)
        file_idx, start_iloc, day_int = self.index[idx]
        target_station = self.file_station_names[file_idx]
        # Load scaled target sequence y for the window
        df_y = pd.read_parquet(
            self.file_paths[file_idx], columns=[self.feature_names[0]]
        )
        df_y.dropna(subset=[self.feature_names[0]], inplace=True)
        chunk = df_y.iloc[start_iloc : start_iloc + self.sample_dimensions[0]].to_numpy(
            dtype=np.float32
        )
        # scale using variable-specific scaler (column 0)
        y = (chunk - self.scaler.bias[0, 0]) / self.scaler.scale[0, 0] + 5e-8
        y = torch.from_numpy(y.squeeze(-1))

        candidates = self.edge_map[target_station]
        # Split candidates into natural neighbors and trailing record_holders
        tail_n = min(self.record_holders, len(candidates))
        naturals = candidates[:-tail_n] if tail_n > 0 else candidates
        record_tail = candidates[-tail_n:] if tail_n > 0 else []

        # choose up to n_nodes neighbors deterministically from naturals (graph order), then fill from record holders
        chosen = []
        if naturals:
            k = min(self.n_nodes, len(naturals))
            chosen.extend(naturals[:k])
        if len(chosen) < self.n_nodes and record_tail:
            for stn in record_tail:
                if len(chosen) >= self.n_nodes:
                    break
                if stn not in chosen:
                    chosen.append(stn)

        source_stations = chosen if chosen else candidates[: self.n_nodes]
        source_embeddings = [self.embeddings[stn] for stn in source_stations]
        emb_stack = torch.stack(source_embeddings, dim=0)

        # We'll build neighbor sequences and also capture each neighbor's current-day observation
        # before assembling node features so we can append the observation to neighbor rows.

        # Per-edge attributes are (target_attr - neighbor_attr) from precomputed station features
        to_point = self.edge_attr[target_station]
        from_point = [to_point - self.edge_attr[stn] for stn in source_stations]
        edge_attr = torch.stack(from_point, dim=0)
        if self._bearing_map is not None:
            if target_station in self._bearing_map:
                bm = self._bearing_map[target_station]
                ang = torch.tensor(
                    [bm.get(s, 0.0) for s in source_stations], dtype=edge_attr.dtype
                )
                rad = torch.deg2rad(ang)
                sc = torch.stack(
                    [torch.sin(rad), torch.cos(rad)], dim=1
                )  # append sin/cos(bearing)
            else:
                sc = torch.zeros(
                    (edge_attr.shape[0], 2), dtype=edge_attr.dtype
                )  # no bearings available
            edge_attr = torch.cat([edge_attr, sc], dim=1)
        if self._distance_map is not None:
            if target_station in self._distance_map:
                dm = self._distance_map[target_station]
                dist = torch.tensor(
                    [dm.get(s, 0.0) for s in source_stations], dtype=edge_attr.dtype
                ).unsqueeze(1)
            else:
                dist = torch.zeros(
                    (edge_attr.shape[0], 1), dtype=edge_attr.dtype
                )  # missing distances default to 0
            edge_attr = torch.cat([edge_attr, dist], dim=1)
        assert edge_attr.shape[0] == len(source_stations), "edge count mismatch"

        # Build neighbor 12-day sequences for TCN; shape (k, T, C)
        seq_list = []
        mask_list = []
        di = int(day_int)
        obs_curr = []
        nbr_exog_today = []
        for stn in source_stations:
            fp = self._nbr_file_map.get(stn)
            ok = False
            if fp is not None and os.path.exists(fp):
                try:
                    arr, di_idx = self._get_seq_cached(fp)
                    if arr.shape[0] >= self.seq_len:
                        where = np.where(di_idx == di)[0]
                        if where.size > 0:
                            end = int(where[0])
                            start = end - (self.seq_len - 1)
                            if start >= 0:
                                window_days = di_idx[start : end + 1]
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
                                    # capture current-day observation (scaled) for neighbor
                                    obs_curr.append(float(arr[end, 0]))
                                    # capture neighbor exog vector for current day (scaled)
                                    # Retrieve from cached exog array to ensure identical ordering as self.exog_cols
                                    exog_arr, exog_di = self._get_exog_cached(fp)
                                    if exog_arr.shape[0] > 0:
                                        w2 = np.where(exog_di == di)[0]
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

        # build target-node features from day-of-interest exogenous data; neighbors use their own exog for this day
        di = int(day_int)
        exog_vec = self._get_target_exog(target_station, di)
        # target node: no embedding (inference at places without station embeddings)
        tgt_row = torch.cat(
            [
                torch.zeros(self.emb_dim, dtype=emb_stack.dtype),
                exog_vec,
                torch.zeros(1, dtype=emb_stack.dtype),
            ],
            dim=-1,
        )
        # neighbor rows: [embedding, neighbor_exog_today, obs_current]
        obs_tensor = torch.tensor(obs_curr, dtype=emb_stack.dtype).unsqueeze(1)
        nbr_exog_mat = torch.stack(nbr_exog_today, dim=0).to(dtype=emb_stack.dtype)
        nbr_rows = torch.cat([emb_stack, nbr_exog_mat, obs_tensor], dim=1)
        x = torch.cat([tgt_row.unsqueeze(0), nbr_rows], dim=0)
        assert x.shape[0] == len(source_stations) + 1, "node count mismatch"

        # Build directed edges: neighbors (rows 1-k) → target (row 0)
        source_indices = torch.arange(1, len(source_stations) + 1)
        target_index = torch.zeros(len(source_stations), dtype=torch.long)
        edge_index = torch.stack([source_indices, target_index], dim=0)

        # Append dynamic (target_exog_today - neighbor_exog_today) to edge attributes
        if self.exog_dim > 0:
            delta = exog_vec.unsqueeze(0) - nbr_exog_mat
            edge_attr = torch.cat([edge_attr, delta.to(dtype=edge_attr.dtype)], dim=1)

        # Initialize graph; node_ctx will be filled by the model using TCN contexts
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Build target exog-only temporal context (pad y channel with zeros to match neighbor TCN channels)
        target_seq = torch.zeros(
            self.seq_len, self.seq_in_channels, dtype=torch.float32
        )
        try:
            arr_t, di_idx_t = self._get_seq_cached(self.file_paths[file_idx])
            where_t = np.where(di_idx_t == di)[0]
            if where_t.size > 0:
                end_t = int(where_t[0])
                start_t = end_t - (self.seq_len - 1)
                if start_t >= 0:
                    window_days_t = di_idx_t[start_t : end_t + 1]
                    if np.all(window_days_t[1:] - window_days_t[:-1] == 1):
                        exog_t = arr_t[start_t : end_t + 1, 1:]  # exog-only channels
                        if exog_t.shape[0] == self.seq_len:
                            target_seq[:, 1:] = torch.from_numpy(exog_t)
        except Exception:
            pass

        return graph, y, neighbor_seq, neighbor_mask, target_seq


if __name__ == "__main__":
    pass
# ========================= EOF ====================================================================
