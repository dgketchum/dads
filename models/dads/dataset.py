import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from models.lstm.dataset import LSTMDataset
from models.scalers import MinMaxScaler
from tqdm import tqdm
from prep.columns_desc import GEO_FEATURES


class DadsDataset(Dataset):
    def __init__(self, n_nodes, lstm_input_files, lstm_meta, embedding_dir, edge_map_file, edge_attr_file,
                 scaler, sample=None, node_ctx_dir=None, lstm_workers=1, record_holders: int = 3):
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
        - Node contexts (node_ctx_dir): per-station/day .npy vectors captured from the LSTM
          pretrainer via models.dads.cache_node_contexts (hook on LSTMPredictor.fc1); used for
          neighbor rows at the day-of-interest; missing neighbors are mean-imputed.
        - Variable scaler (scaler): MinMaxScaler json from prep.build_variable_scaler; fit on graph
          train_ids only and used to scale day-of-interest exogenous features at the target node.

        Assembled sample (returned as (graph, y))
        - Nodes: 1 target (index 0) + n_nodes neighbors chosen from edge_map.
        - graph.x: shape (1 + n_nodes, emb_dim + exog_dim)
          * Target row (idx 0): [zeros(emb_dim), exog_vec(day-of-interest)] where exog_vec is the
            scaled slice of GEO_FEATURES present in the station Parquet on that day
            (exog = rsun, NOAA CDR daily bands, and available lat/lon/doy/terrain).
          * Neighbor rows (idx 1..k): [embedding, zeros(exog_dim)].
        - graph.node_lstm: shape (1 + n_nodes, ctx_dim)
          * Target row is zeros; neighbor rows hold LSTM contexts for `day_int` if available,
            otherwise filled with a distance-weighted mean of available neighbor contexts.
        - graph.edge_index: shape (2, n_nodes) with directed edges neighbor->target.
        - graph.edge_attr: shape (n_nodes, edge_dim) with (target_attr - neighbor_attr),
          optionally augmented with sin/cos(bearing) and distance (km).
        - y: shape (T,), observed target sequence aligned so that y[-1] corresponds to `day_int`.
          DadsMetGNN consumes y[-1] during training/validation.

        Builders
        - Embeddings: models/autoencoder/infer.py
        - Graph files: prep/graph.py (train_edge_index.json, train_edge_attr.json, and val analogs)
        - Node contexts: models/dads/cache_node_contexts.py
        - Variable scaler: prep/build_variable_scaler.py
        """

        self.n_nodes = n_nodes
        embedding_dict_file = os.path.join(embedding_dir, 'embeddings.json')
        with open(embedding_dict_file, 'r') as f:
            embeddings = json.load(f)
        embed_dict = embeddings
        _emb_lens = {len(v) for v in embed_dict.values()}
        assert len(_emb_lens) == 1, "embedding vectors must be uniform length"
        embeddings = [(k, torch.tensor(v)) for k, v in embeddings.items()]
        # Embeddings may be saved as [latent, 1]; squeeze trailing dim to get 1D per station
        embedding_arr = torch.stack([v[1] for v in embeddings], dim=0)
        if embedding_arr.dim() == 3 and embedding_arr.shape[-1] == 1:
            embedding_arr = embedding_arr.squeeze(-1)
        min_ = embedding_arr.min(dim=0).values
        max_ = embedding_arr.max(dim=0).values
        denom = (max_ - min_)
        denom[denom == 0] = 1.0
        embedding_arr = (embedding_arr - min_) / denom

        self.embeddings = {k[0]: embedding_arr[i] for i, k in enumerate(embeddings)}

        with open(edge_attr_file, 'r') as f:
            edge_attr = json.load(f)
        _ea_lens = {len(v) for v in edge_attr.values()}
        assert len(_ea_lens) == 1, "edge_attr vectors must be uniform length"
        self.edge_attr = {k: torch.tensor(v) for k, v in edge_attr.items()}

        with open(edge_map_file, 'r') as f:
            edge_map = json.load(f)

        # Optional bearing file (same directory as edge_attr_file)
        bearing_file = edge_attr_file.replace('edge_attr', 'edge_bearing')
        bearing_map = None
        if os.path.exists(bearing_file):
            with open(bearing_file, 'r') as f:
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
        distance_file = edge_attr_file.replace('edge_attr', 'edge_distance')
        distance_map = None
        if os.path.exists(distance_file):
            with open(distance_file, 'r') as f:
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

        attributes = {}
        for k in tqdm(edge_map.keys(), desc='Building Graph Edge Attributes'):
            if k not in valid_nodes:
                continue
            try:
                possible_edges = edge_map[k]
            except KeyError:
                continue
            # Preserve order from possible_edges; keep only valid neighbors and exclude self
            filtered = [e for e in possible_edges if e != k and e in valid_nodes]
            if len(filtered) >= n_nodes:
                attributes[k] = filtered  # keep all candidates; select n at sample-time

        file_keys = {os.path.splitext(os.path.basename(f))[0] for f in lstm_input_files}
        attributes = {k: v for k, v in attributes.items() if k in file_keys}

        if sample:
            keys = list(attributes.keys())
            keys = np.random.choice(keys, sample)
            attributes = {k: v for k, v in attributes.items() if k in keys}

        attr_set = set(attributes.keys())
        filter_files = [f for f in lstm_input_files if os.path.splitext(os.path.basename(f))[0] in attr_set]

        self.edge_map = attributes
        self._bearing_map = bearing_map
        self._distance_map = distance_map

        self.model = 'lstm'
        self.record_holders = int(record_holders)
        # Determine feature names and tensor width from first file
        first_file = filter_files[0]
        feature_names = pd.read_parquet(first_file).columns.tolist()
        num_features = len(feature_names)
        assert feature_names[0].endswith('_obs') and not feature_names[1].endswith('_obs'), "unexpected column order"
        # column indices: y=0, comparator=1, features start=2; no hf; set hf idx to tensor_width for downstream math
        self.tensor_width = num_features
        self.column_indices = (0, 1, 2, self.tensor_width)
        chunk_size = lstm_meta.get('chunk_size', 12)

        # load scaler from json path
        with open(scaler, 'r') as f:
            scaler_params = json.load(f)
        scaler_obj = MinMaxScaler()
        scaler_obj.bias = np.array(scaler_params['bias']).reshape(1, -1)
        scaler_obj.scale = np.array(scaler_params['scale']).reshape(1, -1)
        if 'feature_names' in scaler_params:
            assert scaler_params['feature_names'] == feature_names, "scaler feature_names mismatch"

        station_names = [os.path.splitext(os.path.basename(f))[0] for f in filter_files]
        self.lstm_dataset = LSTMDataset(file_paths=filter_files,
                                        station_names=station_names,
                                        feature_names=feature_names,
                                        sample_dimensions=(chunk_size, num_features),
                                        scaler=scaler_obj,
                                        return_station_name=True,
                                        n_workers=lstm_workers)
        assert node_ctx_dir is not None and os.path.isdir(node_ctx_dir), "node_ctx_dir required and must exist"
        self.node_ctx_dir = node_ctx_dir
        # map station -> parquet path for fetching day-of-interest exogenous features
        self._file_map = {os.path.splitext(os.path.basename(f))[0]: f for f in filter_files}
        # exogenous feature columns available in parquet and desired for target-node features
        self.exog_cols = [c for c in GEO_FEATURES if c in feature_names]
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
        self.edge_dim = base_edge_dim + (2 if self._bearing_map is not None else 0) + (1 if self._distance_map is not None else 0)
        # infer node context dimensionality once for stable batching
        self.ctx_dim = None
        for stn in self.edge_map.keys():
            stn_dir = os.path.join(self.node_ctx_dir, stn)
            if not os.path.isdir(stn_dir):
                continue
            for name in os.listdir(stn_dir):
                if name.endswith('.npy'):
                    arr = np.load(os.path.join(stn_dir, name))
                    arr = np.asarray(arr).squeeze()
                    self.ctx_dim = int(arr.shape[-1])
                    break
            if self.ctx_dim is not None:
                break
        assert self.ctx_dim is not None, "no node contexts found for any station in this dataset"

    def __len__(self):
        return len(self.lstm_dataset)

    def _get_target_exog(self, station, day_int):
        # fetch scaled exogenous features for the day-of-interest
        if self.exog_dim == 0:
            return torch.empty(0, dtype=torch.float32)
        fp = self._file_map[station]
        df = pd.read_parquet(fp, columns=self.exog_cols)
        di = df.index.to_julian_date().astype(np.int32).to_numpy()
        m = (di == int(day_int))
        idx = np.where(m)[0]
        assert idx.size > 0, "no exogenous row for this day"
        row = df.iloc[idx[0]].to_numpy(dtype=np.float32)
        row = (row - self.exog_bias) / self.exog_scale + 5e-8
        return torch.from_numpy(row)

    def __getitem__(self, idx):

        # in the reanalysis-independent model, seq should be unused
        y, comparator, _, target_station, day_int = self.lstm_dataset[idx]

        candidates = self.edge_map[target_station]
        # Split candidates into natural neighbors and trailing record_holders
        tail_n = min(self.record_holders, len(candidates))
        naturals = candidates[:-tail_n] if tail_n > 0 else candidates
        record_tail = candidates[-tail_n:] if tail_n > 0 else []

        # choose up to n_nodes neighbors preferring those with available context for this day
        chosen, ctx_list, missing_flags = [], [], []
        # 1) First pass: naturals that HAVE contexts for this day (sample uniformly, deterministic per (station, day))
        avail_pairs = []
        di = int(day_int)
        for stn in naturals:
            p = os.path.join(self.node_ctx_dir, stn, f"{di}.npy")
            if os.path.exists(p):
                v = np.load(p)
                v = np.asarray(v).squeeze()
                t = torch.from_numpy(v)
                if t.dim() != 1:
                    t = t.view(-1)  # likely error if not 1D
                assert t.shape[-1] == self.ctx_dim, "node context width mismatch"
                avail_pairs.append((stn, t))
        if avail_pairs:
            seed_ = (sum(ord(ch) for ch in str(target_station)) + di) & 0xFFFFFFFF  # stable seed
            rng = np.random.default_rng(seed_)
            k = min(self.n_nodes, len(avail_pairs))
            sel = rng.choice(len(avail_pairs), size=k, replace=False)
            for i in sel:
                stn, t = avail_pairs[int(i)]
                chosen.append(stn)
                ctx_list.append(t)
                missing_flags.append(False)
        # 2) Second pass: remaining naturals WITHOUT contexts
        if len(chosen) < self.n_nodes:
            selected = set(chosen)
            for stn in naturals:
                if len(chosen) >= self.n_nodes:
                    break
                if stn in selected:
                    continue
                chosen.append(stn)
                ctx_list.append(None)
                missing_flags.append(True)
        # 3) Final pass: record holders (appended by prep) as last resort
        if len(chosen) < self.n_nodes and record_tail:
            for stn in record_tail:
                if len(chosen) >= self.n_nodes:
                    break
                if stn in chosen:
                    continue
                chosen.append(stn)
                ctx_list.append(None)
                missing_flags.append(True)

        source_stations = chosen if chosen else candidates[:self.n_nodes]
        source_embeddings = [self.embeddings[stn] for stn in source_stations]
        emb_stack = torch.stack(source_embeddings, dim=0)
        # build target-node features from day-of-interest exogenous data; neighbors keep embeddings
        di = int(day_int)
        exog_vec = self._get_target_exog(target_station, di)
        zeros_exog = torch.zeros(self.exog_dim, dtype=emb_stack.dtype)
        tgt_row = torch.cat([torch.zeros(self.emb_dim, dtype=emb_stack.dtype), exog_vec], dim=-1)
        nbr_rows = torch.cat([emb_stack, zeros_exog.repeat(len(source_stations), 1)], dim=1)
        x = torch.cat([tgt_row.unsqueeze(0), nbr_rows], dim=0)
        assert x.shape[0] == len(source_stations) + 1, "node count mismatch"

        source_indices = torch.arange(1, len(source_stations) + 1)
        target_index = torch.zeros(len(source_stations), dtype=torch.long)
        edge_index = torch.stack([source_indices, target_index], dim=0)

        to_point = self.edge_attr[target_station]
        from_point = [to_point - self.edge_attr[stn] for stn in source_stations]
        edge_attr = torch.stack(from_point, dim=0)
        if self._bearing_map is not None:
            if target_station in self._bearing_map:
                bm = self._bearing_map[target_station]
                ang = torch.tensor([bm.get(s, 0.0) for s in source_stations], dtype=edge_attr.dtype)
                rad = torch.deg2rad(ang)
                sc = torch.stack([torch.sin(rad), torch.cos(rad)], dim=1)
            else:
                sc = torch.zeros((edge_attr.shape[0], 2), dtype=edge_attr.dtype)
            edge_attr = torch.cat([edge_attr, sc], dim=1)
        if self._distance_map is not None:
            if target_station in self._distance_map:
                dm = self._distance_map[target_station]
                dist = torch.tensor([dm.get(s, 0.0) for s in source_stations], dtype=edge_attr.dtype).unsqueeze(1)
            else:
                dist = torch.zeros((edge_attr.shape[0], 1), dtype=edge_attr.dtype)
            edge_attr = torch.cat([edge_attr, dist], dim=1)
        assert edge_attr.shape[0] == len(source_stations), "edge count mismatch"

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph.node_lstm = torch.zeros(x.shape[0], self.ctx_dim)

        if any(missing_flags):
            tmpl = next((t for t in ctx_list if t is not None), None)
            assert tmpl is not None, "no available node contexts for neighbors on this day"
            # distance-weighted mean if distances provided, else simple mean
            avail_pairs = [(stn, t) for stn, t, m in zip(source_stations, ctx_list, missing_flags) if t is not None]
            if self._distance_map is not None and target_station in self._distance_map:
                dm = self._distance_map[target_station]
                ws = []
                for stn, _ in avail_pairs:
                    d_km = float(dm.get(stn, 0.0))
                    ws.append(1.0 / (d_km + 1e-6))
                w = torch.tensor(ws, dtype=torch.float32)
                w = w / w.sum()
                avail = torch.stack([t for _, t in avail_pairs], dim=0).float()
                mean_vec = (avail * w.view(-1, 1)).sum(dim=0)
            else:
                avail = torch.stack([t for _, t in avail_pairs], dim=0).float()
                mean_vec = avail.mean(dim=0)
            ctx_filled = [mean_vec if m else t for t, m in zip(ctx_list, missing_flags)]
        else:
            ctx_filled = ctx_list
        ctx_t = torch.stack(ctx_filled, dim=0).float()
        zero_row = torch.zeros_like(ctx_t[0])
        node_ctx = torch.cat([zero_row.unsqueeze(0), ctx_t], dim=0)
        assert node_ctx.shape[0] == x.shape[0], "node_lstm rows must match nodes"
        graph.node_lstm = node_ctx

        return graph, y


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
