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


class DadsDataset(Dataset):
    def __init__(self, n_nodes, lstm_input_files, lstm_meta, embedding_dir, edge_map_file, edge_attr_file,
                 scaler, sample=None, node_ctx_dir=None, lstm_workers=1, record_holders: int = 3):
        """"""

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
        target_row = torch.zeros_like(emb_stack[0])
        x = torch.cat([target_row.unsqueeze(0), emb_stack], dim=0)
        assert x.shape[0] == len(source_stations) + 1, "node count mismatch"

        source_indices = torch.arange(1, len(source_stations) + 1)
        target_index = torch.zeros(len(source_stations), dtype=torch.long)
        edge_index = torch.stack([source_indices, target_index], dim=0)

        to_point = self.edge_attr[target_station]
        from_point = [to_point - self.edge_attr[stn] for stn in source_stations]
        edge_attr = torch.stack(from_point, dim=0)
        assert edge_attr.shape[0] == len(source_stations), "edge count mismatch"

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph.node_lstm = torch.zeros(x.shape[0], self.ctx_dim)

        if any(missing_flags):
            tmpl = next((t for t in ctx_list if t is not None), None)
            assert tmpl is not None, "no available node contexts for neighbors on this day"
            avail = torch.stack([t for t in ctx_list if t is not None], dim=0).float()
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
