import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from models.lstm.dataset import LSTMDataset
from models.scalers import MinMaxScaler


class DadsDataset(Dataset):
    def __init__(self, n_nodes, lstm_input_files, lstm_meta, embedding_dir, edge_map_file, edge_attr_file,
                 scaler, sample=None, node_ctx_dir=None):
        """"""

        embedding_dict_file = os.path.join(embedding_dir, 'embeddings.json')
        with open(embedding_dict_file, 'r') as f:
            embeddings = json.load(f)
        embed_dict = embeddings
        _emb_lens = {len(v) for v in embed_dict.values()}
        assert len(_emb_lens) == 1, "embedding vectors must be uniform length"
        embeddings = [(k, torch.tensor(v)) for k, v in embeddings.items()]
        # Stack to [N_stations, emb_dim]
        embedding_arr = torch.stack([v[1] for v in embeddings], dim=0)
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

        node_keys = list(self.embeddings.keys())
        attr_keys = list(self.edge_attr.keys())
        edge_keys = list(edge_map.keys())

        attributes = {}
        for k in edge_keys:
            final_edges = []

            try:
                possible_edges = edge_map[k]
            except KeyError:
                continue

            for e in possible_edges:

                if k == e:
                    continue

                if e in node_keys and e in attr_keys:
                    final_edges.append(e)

                if len(final_edges) == n_nodes:
                    if k in node_keys and k in attr_keys:
                        attributes[k] = final_edges
                    break

        file_keys = [os.path.splitext(os.path.basename(f))[0] for f in lstm_input_files]
        attributes = {k: v for k, v in attributes.items() if k in file_keys}

        if sample:
            keys = list(attributes.keys())
            keys = np.random.choice(keys, sample)
            attributes = {k: v for k, v in attributes.items() if k in keys}

        filter_files = [f for f in lstm_input_files if os.path.splitext(os.path.basename(f))[0] in attributes.keys()]

        self.edge_map = attributes

        data_frequency = lstm_meta.get('data_frequency', 'daily')
        self.model = 'lstm'
        # Determine feature names and tensor width from first file
        first_file = filter_files[0]
        feature_names = pd.read_parquet(first_file).columns.tolist()
        num_features = len(feature_names)
        assert feature_names[0].endswith('_obs') and not feature_names[1].endswith('_obs'), "unexpected column order"
        # column indices: y=0, gm=1, lf start=2, no hf; set hf idx to tensor_width for downstream math
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
                                        n_workers=1)
        self.node_ctx_dir = node_ctx_dir

    def __len__(self):
        return len(self.lstm_dataset)

    def __getitem__(self, idx):

        y, gm, seq, target_station, day_int = self.lstm_dataset[idx]

        source_stations = self.edge_map[target_station]
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

        if self.node_ctx_dir is not None:
            ctx = []
            for stn in source_stations:
                p = os.path.join(self.node_ctx_dir, stn, f"{day_int}.npy")
                v = np.load(p)
                ctx.append(torch.from_numpy(v))
            ctx_t = torch.stack(ctx, dim=0).float()
            zero_row = torch.zeros_like(ctx_t[0])
            node_ctx = torch.cat([zero_row.unsqueeze(0), ctx_t], dim=0)
            assert node_ctx.shape[0] == x.shape[0], "node_lstm rows must match nodes"
            graph.node_lstm = node_ctx

        return graph, y, seq


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
