import os
import json
import resource
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import pandas as pd
import torch
from torch_geometric.data import Data
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from models.dads.dads_gnn import DadsMetGNN
from models.lstm.dataset import LSTMDataset
from models.scalers import MinMaxScaler
from prep.build_variable_scaler import load_variable_scaler, get_variable_scaler_path

torch.set_float32_matmul_precision('medium')
torch.cuda.get_device_name(torch.cuda.current_device())

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

device_name = None
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f'Using GPU: {device_name}')
else:
    print('CUDA is not available. PyTorch will use the CPU.')


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
        embedding_arr = torch.cat([v[1] for v in embeddings], dim=1).permute(-1, 0)
        min_ = embedding_arr.min(dim=0).values
        max_ = embedding_arr.max(dim=0).values
        embedding_arr = (embedding_arr - min_) / (max_ - min_)

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

        filter_files = [f for f in lstm_input_files if os.path.basename(f).replace('.pth', '') in attributes.keys()]

        self.edge_map = attributes

        data_frequency = lstm_meta['data_frequency']
        self.model = 'lstm'
        # Determine feature names and tensor width from first file
        first_file = filter_files[0]
        feature_names = pd.read_parquet(first_file).columns.tolist()
        num_features = len(feature_names)
        assert feature_names[0].endswith('_obs') and not feature_names[1].endswith('_obs'), "unexpected column order"
        # column indices: y=0, gm=1, lf start=2, no hf; set hf idx to tensor_width for downstream math
        self.tensor_width = num_features
        self.column_indices = (0, 1, 2, self.tensor_width)
        chunk_size = lstm_meta['chunk_size']

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


def train_model(dirpath, lstm_data, lstm_model, embeddings, edge_info, nodes=5, batch_size=1, strided=False,
                dropout=0.2, learning_rate=0.01, n_workers=1, logging_csv=None, device='gpu', sample=None,
                node_ctx_dir=None):
    """"""

    if sample is None:
        sample = None, None

    metadata_ = os.path.join(lstm_data, 'training_metadata.json')
    with open(metadata_, 'r') as f:
        meta = json.load(f)

    meta['model'] = 'lstm'
    pth = os.path.join(lstm_data, 'pth')

    # Prefer variable-specific scaler under training/scalers/{var}.json
    var_name = os.path.basename(lstm_data)
    training_root = os.path.dirname(os.path.dirname(lstm_data))
    parquet_root = os.path.join(training_root, 'parquet')
    _scaler_obj, scaler, _ = load_variable_scaler(parquet_root, var_name)

    tdir = os.path.join(pth, 'train')
    t_files = [os.path.join(tdir, f) for f in os.listdir(tdir)]
    train_edges = os.path.join(edge_info, 'train_edge_index.json')
    train_attr = os.path.join(edge_info, 'train_edge_attr.json')

    train_dataset = DadsDataset(nodes, t_files, meta, embeddings, train_edges, train_attr,
                                scaler=scaler, sample=sample[0], node_ctx_dir=node_ctx_dir)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers)

    vdir = os.path.join(pth, 'val')
    v_files = [os.path.join(vdir, f) for f in os.listdir(vdir)]
    val_edges = os.path.join(edge_info, 'val_edge_index.json')
    val_attr = os.path.join(edge_info, 'val_edge_attr.json')

    val_dataset = DadsDataset(nodes, v_files, meta, embeddings, val_edges, val_attr,
                              scaler=scaler, sample=sample[1], node_ctx_dir=node_ctx_dir)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=n_workers)

    meta['column_indices'] = train_dataset.column_indices
    meta['tensor_width'] = train_dataset.tensor_width

    meta['scaler'] = train_dataset.lstm_dataset.scaler
    edge_dim = next(iter(train_dataset.edge_attr.values())).shape[-1]  # infer edge attribute width
    model = DadsMetGNN(lstm_model, output_dim=1, n_nodes=nodes, hidden_dim=1024,
                       edge_attr_dim=edge_dim, dropout=dropout, learning_rate=learning_rate,
                       log_csv=logging_csv, **meta)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='best_model',
        dirpath=dirpath,
        save_top_k=1,
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=100,
        verbose=False,
        mode='min',
        check_finite=True,
    )

    trainer = pl.Trainer(max_epochs=100, callbacks=[checkpoint_callback, early_stop_callback],
                         accelerator=device, devices=1)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':

    target_var = 'tmax'

    d = '/home/dgketchum/data/IrrigationGIS/dads'
    training = '/home/dgketchum/training'

    print('========================== modeling {} =========================='.format(target_var))

    # lstm model
    param_dir = os.path.join(training, 'lstm', target_var)
    model_dir = os.path.join(param_dir, 'checkpoints', '10221508')

    # graph
    dads = os.path.join(training, 'dads')
    edges = os.path.join(dads, 'graph')

    # climate embedding
    encoder_dir = os.path.join(training,  'autoencoder', 'checkpoints', '10011529')

    now = datetime.now().strftime('%m%d%H%M')
    chk = os.path.join(dads, 'checkpoints', now)
    os.mkdir(chk)
    logger_csv = os.path.join(chk, 'training_{}.csv'.format(now))

    workers = 12
    device_ = 'gpu'

    node_ctx_dir = os.path.join(dads, 'node_ctx')
    train_model(chk, param_dir, model_dir, encoder_dir, edges, batch_size=256, nodes=5, dropout=0.5, strided=True,
                learning_rate=0.001, n_workers=workers, logging_csv=logger_csv, device=device_, sample=None,
                node_ctx_dir=node_ctx_dir)
# ========================= EOF ====================================================================
