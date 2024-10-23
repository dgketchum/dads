import os
import json
import resource
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
from torch_geometric.data import Data
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from models.dads.dads_gnn import DadsMetGNN

torch.set_float32_matmul_precision('medium')
torch.cuda.get_device_name(torch.cuda.current_device())

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


class DadsDataset(Dataset):
    def __init__(self, n_nodes, lstm_input_files, lstm_meta, embedding_dir, edge_map_file, edge_attr_file,
                 scaler, sample=None):
        """"""

        embedding_dict_file = os.path.join(embedding_dir, 'embeddings.json')
        with open(embedding_dict_file, 'r') as f:
            embeddings = json.load(f)

        embeddings = [(k, torch.tensor(v)) for k, v in embeddings.items()]
        embedding_arr = torch.cat([v[1] for v in embeddings], dim=1).permute(-1, 0)
        min_ = embedding_arr.min(dim=0).values
        max_ = embedding_arr.max(dim=0).values
        embedding_arr = (embedding_arr - min_) / (max_ - min_)

        self.embeddings = {k[0]: embedding_arr[i] for i, k in enumerate(embeddings)}

        with open(edge_attr_file, 'r') as f:
            edge_attr = json.load(f)
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

        file_keys = [os.path.basename(f).replace('.pth', '') for f in lstm_input_files]
        attributes = {k: v for k, v in attributes.items() if k in file_keys}

        if sample:
            keys = list(attributes.keys())
            keys = np.random.choice(keys, sample)
            attributes = {k: v for k, v in attributes.items() if k in keys}

        filter_files = [f for f in lstm_input_files if os.path.basename(f).replace('.pth', '') in attributes.keys()]

        self.edge_map = attributes

        data_frequency = lstm_meta['data_frequency']

        if lstm_meta['model'] == 'lstm':

            from models.lstm.train import PTHLSTMDataset
            self.model = 'lstm'
            # idxs: obs, gm, start_daily: end_daily, start_hourly:
            hf_idx = data_frequency.index('hf')
            self.column_indices = (0, 1, 2, hf_idx)
            self.tensor_width = data_frequency.count('lf') + data_frequency.count('hf')
            chunk_size = lstm_meta['chunk_size']

            self.lstm_dataset = PTHLSTMDataset(file_paths=filter_files,
                                               col_index=self.column_indices,
                                               expected_width=self.tensor_width,
                                               chunk_size=chunk_size,
                                               return_station_name=True,
                                               scaler_json=scaler)

        elif lstm_meta['model'] == 'simple_lstm':

            from models.simple_lstm.train import PTHLSTMDataset
            self.model = 'simple_lstm'
            # target, GridMET, start_daily:
            self.column_indices = (0, 1, 2, len(data_frequency))
            self.tensor_width = data_frequency.count('lf')
            chunk_size = lstm_meta['chunk_size']

            self.lstm_dataset = PTHLSTMDataset(file_paths=filter_files,
                                               col_index=self.column_indices,
                                               expected_width=self.tensor_width,
                                               chunk_size=chunk_size,
                                               return_station_name=True,
                                               scaler_json=scaler)

    def __len__(self):
        return len(self.lstm_dataset)

    def __getitem__(self, idx):

        lf, hf = None, None

        if self.model == 'simple_lstm':
            y, gm, lf, target_station = self.lstm_dataset[idx]
        else:
            y, gm, lf, hf, target_station = self.lstm_dataset[idx]

        source_stations = self.edge_map[target_station]
        source_embeddings = [self.embeddings[stn] for stn in source_stations]

        x = torch.tensor(np.hstack([source_embeddings]), dtype=torch.float)

        source_indices = torch.arange(1, len(source_stations) + 1)
        target_index = torch.tensor([0] * len(source_stations))
        edge_index = torch.stack([source_indices, target_index], dim=0)

        to_point = self.edge_attr[target_station]
        from_point = [to_point - self.edge_attr[stn] for stn in source_stations]
        edge_attr = torch.stack(from_point, dim=0)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        if self.model == 'simple_lstm':
            return data, y, gm, lf
        else:
            return data, y, gm, lf, hf


def train_model(dirpath, lstm_data, lstm_model, embeddings, edge_info, nodes=5, batch_size=1, strided=False,
                dropout=0.2, learning_rate=0.01, n_workers=1, logging_csv=None, device='gpu', sample=None):
    """"""

    if sample is None:
        sample = None, None

    metadata_ = os.path.join(lstm_data, 'training_metadata.json')
    with open(metadata_, 'r') as f:
        meta = json.load(f)

    if strided and 'simple_lstm' in lstm_data:
        meta['model'] = 'simple_lstm'
        pth = os.path.join(lstm_data, 'strided_pth')
    elif 'simple_lstm' in lstm_data:
        meta['model'] = 'simple_lstm'
        pth = os.path.join(lstm_data, 'consecutive_pth')
    else:
        meta['model'] = 'lstm'
        pth = os.path.join(lstm_data, 'pth')

    scaler = os.path.join(lstm_model, 'scaler.json')

    tdir = os.path.join(pth, 'train')
    t_files = [os.path.join(tdir, f) for f in os.listdir(tdir)]
    train_edges = os.path.join(edge_info, 'train_edge_index.json')
    train_attr = os.path.join(edge_info, 'train_edge_attr.json')

    train_dataset = DadsDataset(nodes, t_files, meta, embeddings, train_edges, train_attr,
                                scaler=scaler, sample=sample[0])

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers)

    vdir = os.path.join(pth, 'val')
    v_files = [os.path.join(vdir, f) for f in os.listdir(vdir)]
    val_edges = os.path.join(edge_info, 'val_edge_index.json')
    val_attr = os.path.join(edge_info, 'val_edge_attr.json')

    val_dataset = DadsDataset(nodes, v_files, meta, embeddings, val_edges, val_attr,
                              scaler=scaler, sample=sample[1])

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=n_workers)

    meta['column_indices'] = train_dataset.column_indices
    meta['tensor_width'] = train_dataset.tensor_width

    meta['scaler'] = train_dataset.lstm_dataset.scaler
    model = DadsMetGNN(lstm_model, output_dim=1, n_nodes=nodes, edge_emb_dim=6, hidden_dim=1024,
                       edge_attr_dim=20, dropout=dropout, learning_rate=learning_rate, freeze_lstm=True,
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

    device_name = None
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f'Using GPU: {device_name}')
    else:
        print('CUDA is not available. PyTorch will use the CPU.')

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    target_var = 'mean_temp'

    zoran = '/home/dgketchum/training'
    nvm = '/media/nvm/training'
    if os.path.exists(zoran):
        print('modeling with data from zoran')
        training = zoran
    elif os.path.exists(nvm):
        print('modeling with data from NVM drive')
        training = nvm
    else:
        print('modeling with data from UM drive')
        training = os.path.join(d, 'training')

    print('========================== modeling {} =========================='.format(target_var))

    # lstm model
    param_dir = os.path.join(training, 'lstm', target_var)
    model_dir = os.path.join(param_dir, 'checkpoints', '10221508')

    # graph
    dads = os.path.join(training, 'dads')
    edges = os.path.join(dads, 'graph')

    # climate embedding
    encoder_dir = os.path.join(training, 'autoencoder', 'checkpoints', '10171216')

    now = datetime.now().strftime('%m%d%H%M')
    chk = os.path.join(dads, 'checkpoints', now)
    os.mkdir(chk)
    logger_csv = os.path.join(chk, 'training_{}.csv'.format(now))
    # logger_csv = None

    workers = 12
    device_ = 'gpu'

    train_model(chk, param_dir, model_dir, encoder_dir, edges, batch_size=256, nodes=5, dropout=0.5, strided=True,
                learning_rate=0.001, n_workers=workers, logging_csv=logger_csv, device=device_, sample=None)
# ========================= EOF ====================================================================
