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
from models.simple_lstm.train import PTHLSTMDataset

from models.dads.dads_gnn import DadsMetGNN

device_name = None
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f'Using GPU: {device_name}')
else:
    print('CUDA is not available. PyTorch will use the CPU.')

torch.set_float32_matmul_precision('medium')
torch.cuda.get_device_name(torch.cuda.current_device())

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


class WeatherGNNDataset(Dataset):
    def __init__(self, lstm_input_files, lstm_meta, embedding_dir, edge_map_file, edge_attr_file):
        """"""

        with open(edge_map_file, 'r') as f:
            self.edge_map = json.load(f)

        with open(edge_attr_file, 'r') as f:
            edge_attr = json.load(f)
        self.edge_attr = {k: torch.tensor(v) for k, v in edge_attr.items()}

        embedding_dict_file = os.path.join(embedding_dir, 'embeddings.json')
        with open(embedding_dict_file, 'r') as f:
            embeddings = json.load(f)
        self.embedding_dict = {k: np.array(v) for k, v in embeddings.items()}

        data_frequency = lstm_meta['data_frequency']
        idxs = (0, 1, 2, len(data_frequency))
        tensor_width = data_frequency.count('lf')
        chunk_size = lstm_meta['chunk_size']

        # restrict input data to those stations for which we have embeddings
        embed_keys = list(self.embedding_dict.keys())
        edge_keys = list([k for k, v in self.edge_map.items() if all([vv in embed_keys for vv in v])])

        filter_files = [f for f in lstm_input_files if os.path.basename(f).replace('.pth', '') in embed_keys]
        filter_files = [f for f in filter_files if os.path.basename(f).replace('.pth', '') in edge_keys]

        self.lstm_dataset = PTHLSTMDataset(file_paths=filter_files,
                                           col_index=idxs,
                                           expected_width=tensor_width,
                                           chunk_size=chunk_size,
                                           return_station_name=True)

    def __len__(self):
        return len(self.lstm_dataset)

    def __getitem__(self, idx):
        y, gm, sequence, target_station = self.lstm_dataset[idx]

        source_stations = self.edge_map[target_station]
        source_embeddings = [self.embedding_dict[stn] for stn in source_stations]

        x = torch.tensor(np.hstack([source_embeddings]), dtype=torch.float)

        source_indices = torch.arange(1, len(source_stations) + 1)
        target_index = torch.tensor([0] * len(source_stations))
        edge_index = torch.stack([source_indices, target_index], dim=0)

        to_point = self.edge_attr[target_station]
        from_point = [to_point - self.edge_attr[stn] for stn in source_stations]
        edge_attr = torch.stack(from_point, dim=0)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data, y, gm, sequence


def train_model(dirpath, lstm_data, lstm_model, embeddings, edge_info, batch_size=1, learning_rate=0.01,
                n_workers=1, logging_csv=None):
    """"""

    metadata_ = os.path.join(lstm_data, 'training_metadata.json')
    with open(metadata_, 'r') as f:
        meta = json.load(f)

    tdir = os.path.join(lstm_data, 'pth', 'train')
    t_files = [os.path.join(tdir, f) for f in os.listdir(tdir)]
    train_edges = os.path.join(edge_info,  'train_edge_index.json')
    train_attr = os.path.join(edge_info,  'train_edge_attr.json')
    train_dataset = WeatherGNNDataset(t_files, meta, embeddings, train_edges, train_attr)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers)

    batch = next(iter(train_dataloader))
    data, y, gm, sequence = batch

    vdir = os.path.join(lstm_data, 'pth', 'val')
    v_files = [os.path.join(vdir, f) for f in os.listdir(vdir)]
    val_edges = os.path.join(edge_info,  'val_edge_index.json')
    val_attr = os.path.join(edge_info,  'val_edge_attr.json')
    val_dataset = WeatherGNNDataset(v_files, meta, embeddings, val_edges, val_attr)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=n_workers)

    model = DadsMetGNN(lstm_model, output_dim=1, edge_emb_dim=6, hidden_dim=64,
                       num_gnn_layers=5, dropout=0.2, learning_rate=1e-3, freeze_lstm=True,
                       **meta)

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

    trainer = pl.Trainer(max_epochs=1000, callbacks=[checkpoint_callback, early_stop_callback],
                         accelerator='gpu', devices=1)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    target_var = 'mean_temp'

    if device_name == 'NVIDIA GeForce RTX 2080':
        workers = 0
    elif device_name == 'NVIDIA RTX A6000':
        workers = 6
    else:
        raise NotImplementedError('Specify the machine this is running on')

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
    param_dir = os.path.join(training, 'lstm_simple', target_var)
    model_dir = os.path.join(param_dir, 'checkpoints', '10041415')

    # graph
    edges = os.path.join(training, 'dads', 'graph')

    # climate embedding
    encoder_dir = os.path.join(training, 'autoencoder', 'checkpoints', '10041435')

    now = datetime.now().strftime('%m%d%H%M')
    chk = os.path.join(param_dir, 'checkpoints', now)
    # os.mkdir(chk)
    # logger_csv = os.path.join(chk, 'training_{}.csv'.format(now))
    logger_csv = None

    train_model(chk, param_dir, model_dir, encoder_dir, edges,
                batch_size=2, learning_rate=0.001, n_workers=workers, logging_csv=logger_csv)
# ========================= EOF ====================================================================
