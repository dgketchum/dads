import os
import json
import resource
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, Dataset

from models.lstm.train import PTHLSTMDataset

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
    def __init__(self, lstm_dataset, embedding_dict, edge_index):
        """
        Dataset for the WeatherGNN model.

        Args:
            lstm_dataset (LSTMDataset): An instance of your LSTMDataset class.
            embedding_dict (dict): Dictionary mapping station names to their embeddings.
            edge_index (torch.Tensor): Tensor of shape [2, num_edges] representing the
                                       connections between stations.
        """
        self.lstm_dataset = lstm_dataset
        self.embedding_dict = embedding_dict
        self.edge_index = edge_index

    def __len__(self):
        return len(self.lstm_dataset)

    def __getitem__(self, idx):
        # Get y, gm, lf, hf, and station_name from the LSTMDataset
        y, gm, lf, hf, station_name = self.lstm_dataset[idx]

        # Get the embedding for the station
        embedding = self.embedding_dict[station_name]

        # You might need to add additional processing or scaling here
        # ...

        return y, gm, lf, hf, embedding


def train_model(dirpath, lstm_path, embeddings, metadata, batch_size=1, learning_rate=0.01,
                n_workers=1, logging_csv=None):
    """"""

    with open(metadata, 'r') as f:
        meta = json.load(f)

    model = DadsMetGNN(lstm_path, output_dim=1, edge_emb_dim=6, hidden_dim=64,
                       num_gnn_layers=5, dropout=0.2, learning_rate=1e-3, freeze_lstm=True)

    train_dataset = WeatherGNNDataset(lstm_path, embeddings, edge_index)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers)

    val_dataset = WeatherGNNDataset(lstm_path, embeddings, edge_index)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="best_model",
        dirpath=dirpath,
        save_top_k=1,
        mode="min"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=100,
        verbose=False,
        mode="min",
        check_finite=True,
    )

    trainer = pl.Trainer(max_epochs=1000, callbacks=[checkpoint_callback, early_stop_callback],
                         accelerator='gpu', devices=1)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    target_var = 'vpd'

    if device_name == 'NVIDIA GeForce RTX 2080':
        workers = 6
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

    param_dir = os.path.join(training, target_var)
    pth_ = os.path.join(param_dir, 'pth')
    metadata_ = os.path.join(param_dir, 'training_metadata.json')

    now = datetime.now().strftime('%m%d%H%M')
    chk = os.path.join(param_dir, 'checkpoints', now)
    os.mkdir(chk)
    logger_csv = os.path.join(chk, 'training_{}.csv'.format(now))

    train_model(chk, pth_, metadata_, batch_size=512, learning_rate=0.001, n_workers=workers, logging_csv=logger_csv)
# ========================= EOF ====================================================================
