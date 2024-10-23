import json
import os
import resource
import shutil
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

from models.autoencoder.weather_encoder import WeatherAutoencoder
from models.scalers import MinMaxScaler

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


class WeatherDataset(Dataset):
    def __init__(self, file_paths, expected_width, data_width, chunk_size,
                 features_path, target_indices=None, similarity_file=None, transform=None):

        self.chunk_size = chunk_size
        self.transform = transform
        self.data_width = data_width
        self.target_indices = target_indices

        self.station_idx = []

        self.features_path = features_path
        self.station_features, self.stations = self.load_station_features(features_path, file_paths)
        self.similarity_matrix, ub, lb = self.calculate_similarity_matrix(similarity_file)
        self.similarity_thresholds = ub, lb

        all_data = []
        for file_path in file_paths:

            staid = os.path.basename(file_path).replace('.pth', '')
            if staid not in self.stations:
                continue

            data = torch.load(file_path, weights_only=True)

            if data.shape[2] != expected_width:
                print(f"Skipping {file_path}, shape mismatch. Expected {expected_width} columns, got {data.shape[2]}")
                continue

            all_data.append(data)
            self.station_idx.extend([self.stations.index(staid)] * len(data))

        self.data = torch.cat(all_data, dim=0)

        self.scaler = MinMaxScaler()
        self.scaler.fit(self.get_valid_data_for_scaling())

    @staticmethod
    def load_station_features(features_path, file_paths):

        seq_stations = [os.path.basename(fp).replace('.pth', '') for fp in file_paths]

        with open(features_path, 'r') as f:
            features = json.load(f)

        data, stations = [], []
        for s in seq_stations:
            try:
                data.append(features[s])
                stations.append(s)
            except KeyError:
                continue

        features = np.array(data)
        return features, stations

    def calculate_similarity_matrix(self, similarity_file):
        num_stations = len(self.station_features)

        if os.path.exists(similarity_file):
            similarity_matrix = np.loadtxt(similarity_file, dtype=float)
            self.stations = np.loadtxt(similarity_file.replace('.np', '.txt'), dtype=str)

        else:

            similarity_matrix = np.zeros((num_stations, num_stations))

            for i in range(num_stations):
                for j in range(i, num_stations):
                    features_i = np.array(self.station_features[i])
                    features_j = np.array(self.station_features[j])
                    similarity = F.cosine_similarity(
                        torch.from_numpy(features_i), torch.from_numpy(features_j), dim=0
                    ).item()
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        min_value = np.min(similarity_matrix)
        max_value = np.max(similarity_matrix)
        similarity_matrix = (similarity_matrix - min_value) / (max_value - min_value)

        ub, lb = find_similarity_thresholds(similarity_matrix, 0.7)

        return similarity_matrix, ub, lb

    def get_positive_pair(self, idx):
        station_idx = self.station_idx[idx]
        for j, other_idx in enumerate(self.station_idx):
            if other_idx == station_idx:
                continue
            if self.similarity_matrix[station_idx, other_idx] > self.similarity_thresholds[1]:
                return self.data[j]
        return None

    def get_negative_pair(self, idx):
        s_idx = self.station_idx[idx]
        for j, other_idx in enumerate(self.station_idx):
            if other_idx == s_idx:
                continue
            if self.similarity_matrix[s_idx, other_idx] < self.similarity_thresholds[0]:
                return self.data[j]
        return None

    def scale_chunk(self, chunk):
        chunk_np = chunk.numpy()
        reshaped_chunk = chunk_np.reshape(-1, chunk_np.shape[-1])
        scaled_chunk = self.scaler.transform(reshaped_chunk)
        return torch.from_numpy(scaled_chunk.reshape(chunk_np.shape))

    def get_valid_data_for_scaling(self):
        valid_data = []
        for chunk in self.data:
            chunk_without_pe = chunk[:, :self.data_width]
            valid_rows = chunk_without_pe[~torch.isnan(chunk_without_pe).any(dim=1)]
            valid_data.append(valid_rows)
        return torch.cat(valid_data, dim=0).numpy()

    def save_scaler(self, scaler_path):

        bias_ = self.scaler.bias.flatten().tolist()
        scale_ = self.scaler.scale.flatten().tolist()
        dct = {'bias': bias_, 'scale': scale_}
        with open(scaler_path, 'w') as fp:
            json.dump(dct, fp, indent=4)
        print(f"Scaler saved to {scaler_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        chunk = self.data[idx]

        chunk[:, :self.data_width] = self.scale_chunk(chunk[:, :self.data_width])
        if isinstance(self.target_indices[0], tuple):
            target = diff_pairs(chunk, self.target_indices)
        else:
            target = chunk[:, self.target_indices]

        # pytorch has counterintuitive mask logic (opposite)
        # i.e., False where there is valid data
        mask = ~torch.isnan(chunk[:, 0]).unsqueeze(1).repeat_interleave(chunk.size(1), dim=1)

        positive_chunk = self.get_positive_pair(idx)
        negative_chunk = self.get_negative_pair(idx)

        return chunk, target, mask, positive_chunk, negative_chunk


def diff_pairs(tensor, pairs):
    diffs = []
    for i, j in pairs:
        diff = tensor[:, i] - tensor[:, j]
        diffs.append(diff.unsqueeze(-1))
    return torch.cat(diffs, dim=-1)


def find_similarity_thresholds(similarity_matrix, percentage=0.5):
    similarity_values = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

    sorted_values = np.sort(similarity_values)

    lower_index = int(len(sorted_values) * (percentage / 2))
    upper_index = int(len(sorted_values) * (1 - percentage / 2))

    lower_threshold = sorted_values[lower_index]
    upper_threshold = sorted_values[upper_index]

    return upper_threshold, lower_threshold


def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return default_collate(batch)


def train_model(dirpath, pth, metadata, feature_dir, batch_size=64, learning_rate=0.001, n_workers=1, device='gpu',
                bias_target=False, logging_csv=None):
    """"""

    with open(metadata, 'r') as f:
        meta = json.load(f)

    cols = meta['column_order']
    chunk_size = meta['chunk_size']
    tensor_width = len(cols)
    data_width = len(meta['data_columns'])

    variables = list(set(['_'.join(v.split('_')[:-1]) for v in cols if 'pe' not in v]))

    if bias_target:
        difference_idx = [(cols.index(f'{v}_nl'), cols.index(f'{v}_obs')) for v in variables]
    else:
        difference_idx = [cols.index(f'{v}_obs') for v in variables]

    tdir = os.path.join(pth, 'train')
    t_files = [os.path.join(tdir, f) for f in os.listdir(tdir)]
    train_features = os.path.join(feature_dir, 'train_edge_attr.json')
    train_similarity = os.path.join(feature_dir, 'train_similarity.np')
    train_dataset = WeatherDataset(file_paths=t_files,
                                   expected_width=tensor_width,
                                   data_width=data_width,
                                   chunk_size=chunk_size,
                                   target_indices=difference_idx,
                                   features_path=train_features,
                                   similarity_file=train_similarity)

    if logging_csv:
        train_dataset.save_scaler(os.path.join(dirpath, 'scaler.json'))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers,
                                  collate_fn=lambda batch: [x for x in batch if x is not None])

    vdir = os.path.join(pth, 'val')
    v_files = [os.path.join(vdir, f) for f in os.listdir(vdir)]
    val_features = os.path.join(feature_dir, 'val_edge_attr.json')
    val_similarity = os.path.join(feature_dir, 'val_similarity.np')
    val_dataset = WeatherDataset(file_paths=v_files,
                                 expected_width=tensor_width,
                                 data_width=data_width,
                                 chunk_size=chunk_size,
                                 target_indices=difference_idx,
                                 features_path=val_features,
                                 similarity_file=val_similarity)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=n_workers,
                                collate_fn=lambda batch: [x for x in batch if x is not None])

    model = WeatherAutoencoder(input_dim=tensor_width,
                               output_dim=len(difference_idx),
                               latent_size=128,
                               hidden_size=128,
                               dropout=0.1,
                               learning_rate=learning_rate,
                               log_csv=logging_csv,
                               scaler=val_dataset.scaler,
                               **meta)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

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
                         accelerator=device, devices=1)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    variable = 'mean_temp'

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

    print(f'========================== training autoencoder {variable} ==========================')

    param_dir = os.path.join(training, 'autoencoder')
    pth_ = os.path.join(param_dir, 'pth')
    metadata_ = os.path.join(param_dir, 'training_metadata.json')

    # graph
    edges = os.path.join(training, 'dads', 'graph')

    now = datetime.now().strftime('%m%d%H%M')
    chk = os.path.join(param_dir, 'checkpoints', now)
    os.mkdir(chk)
    logger_csv = os.path.join(chk, 'training_{}.csv'.format(now))
    # logger_csv = None
    workers = 6
    device_ = 'gpu'

    train_model(chk, pth_, metadata_, feature_dir=edges, batch_size=16, learning_rate=0.001,
                n_workers=workers, logging_csv=logger_csv, device=device_, bias_target=True)
# ========================= EOF ====================================================================
