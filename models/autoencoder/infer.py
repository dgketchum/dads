import os
import json

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from models.autoencoder.weather_encoder import WeatherAutoencoder
from models.scalers import MinMaxScaler
from prep.build_variable_scaler import load_variable_scaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device_name = None
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f'Using GPU: {device_name}')
else:
    print('CUDA is not available. PyTorch will use the CPU.')

torch.set_float32_matmul_precision('medium')
if torch.cuda.is_available():  # avoid CPU-only crash
    torch.cuda.get_device_name(torch.cuda.current_device())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InferenceDataset(Dataset):
    """Inference-time yearly chunking aligned with training metadata.

    - Requires identical Parquet schema as recorded in training metadata.
    - Selects the same input columns (including RS_MISS_FEATURES when present).
    - Builds 365-day sequences (Feb 29 removed) and applies the saved scaler.
    """
    def __init__(self, file_path, expected_width, selected_indices, chunk_size, scaler, expected_columns):
        self.chunk_size = chunk_size
        self.selected_indices = selected_indices
        self.scaler = scaler
        self.expected_width = expected_width
        self.expected_columns = expected_columns

        # Build yearly 365-day chunks from a Parquet file, mirroring training
        self.data = []

        df = pd.read_parquet(file_path)
        if self.expected_columns is not None:
            assert list(df.columns) == list(self.expected_columns), "Parquet columns do not match training metadata"
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Parquet file index must be a DatetimeIndex")
        df = df.sort_index()
        # drop leap day to enforce 365-day alignment
        df = df[~((df.index.month == 2) & (df.index.day == 29))]
        if df.shape[1] != expected_width:
            # Skip file if width mismatch
            self.data = []
            return

        years = range(df.index.min().year, df.index.max().year + 1)
        for y in years:
            start = pd.Timestamp(y, 1, 1)
            end = pd.Timestamp(y, 12, 31)
            idx = pd.date_range(start, end, freq='D')
            idx = idx[~((idx.month == 2) & (idx.day == 29))]
            sub = df.reindex(idx)
            if len(sub) != self.chunk_size:
                continue
            arr = torch.as_tensor(sub.values, dtype=torch.float32)
            self.data.append(arr)

    def scale_chunk(self, chunk):
        chunk_np = chunk.numpy()
        reshaped_chunk = chunk_np.reshape(-1, chunk_np.shape[-1])
        scaled_chunk = self.scaler.transform(reshaped_chunk)
        return torch.from_numpy(scaled_chunk.reshape(chunk_np.shape))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        full = self.scale_chunk(chunk)
        x = full[:, self.selected_indices]
        return x


def infer_embeddings(model_dir, data_dir, metadata_path, embedding_path, plot=False):
    """Infer per-station embeddings with the trained autoencoder.

    Uses the training metadata to enforce column identity and scaling, then averages
    latent vectors across yearly chunks per station. Writes a JSON mapping from
    station id to embedding vector suitable for DADS.
    """
    with open(metadata_path, 'r') as f:
        meta = json.load(f)

    sequence_length = meta['sequence_length']
    input_dim = meta['input_dim']
    output_dim = meta['output_dim']
    selected_indices = meta['selected_indices']
    expected_width = meta['expected_width']
    expected_columns = meta.get('actual_columns', None)

    model = WeatherAutoencoder.load_from_checkpoint(
        os.path.join(model_dir, f'best_model.ckpt'),
        input_dim=input_dim,
        output_dim=output_dim,
        latent_size=meta['latent_size'],
        hidden_size=meta['hidden_size'],
        dropout=meta['dropout'],
        learning_rate=meta['learning_rate'],
        sequence_length=sequence_length,
        margin=meta['margin'],
        data_columns=meta['data_columns'],
        column_order=meta['column_order'],
    )
    model.to(device)
    model.eval()

    var_name = meta['target_var'].replace('_obs', '')
    parquet_root = os.path.dirname(data_dir)
    scaler, _, _ = load_variable_scaler(parquet_root, var_name, feature_names=expected_columns)

    embeddings = {}
    all_embeddings = []
    station_names = []

    files_ = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.parquet')]

    for i, file_path in enumerate(files_):

        station_name = os.path.basename(file_path).replace('.parquet', '')

        dataset = InferenceDataset(file_path, expected_width, selected_indices, sequence_length, scaler, expected_columns)
        if len(dataset) == 0:
            # No valid yearly chunks; skip this file
            print(f"Skipping {station_name}: no valid 365-day chunks found.")
            continue
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        station_embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch.to(device)
                x = torch.nan_to_num(x)
                _, z = model(x)
                station_embeddings.append(z.unsqueeze(2).detach().cpu())

        # Average embeddings if there are multiple samples per station
        mean_embedding = torch.cat(station_embeddings, dim=0).mean(dim=0)
        # std_embedding = torch.cat(station_embeddings, dim=0).std(dim=0)
        embeddings[station_name] = mean_embedding.tolist()
        all_embeddings.append(mean_embedding)
        station_names.append(station_name)
        # print('{:.3f}'.format(mean_embedding.mean().item()), station_name)
        # print('{:.3f}'.format(std_embedding.mean().item()), station_name)
        # print(f'...of {len(station_embeddings)}')
        # print('')

    all_embeddings = torch.cat(all_embeddings, dim=1).T.numpy()

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    for i, name in enumerate(station_names):
        plt.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

    plt.title("t-SNE Visualization of Station Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()

    with open(embedding_path, 'w') as fp:
        json.dump(embeddings, fp, indent=4)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    variable_ = 'tmax'
    target_var_ = f'{variable_}_obs'

    if device_name == 'NVIDIA GeForce RTX 2080':
        workers = 4
    elif device_name == 'NVIDIA RTX A6000':
        workers = 8
    else:
        raise NotImplementedError('Specify the machine this is running on')

    zoran = '/data/ssd2/dads/training'
    nvm = '/media/nvm/training'
    if os.path.exists(zoran):
        print('Modeling with data from Zoran')
        training = zoran
    elif os.path.exists(nvm):
        print('Modeling with data from NVM drive')
        training = nvm
    else:
        raise NotImplementedError

    param_dir = os.path.join(training, 'autoencoder')
    parq_ = os.path.join(training, 'parquet', target_var_)

    model_run = os.path.join(param_dir, 'checkpoints', '10011529')
    model_ = os.path.join(model_run, 'best_model.ckpt')
    scaler_ = os.path.join(model_run, 'scaler.json')
    metadata_ = os.path.join(model_run, 'training_metadata.json')
    embeddings_file = os.path.join(model_run, 'embeddings.json')

    infer_embeddings(model_run, parq_, metadata_, embeddings_file, plot=False)
# ========================= EOF ====================================================================
