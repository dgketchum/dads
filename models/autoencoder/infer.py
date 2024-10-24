import os
import json

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.autoencoder.weather_encoder import WeatherAutoencoder
from models.scalers import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device_name = None
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f'Using GPU: {device_name}')
else:
    print('CUDA is not available. PyTorch will use the CPU.')

torch.set_float32_matmul_precision('medium')
torch.cuda.get_device_name(torch.cuda.current_device())


class InferenceDataset(Dataset):
    def __init__(self, file_path, expected_width, data_width, chunk_size, scaler):
        self.chunk_size = chunk_size
        self.data_width = data_width
        self.scaler = scaler

        self.data = torch.load(file_path, weights_only=True)
        if self.data.shape[2] != expected_width:
            raise ValueError(f"Shape mismatch in {file_path}. "
                             f"Expected {expected_width} columns, got {self.data.shape[2]}")

    def scale_chunk(self, chunk):
        chunk_np = chunk.numpy()
        reshaped_chunk = chunk_np.reshape(-1, chunk_np.shape[-1])
        scaled_chunk = self.scaler.transform(reshaped_chunk)
        return torch.from_numpy(scaled_chunk.reshape(chunk_np.shape))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        chunk[:, :self.data_width] = self.scale_chunk(chunk[:, :self.data_width])
        return chunk


def infer_embeddings(model_dir, data_dir, metadata_path, embedding_path, plot=False):
    with open(metadata_path, 'r') as f:
        meta = json.load(f)

    chunk_size = meta['chunk_size']
    tensor_width = len(meta['column_order'])
    data_width = len(meta['data_columns'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(model_dir, 'best_model.ckpt')
    model = WeatherAutoencoder.load_from_checkpoint(model_path,
                                                    input_dim=tensor_width,
                                                    output_dim=5,
                                                    latent_size=128,
                                                    hidden_size=128,
                                                    dropout=0.1,
                                                    **meta)
    model.to(device)
    model.eval()

    scaler_path = os.path.join(model_dir, 'scaler.json')
    with open(scaler_path, 'r') as f:
        dct = json.load(f)

    scaler = MinMaxScaler(out_range=(0, 1.0), axis=0)
    scaler.bias = np.array(dct['bias']).reshape(1, -1)
    scaler.scale = np.array(dct['scale']).reshape(1, -1)

    embeddings = {}
    all_embeddings = []
    station_names = []

    train_files = [os.path.join(data_dir, 'train', f) for f in os.listdir(os.path.join(data_dir, 'train'))]
    val_files = [os.path.join(data_dir, 'val', f) for f in os.listdir(os.path.join(data_dir, 'val'))]
    files_ = train_files + val_files

    for i, f in enumerate(files_):
        station_name = os.path.basename(f).replace('.pth', '')
        file_path = os.path.join(data_dir, f)

        dataset = InferenceDataset(file_path, tensor_width, data_width, chunk_size, scaler)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        # TODO: check validation with the reconstitute model
        # also check clustering to ensure vector diversity

        station_embeddings = []
        for batch in dataloader:
            x = batch.to(device)
            x = torch.nan_to_num(x)
            _, z = model(x)
            station_embeddings.append(z.unsqueeze(2).detach().cpu())

        # Average embeddings if there are multiple samples per station
        mean_embedding = torch.cat(station_embeddings, dim=0).mean(dim=0)
        embeddings[station_name] = mean_embedding.tolist()
        all_embeddings.append(mean_embedding)
        station_names.append(station_name)
        print('{:.3f}'.format(mean_embedding.mean().item()), station_name)

    if plot:
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

    print('========================== training autoencoder ==========================')

    param_dir = os.path.join(training, 'autoencoder')
    pth_ = os.path.join(param_dir, 'pth')
    metadata_ = os.path.join(param_dir, 'training_metadata.json')

    model_run = os.path.join(param_dir, 'checkpoints', '10231635')
    model_ = os.path.join(model_run, 'best_model.ckpt')
    scaler_ = os.path.join(model_run, 'scaler.json')
    embeddings_file = os.path.join(model_run, 'embeddings.json')

    infer_embeddings(model_run, pth_, metadata_, embeddings_file, plot=False)
# ========================= EOF ====================================================================
