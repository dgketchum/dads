import os
import json

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from models.autoencoder.weather_encoder import WeatherAutoencoder
from models.scalers import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
    def __init__(self, file_path, expected_width, selected_indices, chunk_size, scaler, expected_columns, window_stride):
        self.chunk_size = chunk_size
        self.selected_indices = selected_indices
        self.scaler = scaler
        self.expected_width = expected_width
        self.expected_columns = expected_columns
        self.window_stride = int(window_stride)

        self.data = []
        self.months = []

        df = pd.read_parquet(file_path)
        if self.expected_columns is not None:
            assert list(df.columns) == list(self.expected_columns), "Parquet columns do not match training metadata"
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Parquet file index must be a DatetimeIndex")
        df = df.sort_index()
        df = df[~((df.index.month == 2) & (df.index.day == 29))]
        if df.shape[1] != expected_width:
            self.data = []
            return

        # continuous daily index across full record and sliding windows
        start = pd.Timestamp(df.index.min().year, df.index.min().month, df.index.min().day)
        end = pd.Timestamp(df.index.max().year, df.index.max().month, df.index.max().day)
        idx = pd.date_range(start, end, freq='D')
        idx = idx[~((idx.month == 2) & (idx.day == 29))]
        sub = df.reindex(idx)

        if len(sub) >= self.chunk_size:
            for start_i in range(0, len(sub) - self.chunk_size + 1, self.window_stride):
                win = sub.iloc[start_i:start_i + self.chunk_size]
                arr = torch.as_tensor(win.values, dtype=torch.float32)
                self.data.append(arr)
                center = win.index[self.chunk_size // 2]
                self.months.append(int(center.month))

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


def infer_embeddings(model_dir, data_dir, metadata_path, embedding_path, scaler_json, plot=False):
    """Infer per-station embeddings with the trained autoencoder.

    Uses the training metadata to enforce column identity and scaling, then averages
    latent vectors across yearly chunks per station. Scaler comes from the graph's
    train_ids (train-only). Writes a JSON mapping from
    station id to embedding vector suitable for DADS.
    """
    with open(metadata_path, 'r') as f:
        meta = json.load(f)

    sequence_length = meta['sequence_length']
    window_stride = int(meta.get('window_stride', sequence_length))
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
        zero_target_in_encoder=False,
        target_input_idx=0,
    )
    model.to(device)
    model.eval()

    assert scaler_json is not None and os.path.exists(scaler_json), "scaler_json required and must exist (from graph)"
    with open(scaler_json, 'r') as f:
        sp = json.load(f)
    scaler = MinMaxScaler()
    scaler.bias = np.array(sp['bias']).reshape(1, -1)
    scaler.scale = np.array(sp['scale']).reshape(1, -1)

    embeddings = {}
    seasonal_embeddings = {}
    all_embeddings = []
    station_names = []

    files_ = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.parquet')]

    for i, file_path in enumerate(tqdm(files_, desc='Inferring station embeddings')):

        station_name = os.path.basename(file_path).replace('.parquet', '')

        try:
            dataset = InferenceDataset(file_path, expected_width, selected_indices, sequence_length, scaler, expected_columns, window_stride)
        except Exception as e:
            print(f"Skipping {station_name}: {e}")
            continue
        if len(dataset) == 0:
            # no valid chunks; skip this file
            print(f"Skipping {station_name}: no valid chunks found.")
            continue
        try:
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

            station_embeddings = []
            station_embeddings_by_month = {}
            with torch.no_grad():
                k = 0
                for batch in dataloader:
                    x = batch.to(device)
                    x = torch.nan_to_num(x)
                    _, z = model(x)
                    station_embeddings.append(z.unsqueeze(2).detach().cpu())
                    m = dataset.months[k]
                    if m not in station_embeddings_by_month:
                        station_embeddings_by_month[m] = []
                    station_embeddings_by_month[m].append(z.detach().cpu())
                    k += 1

            # Average embeddings if there are multiple samples per station
            mean_embedding = torch.cat(station_embeddings, dim=0).mean(dim=0)
            # Per-month averages
            month_means = {}
            for m, arrs in station_embeddings_by_month.items():
                collapsed = []
                for t in arrs:
                    if t.dim() == 2 and t.shape[0] == 1:
                        collapsed.append(t.squeeze(0))
                    else:
                        collapsed.append(t)
                mm = torch.stack(collapsed, dim=0).mean(dim=0)
                month_means[str(int(m))] = mm.tolist()
        except Exception as e:
            print(f"Skipping {station_name}: {e}")
            continue
        # std_embedding = torch.cat(station_embeddings, dim=0).std(dim=0)
        embeddings[station_name] = mean_embedding.tolist()
        all_embeddings.append(mean_embedding)
        station_names.append(station_name)
        seasonal_embeddings[station_name] = month_means
        # print('{:.3f}'.format(mean_embedding.mean().item()), station_name)
        # print('{:.3f}'.format(std_embedding.mean().item()), station_name)
        # print(f'...of {len(station_embeddings)}')
        # print('')

    if plot and all_embeddings:
        emb_arr = torch.cat(all_embeddings, dim=1).T.numpy()
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(emb_arr)
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
    seasonal_path = embedding_path.replace('.json', '_by_month.json')
    with open(seasonal_path, 'w') as fp:
        json.dump(seasonal_embeddings, fp, indent=4)

    # Optional quick probe: monthly mean R^2 using seasonal embeddings
    try:
        probe_monthly_means(seasonal_path, data_dir, expected_columns[0])
    except Exception:
        pass


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

    model_run = os.path.join(param_dir, 'checkpoints', '10211323')
    model_ = os.path.join(model_run, 'best_model.ckpt')
    scaler_json_ = os.path.join(training, 'scalers', f"{variable_}.json")
    metadata_ = os.path.join(model_run, 'training_metadata.json')
    embeddings_file = os.path.join(model_run, 'embeddings.json')

    infer_embeddings(model_run, parq_, metadata_, embeddings_file, scaler_json_, plot=False)


def _safe_monthly_mean(parquet_path, target_col):
    try:
        df = pd.read_parquet(parquet_path, columns=[target_col])
        if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
            return None
        df = df.sort_index()
        df = df[~((df.index.month == 2) & (df.index.day == 29))]
        g = df.groupby(df.index.month)[target_col].mean()
        return g.to_dict()
    except Exception:
        return None


def probe_monthly_means(seasonal_embeddings_path, data_dir, target_col):
    """Linear probe: predict station monthly mean of target from seasonal embeddings.

    Prints per-month R^2 across stations and overall average R^2.
    """
    with open(seasonal_embeddings_path, 'r') as f:
        seasonal = json.load(f)
    # Build features and targets per month
    months = [str(i) for i in range(1, 13)]
    Xm, ym = {}, {}
    for st in os.listdir(data_dir):
        if not st.endswith('.parquet'):
            continue
        sid = os.path.splitext(st)[0]
        if sid not in seasonal:
            continue
        z_by_m = seasonal[sid]
        m_means = _safe_monthly_mean(os.path.join(data_dir, st), target_col)
        if m_means is None:
            continue
        for m in months:
            if m in z_by_m and int(m) in m_means and z_by_m[m] is not None:
                zm = np.array(z_by_m[m], dtype=float).reshape(1, -1)
                yv = float(m_means[int(m)])
                Xm.setdefault(m, []).append(zm.squeeze(0))
                ym.setdefault(m, []).append(yv)
    if not Xm:
        print('[Probe] No seasonal embeddings + monthly means overlap; skipping probe.')
        return
    rs = []
    for m in months:
        X = np.array(Xm.get(m, []), dtype=float)
        y = np.array(ym.get(m, []), dtype=float)
        if len(X) < 8:  # need enough stations per month
            continue
        # simple linear regression
        model = LinearRegression()
        model.fit(X, y)
        yhat = model.predict(X)
        r2 = r2_score(y, yhat)
        rs.append(r2)
        print(f"[Probe][month={m}] stations={len(X)} R2={r2:.3f}")
    if rs:
        print(f"[Probe] Monthly mean R2 avg={np.mean(rs):.3f} over {len(rs)} months")
# ========================= EOF ====================================================================
