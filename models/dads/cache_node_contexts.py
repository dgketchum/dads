import os
import json
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from models.lstm.lstm import LSTMPredictor
from models.scalers import MinMaxScaler

device_name = None
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f'Using GPU: {device_name}')
else:
    print('CUDA is not available. PyTorch will use the CPU.')

torch.set_float32_matmul_precision('medium')
torch.cuda.get_device_name(torch.cuda.current_device())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_features(df):
    y = df.iloc[:, 0].values.astype(np.float32)
    obs = y.reshape(-1, 1)
    obs_shift = obs.copy()
    if len(obs_shift) > 1:
        obs_shift[1:, 0] = obs[:-1, 0]
    input_width = df.shape[1] - 2
    lf = np.repeat(obs_shift, input_width, axis=1)
    return lf


def cache_node_contexts(lstm_model_dir, parquet_dir, scaler_json, out_dir, chunk_size):
    with open(scaler_json, 'r') as f:
        p = json.load(f)
    scaler = MinMaxScaler()
    scaler.bias = np.array(p['bias']).reshape(1, -1)
    scaler.scale = np.array(p['scale']).reshape(1, -1)

    ckpt = os.path.join(lstm_model_dir, 'best_model.ckpt')
    meta_path = os.path.join(lstm_model_dir, 'training_metadata.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    num_bands_ = int(meta['num_bands'])
    expansion_factor_ = int(meta.get('expansion_factor', 2))
    lstm = LSTMPredictor.load_from_checkpoint(ckpt, num_bands=num_bands_, learning_rate=0.001,
                                              expansion_factor=expansion_factor_, log_csv=None)
    for p_ in lstm.parameters():
        p_.requires_grad = False
    lstm.eval()
    # ensure model and tensors are on the same device
    lstm.to(device)

    captured = {}

    def hook(_, __, output):
        captured['ctx'] = output.detach().cpu().numpy()

    h = lstm.fc1.register_forward_hook(hook)

    files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    for fp in tqdm(files, desc="Caching node contexts", unit="file"):
        stn = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_parquet(fp)
        df = df.dropna()
        if len(df) < chunk_size:
            continue
        df['day_int'] = df.index.to_julian_date().astype(np.int32)
        df['day_diff'] = df['day_int'].diff()
        arr = df.to_numpy(dtype=np.float32)
        is_consec = arr[1:, -1] == 1.0
        win = chunk_size - 1
        if len(is_consec) < win:
            continue
        conv = np.convolve(is_consec, np.ones(win, dtype=int), mode='valid')
        starts = np.where(conv == win)[0]
        if starts.size == 0:
            continue
        os.makedirs(os.path.join(out_dir, stn), exist_ok=True)
        for i in starts:
            end_idx = i + chunk_size
            sub = df.iloc[i:end_idx, :]
            day_int = int(sub['day_int'].iloc[-1])
            feats = _build_features(sub.iloc[:, :len(df.columns)])
            x = torch.tensor(scaler.transform(feats), dtype=torch.float32)
            x = x[:, 2:]
            x = x.to(device)
            with torch.no_grad():
                _ = lstm(x)
            ctx = captured['ctx']
            np.save(os.path.join(out_dir, stn, f"{day_int}.npy"), ctx.squeeze())

    h.remove()


if __name__ == '__main__':

    training = '/data/ssd2/dads/training'

    variable_ = 'tmax'
    target_var_ = f'{variable_}_obs'

    lstm_model_dir = os.path.join(training, 'lstm', 'checkpoints', f'{variable_}_20251004_1650')
    parquet_dir = os.path.join(training, 'parquet', target_var_)
    scaler_json =  os.path.join(lstm_model_dir, 'scaler.json')
    out_dir = os.path.join(training, 'dads''node_ctx')

    chunk_size = 12
    cache_node_contexts(lstm_model_dir, parquet_dir, scaler_json, out_dir, chunk_size)
# ========================= EOF ====================================================================
