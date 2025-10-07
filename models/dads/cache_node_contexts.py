import os
import json
import time
import multiprocessing as mp
import queue as queue_mod
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from models.lstm.lstm import LSTMPredictor
from models.scalers import MinMaxScaler
from prep.build_variable_scaler import load_variable_scaler

device_name = None
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
else:
    raise ValueError('No GPU available')

torch.set_float32_matmul_precision('medium')
if torch.cuda.is_available():
    torch.cuda.get_device_name(torch.cuda.current_device())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_features(df, bias, scale):
    y = df.iloc[:, 0].values.astype(np.float32)
    y = (y - bias[0, 0]) / scale[0, 0] + 5e-8
    obs = y.reshape(-1, 1)
    obs_shift = obs.copy()
    if len(obs_shift) > 1:
        obs_shift[1:, 0] = obs[:-1, 0]
    input_width = df.shape[1] - 2
    lf = np.repeat(obs_shift, input_width, axis=1)
    return lf


def _producer(file_q, task_q, bias, scale, chunk):
    while True:
        fp = file_q.get()
        if fp is None:
            file_q.task_done()
            break
        try:
            stn = os.path.splitext(os.path.basename(fp))[0]
            df = pd.read_parquet(fp)
            df = df.dropna()
            if len(df) >= chunk:
                df['day_int'] = df.index.to_julian_date().astype(np.int32)
                df['day_diff'] = df['day_int'].diff()
                arr = df.to_numpy(dtype=np.float32)
                is_consec = arr[1:, -1] == 1.0
                win = chunk - 1
                if len(is_consec) >= win:
                    conv = np.convolve(is_consec, np.ones(win, dtype=int), mode='valid')
                    starts = np.where(conv == win)[0]
                    if starts.size > 0:
                        for i in starts:
                            end_idx = i + chunk
                            sub = df.iloc[i:end_idx, :]
                            day_int = int(sub['day_int'].iloc[-1])
                            x_np = _build_features(sub.iloc[:, :len(df.columns) - 2], bias, scale)
                            task_q.put((stn, day_int, x_np), block=True)
        finally:
            file_q.task_done()


def cache_node_contexts(lstm_model_dir, parquet_dir, scaler_json, out_dir, chunk_size,
                        num_workers: int = 1, queue_size: int = 2048, batch_size: int = 64):
    # prefer unified variable-specific scaler under training/scalers/{var}.json
    var_name = os.path.basename(parquet_dir).replace('_obs', '')
    parquet_root = os.path.dirname(parquet_dir)
    scaler, _, _ = load_variable_scaler(parquet_root, var_name)

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
    if num_workers == 1:
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
                feats = _build_features(sub.iloc[:, :len(df.columns) - 2], scaler.bias, scaler.scale)
                assert feats.shape[1] == num_bands_, "feature width must match LSTM num_bands"
                x = torch.tensor(feats, dtype=torch.float32)
                x = x.to(device)
                with torch.no_grad():
                    _ = lstm(x)
                ctx = captured['ctx']
                np.save(os.path.join(out_dir, stn, f"{day_int}.npy"), ctx.squeeze())
    else:
        ctx_mp = mp.get_context('spawn')

        file_q = ctx_mp.JoinableQueue()
        task_q = ctx_mp.Queue(maxsize=queue_size)

        for fp in files:
            file_q.put(fp)
        for _ in range(num_workers):
            file_q.put(None)

        procs = []
        for _ in range(num_workers):
            p = ctx_mp.Process(target=_producer,
                               args=(file_q, task_q, scaler.bias, scaler.scale, chunk_size))
            p.daemon = True
            p.start()
            procs.append(p)

        pbar_files = tqdm(total=len(files), desc="Files processed", unit="file")
        last_seen_done = 0
        batch_x, batch_meta = [], []

        while True:
            try:
                while len(batch_x) < batch_size:
                    stn, day_int, x_np = task_q.get(timeout=0.1)
                    batch_meta.append((stn, day_int))
                    batch_x.append(x_np)
            except queue_mod.Empty:
                pass

            done_now = 0
            try:
                # approximate progress by counting tasks done via file_q.qsize delta
                # fallback: poll finished processes
                pass
            except Exception:
                pass

            # update by joining tasks processed in queue
            # as a simple heuristic, update when file queue shrinks
            remaining = file_q.qsize() if hasattr(file_q, 'qsize') else None
            if remaining is not None:
                processed = len(files) - remaining
                if processed > last_seen_done:
                    pbar_files.update(processed - last_seen_done)
                    last_seen_done = processed

            if batch_x:
                assert all(bx.shape[1] == num_bands_ for bx in batch_x), "feature width must match LSTM num_bands"
                x = torch.tensor(np.stack(batch_x, axis=0), dtype=torch.float32, device=device)
                with torch.no_grad():
                    _ = lstm(x)
                ctx_batch = captured['ctx']
                for (stn, day_int), ctx_vec in zip(batch_meta, ctx_batch):
                    os.makedirs(os.path.join(out_dir, stn), exist_ok=True)
                    np.save(os.path.join(out_dir, stn, f"{day_int}.npy"), np.asarray(ctx_vec).squeeze())
                batch_x.clear()
                batch_meta.clear()

            if all(not p.is_alive() for p in procs) and task_q.empty():
                break

            time.sleep(0.05)

        if batch_x:
            assert all(bx.shape[1] == num_bands_ for bx in batch_x), "feature width must match LSTM num_bands"
            x = torch.tensor(np.stack(batch_x, axis=0), dtype=torch.float32, device=device)
            with torch.no_grad():
                _ = lstm(x)
            ctx_batch = captured['ctx']
            for (stn, day_int), ctx_vec in zip(batch_meta, ctx_batch):
                os.makedirs(os.path.join(out_dir, stn), exist_ok=True)
                np.save(os.path.join(out_dir, stn, f"{day_int}.npy"), np.asarray(ctx_vec).squeeze())
            batch_x.clear()
            batch_meta.clear()

        for p in procs:
            p.join()
        pbar_files.close()

    h.remove()


if __name__ == '__main__':

    training = '/data/ssd2/dads/training'

    variable_ = 'tmax'
    target_var_ = f'{variable_}_obs'

    lstm_model_dir = os.path.join(training, 'lstm', 'checkpoints', f'{variable_}_20251004_1650')
    parquet_dir = os.path.join(training, 'parquet', target_var_)
    scaler_json = os.path.join(training, 'scalers', f'{variable_}.json')
    out_dir = os.path.join(training, 'node_ctx')

    chunk_size = 12
    cache_node_contexts(lstm_model_dir, parquet_dir, scaler_json, out_dir, chunk_size, num_workers=6)
# ========================= EOF ====================================================================
