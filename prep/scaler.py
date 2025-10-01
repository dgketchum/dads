import json

import pandas as pd
import torch
from tqdm import tqdm

from models.scalers import MinMaxScaler


def fit_and_save_scaler(file_paths, feature_names, scaler_path):
    print("Fitting new scaler from all training data...")
    all_data_chunks = []
    for file_path in tqdm(file_paths, desc="Reading files for scaler"):
        try:
            df = pd.read_parquet(file_path)
            df.dropna(inplace=True)
            if not df.empty:
                all_data_chunks.append(torch.tensor(df.values, dtype=torch.float32))
        except Exception as e:
            continue

    if not all_data_chunks:
        print("No data available to fit scaler. Exiting.")
        return None

    full_dataset = torch.cat(all_data_chunks, dim=0)
    scaler = MinMaxScaler(axis=0)
    scaler.fit(full_dataset.numpy())

    bias_ = scaler.bias.flatten().tolist()
    scale_ = scaler.scale.flatten().tolist()
    dct = {'bias': bias_, 'scale': scale_, 'feature_names': feature_names}
    with open(scaler_path, 'w') as fp:
        json.dump(dct, fp, indent=4)
    print(f"Scaler saved to {scaler_path}")
    return scaler


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
