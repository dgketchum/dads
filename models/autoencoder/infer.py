import os
import torch
import torch.nn as nn
import pandas as pd


def infer_and_save_embedding(model, station_data_path, output_path, chunk_size=72 * 24, stride=24 * 30):
    df = pd.read_csv(station_data_path)
    data = torch.tensor(df.values, dtype=torch.float32).unsqueeze(0)
    seq_len = data.size(1)

    embeddings = []
    for start_idx in range(0, seq_len - chunk_size + 1, stride):
        chunk = data[:, start_idx:start_idx + chunk_size, :]
        with torch.no_grad():
            _, embedding = model(chunk)
        embeddings.append(embedding)

    final_embedding = torch.mean(torch.stack(embeddings), dim=0)

    torch.save(final_embedding, output_path)
    print(f'Saved embedding for {station_data_path} to {output_path}')


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
