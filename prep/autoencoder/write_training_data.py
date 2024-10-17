import shutil
import os
import json
import random

import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import r2_score, root_mean_squared_error

TERRAIN_FEATURES = ['slope', 'aspect', 'elevation', 'tpi_1250', 'tpi_250', 'tpi_150']
RESIDUAL_FEATURES = [
    'mean_temp',
    'rn',
    'rsds',
    'u2',
    'vpd',
]


def print_rmse(o, n, g):
    r2_nl = r2_score(o, n)
    rmse_nl = root_mean_squared_error(o, n)
    print("r2_nldas", r2_nl)
    print('rmse_nldas', rmse_nl)

    r2_gm = r2_score(o, g)
    rmse_gm = root_mean_squared_error(o, g)
    print("r2_gridmet", r2_gm)
    print('rmse_gridmet', rmse_gm)


def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe_str = [f'pe_{i}' for i in range(0, pe.shape[1])]

    return pe, pe_str


def write_pth_training_data(stations, csv_dir, training_metadata, output_dir, chunk_size=72,
                            chunks_per_file=1000, d_model=4, shuffle=False, include_mask=False):
    metadata = {'chunk_size': chunk_size,
                'chunks_per_file': chunks_per_file,
                'column_order': [],
                'data_columns': [],
                'obs_columns': [],
                'reanalysis_columns': [],
                'encoding_columns': [],
                'data_frequency': [],
                'observation_count': 0}

    gdf = gpd.read_file(stations)
    gdf.index = gdf['fid']
    train_split = gdf[['train']]

    files_ = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    names = [f.split('.')[0] for f in files_]
    if shuffle:
        random.shuffle(names)
    else:
        names = sorted(names)

    destiny, stations = [], []
    for s in names:
        try:
            train_status = train_split.loc[s, 'train']
        except KeyError:
            continue

        if train_status:
            destiny.append('train')
            stations.append(s)
        else:
            destiny.append('val')
            stations.append(s)

    obs, gm, nl = [], [], []

    first, write_files = True, 0
    for j, (fate, station) in enumerate(zip(destiny, stations), start=1):

        v_file = os.path.join(output_dir, 'train', '{}.pth'.format(station))
        t_file = os.path.join(output_dir, 'val', '{}.pth'.format(station))

        if any([os.path.exists(t_file), os.path.exists(v_file)]):
            print('{} exists'.format(station))
            continue

        filepath = os.path.join(csv_dir, '{}.csv'.format(station))
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)

            if include_mask:
                years = sorted(list(set([i.year for i in df.index])))
                dt_range = pd.date_range(f'{years[0]}-01-01', f'{years[-1]}-12-31', freq='D')
                dt_index = pd.DatetimeIndex(dt_range)
                df = df.reindex(dt_index)

            df['dt_diff'] = df.index.to_series().diff().dt.days.fillna(1)

        except FileNotFoundError:
            continue
        except TypeError:
            continue
        if df.empty:
            continue

        day_diff = df['dt_diff'].astype(int).to_list()
        df.drop(columns=['dt_diff'], inplace=True)

        metadata['obs_columns'] = [f'{p}_obs' for p in RESIDUAL_FEATURES]
        metadata['reanalysis_columns'] = [f'{p}_nl' for p in RESIDUAL_FEATURES]

        df = df[metadata['obs_columns'] + metadata['reanalysis_columns']]

        [metadata['data_frequency'].append('lf') for _ in df.columns]

        # first valid reanalysis
        idx_reanalysis = find_first_valid_row(df.values, 5, 9)
        # first valid obs
        idx_obs = find_first_valid_row(df.values, 0, 5)
        try:
            idx = max(idx_obs, idx_reanalysis)
        except TypeError:
            continue

        yr = df.index[idx].year
        df = df.loc[f'{yr}-01-01':].copy()

        data_tensor_daily = torch.tensor(df.values, dtype=torch.float32)
        data_tensor_daily = data_tensor_daily[idx:]

        if data_tensor_daily.shape[0] < chunk_size:
            continue

        station_chunk_ct, station_chunks = 0, []
        iters = int(np.floor(data_tensor_daily.shape[0] / chunk_size))
        for i in range(1, iters + 1):

            end_timestamp = df.index[i * chunk_size - 1]

            end_index_daily = df.index.get_loc(end_timestamp)

            chunk_daily_start = max(0, end_index_daily - chunk_size + 1)
            chunk_daily = data_tensor_daily[chunk_daily_start: end_index_daily + 1]

            pe, pe_strings = positional_encoding(chunk_daily.size(0), d_model)
            chunk_daily = torch.cat([chunk_daily, pe], dim=1)

            try:
                check_consecutive = np.array(day_diff[chunk_daily_start: end_index_daily + 1])
            except IndexError:
                continue

            sequence_check = np.all(check_consecutive == 1)
            if not sequence_check:
                continue

            arr = chunk_daily.numpy()[:, :len(RESIDUAL_FEATURES)]
            nanct = np.count_nonzero(np.isnan(arr))
            nan_frac = nanct / (arr.size + 1e-6)

            # if nan_frac > 0.0 and fate == 'val':
            #     continue

            # if nan_frac > 0.5:
            #     continue

            # TODO: piece together data if the station is mostly nan
            station_chunks.append(chunk_daily)
            station_chunk_ct += 1

        if len(station_chunks) > 0:
            outfile = os.path.join(output_dir, fate, '{}.pth'.format(station))
            stack = torch.stack(station_chunks)
            torch.save(stack, outfile)
            write_files += 1
            print('{}; {} of {} to {}, {} chunks, size {}'.format(station, j, len(stations), fate,
                                                                  station_chunk_ct, stack.shape))
            metadata['observation_count'] += chunk_size * len(station_chunks)

            if training_metadata and first:
                metadata['column_order'] = df.columns.to_list() + pe_strings
                metadata['data_columns'] = df.columns.to_list()
                metadata['encoding_columns'] = pe_strings

                with open(training_metadata, 'w') as fp:
                    json.dump(metadata, fp, indent=4)
                first = False

    print('\n{} sites, {} train, {} val\n{} observations'.format(write_files, destiny.count('train'),
                                                                 destiny.count('val'),
                                                                 metadata['observation_count']))

    if len(obs) > 0:
        print_rmse(obs, nl, gm)


def find_first_valid_row(tensor, i, j):
    for row_idx in range(tensor.shape[0]):
        if np.all(np.isfinite(tensor[row_idx, i:j + 1])):
            return row_idx
    return None


def organize_existing_samples(stations, csv_dir, output_dir):

    gdf = gpd.read_file(stations)
    gdf.index = gdf['fid']
    train_split = gdf[['train']]

    files_ = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    names = [f.split('.')[0] for f in files_]

    destiny, stations = [], []
    for s in names:
        try:
            train_status = train_split.loc[s, 'train']
        except KeyError:
            continue

        if train_status:
            destiny.append('train')
            stations.append(s)
        else:
            destiny.append('val')
            stations.append(s)

    for j, (fate, station) in enumerate(zip(destiny, stations), start=1):

        v_file = os.path.join(output_dir, 'train', '{}.pth'.format(station))
        t_file = os.path.join(output_dir, 'val', '{}.pth'.format(station))

        if fate == 'train' and os.path.exists(v_file):
            shutil.move(v_file, t_file)
            print(f'moved {os.path.basename(v_file)} from val to train')

        elif fate == 'val' and os.path.exists(t_file):
            shutil.move(t_file, v_file)
            print(f'moved {os.path.basename(t_file)} from train to val')

        else:
            pass

if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    shapefile = os.path.join(d, 'met', 'stations', 'dads_stations_res_elev_mgrs_split.shp')

    zoran = '/home/dgketchum/training'
    nvm = '/media/nvm/training'
    if os.path.exists(zoran):
        print('writing to zoran')
        training = zoran
    elif os.path.exists(nvm):
        print('writing to local NVM drive')
        training = nvm
    else:
        print('writing to UM drive')
        training = os.path.join(d, 'training')

    param_dir = os.path.join(training, 'autoencoder')

    sta = os.path.join(d, 'met', 'joined', 'daily_untrunc')
    out_pth = os.path.join(param_dir, 'pth')

    if not os.path.exists(out_pth):
        os.mkdir(out_pth)

    if not os.path.exists(os.path.join(out_pth, 'val')):
        os.mkdir(os.path.join(out_pth, 'val'))

    if not os.path.exists(os.path.join(out_pth, 'train')):
        os.mkdir(os.path.join(out_pth, 'train'))

    print('========================== writing autoencoder traing data ==========================')

    # metadata_ = None
    metadata_ = os.path.join(param_dir, 'training_metadata.json')
    write_pth_training_data(shapefile, sta, metadata_, out_pth, chunk_size=365, d_model=4,
                            shuffle=True, include_mask=True)

    # organize_existing_samples(shapefile, sta, out_pth)
# ========================= EOF ==============================================================================
