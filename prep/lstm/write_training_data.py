import os
import json
import random
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error

from prep.lstm.dt_encoder import datetime_encoded

TERRAIN_FEATURES = ['slope', 'aspect', 'elevation', 'tpi_1250', 'tpi_250', 'tpi_150']


def print_rmse(o, n, g):
    r2_nl = r2_score(o, n)
    rmse_nl = root_mean_squared_error(o, n)
    print("r2_nldas", r2_nl)
    print('rmse_nldas', rmse_nl)

    r2_gm = r2_score(o, g)
    rmse_gm = root_mean_squared_error(o, g)
    print("r2_gridmet", r2_gm)
    print('rmse_gridmet', rmse_gm)


def write_pth_training_data(csv_dir, training_metadata, output_dir, train_frac=0.8, chunk_size=72,
                            chunks_per_file=1000, target='rsds', hourly_dir=None, shuffle=False):
    metadata = {'chunk_size': chunk_size,
                'chunks_per_file': chunks_per_file,
                'column_order': [],
                'data_frequency': [],
                'observation_count': 0}

    files_ = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    if shuffle:
        stations = [f.split('.')[0] for f in files_]
        random.shuffle(stations)
    else:
        stations = sorted([f.split('.')[0] for f in files_])

    destiny = ['train' if random.random() < train_frac else 'val' for _ in stations]
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
            yr_dt_encoding = datetime_encoded(df, ['year'])
            df = pd.concat([df, yr_dt_encoding], ignore_index=False, axis=1)

            if hourly_dir:
                hr_filepath = os.path.join(hourly_dir, '{}.csv'.format(station))
                dfhr = pd.read_csv(hr_filepath, index_col=0, parse_dates=True)
                dfhr.index = [datetime(i.year, i.month, i.day, i.hour) for i in dfhr.index]

                start_date = max(df.index.min(), dfhr.index.min())
                end_date = min(df.index.max(), dfhr.index.max())
                df = df.loc[start_date:end_date]
                dfhr = dfhr.loc[start_date:end_date]

                day_dt_encoding = datetime_encoded(dfhr, ['day'])
                dfhr = pd.concat([dfhr, day_dt_encoding], ignore_index=False, axis=1)

        except FileNotFoundError:
            continue
        except TypeError:
            continue
        if df.empty:
            continue

        if fate == 'val':
            obs.extend(df[f'{target}_obs'].to_list())
            gm.extend(df[f'{target}_gm'].to_list())
            nl.extend(df[f'{target}_nl'].to_list())

        day_diff = df['dt_diff'].astype(int).to_list()
        df.drop(columns=['dt_diff', 'doy'], inplace=True)
        dfhr.drop(columns=['doy', 'hour'], inplace=True)

        if first:
            metadata['column_order'] = df.columns.to_list()
            metadata['column_order'].extend(dfhr.columns.to_list())

            [metadata['data_frequency'].append('lf') for _ in df.columns]
            [metadata['data_frequency'].append('hf') for _ in dfhr.columns]

            first = False

        data_tensor_daily = torch.tensor(df.values, dtype=torch.float32)
        data_tensor_hourly = torch.tensor(dfhr.values, dtype=torch.float32)

        matching_timestamps = df.index.intersection(dfhr.index)

        station_chunk_ct, station_chunks = 0, []
        for i in range(len(matching_timestamps) - chunk_size + 1):
            end_timestamp = matching_timestamps[i + chunk_size - 1]

            end_index_daily = df.index.get_loc(end_timestamp)
            end_index_hourly = dfhr.index.get_loc(end_timestamp)

            chunk_daily_start = max(0, end_index_daily - chunk_size + 1)
            chunk_daily = data_tensor_daily[chunk_daily_start: end_index_daily + 1]

            chunk_hourly_start = max(0, end_index_hourly - chunk_size + 1)
            chunk_hourly = data_tensor_hourly[chunk_hourly_start: end_index_hourly + 1]

            check_consecutive = np.array(day_diff[i + chunk_size - 1])
            sequence_check = np.all(check_consecutive == 1)
            if not sequence_check:
                continue

            if len(chunk_hourly) < chunk_size or len(chunk_daily) < chunk_size:
                continue

            combined_chunk = torch.cat([chunk_daily, chunk_hourly], dim=1)
            station_chunks.append(combined_chunk)
            station_chunk_ct += 1

        if len(station_chunks) > 0:
            outfile = os.path.join(output_dir, fate, '{}.pth'.format(station))
            stack = torch.stack(station_chunks)
            torch.save(stack, outfile)
            write_files += 1
            print('{}; {} of {} to {}, {} chunks, size {}'.format(station, j, len(stations), fate,
                                                                  station_chunk_ct, stack.shape))
            metadata['observation_count'] += chunk_size * len(station_chunks)

    if training_metadata:
        with open(training_metadata, 'w') as fp:
            json.dump(metadata, fp, indent=4)

    print('\n{} sites\n{}; {} observations, {} held out for validation'.format(write_files, target_var,
                                                                               metadata['observation_count'],
                                                                               len(obs)))

    if len(obs) > 0:
        print_rmse(obs, nl, gm)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    target_var = 'mean_temp'

    zoran = '/home/dgketchum/training/lstm'
    nvm = '/media/nvm/training/lstm'
    if os.path.exists(zoran):
        print('reading from zoran')
        training = zoran
    elif os.path.exists(nvm):
        print('reading from local NVM drive')
        training = nvm
    else:
        print('reading from UM drive')
        training = os.path.join(d, 'training')

    param_dir = os.path.join(training, target_var)

    out_csv = os.path.join(param_dir, 'compiled_csv')
    out_pth = os.path.join(param_dir, 'pth')

    # hourly_data = None
    hourly_data = os.path.join(d, 'met', 'gridded', 'nldas2_hourly')

    if not os.path.exists(out_pth):
        os.mkdir(out_pth)

    if not os.path.exists(os.path.join(out_pth, 'val')):
        os.mkdir(os.path.join(out_pth, 'val'))

    if not os.path.exists(os.path.join(out_pth, 'train')):
        os.mkdir(os.path.join(out_pth, 'train'))

    print('========================== writing {} traing data =========================='.format(target_var))

    # metadata_ = None
    metadata_ = os.path.join(param_dir, 'training_metadata.json')
    write_pth_training_data(out_csv, metadata_, out_pth, target=target_var, hourly_dir=hourly_data,
                            chunk_size=72, shuffle=True)
# ========================= EOF ==============================================================================
