import os
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, root_mean_squared_error

TERRAIN_FEATURES = ['slope', 'aspect', 'elevation', 'tpi_1250', 'tpi_250', 'tpi_150']


def join_training(stations, ts_dir, rs_file, dem_dir, out_dir, scaling_json, var='rsds'):
    """"""

    stations = pd.read_csv(stations)
    stations['fid'] = [f.strip() for f in stations['fid']]
    stations.index = stations['fid']
    stations.sort_index(inplace=True)

    stations = stations[stations['source'] == 'madis']

    rs_df = pd.read_csv(rs_file)

    features = None
    if var == 'rsds':
        features = TERRAIN_FEATURES
    elif var == 'vpd':
        features = TERRAIN_FEATURES

    features = ['fid'] + features
    rs_df = rs_df[features]
    rs_df.index = rs_df['fid']
    rs_df.drop(columns=['fid'], inplace=True)

    ts, ct, scaling, first, shape = None, 0, {}, True, None

    scaling['stations'] = []

    for tile in stations['MGRS_TILE'].unique():

        sol_file = os.path.join(dem_dir, 'tile_{}.csv'.format(tile))
        sol_df = pd.read_csv(sol_file)
        tile_sites = stations[stations['MGRS_TILE'] == tile]

        for i, (f, row) in enumerate(tile_sites.iterrows(), start=1):

            sta_file = os.path.join(ts_dir, '{}.csv'.format(f))
            if not os.path.exists(sta_file):
                print(os.path.basename(sta_file), 'not found, skipping')
                continue
            ts = pd.read_csv(sta_file, index_col='Unnamed: 0', parse_dates=True)
            rs = rs_df.loc[f].values
            sol = sol_df[f].to_dict()

            # training depends on having the first three columns like so
            feats = [c for c in ts.columns if c.endswith('_nl') and var not in c]
            ts = ts[[f'{var}_obs', f'{var}_gm', f'{var}_nl'] + feats]

            ts['doy'] = ts.index.dayofyear
            ts['rsun'] = ts['doy'].map(sol) * 0.0036
            ts[rs_df.columns] = np.ones((ts.shape[0], len(rs_df.columns))) * rs

            removed_nan = False
            if np.count_nonzero(np.isnan(ts.values)) > 1:
                pre_ = ts.shape[0]
                ts.dropna(how='any', axis=0, inplace=True)
                post_ = ts.shape[0]
                removed_nan = True

            for c in ts.columns:

                for mm in ['max', 'min']:

                    p = '{}_{}'.format(c, mm)
                    v = ts[c].agg(mm, axis=0)

                    if p not in scaling.keys():
                        scaling[p] = v.item()

                    if mm == 'max' and v > scaling[p]:
                        scaling[p] = v.item()

                    if mm == 'min' and v < scaling[p]:
                        scaling[p] = v.item()

            # mark consecutive days
            ts['doy_diff'] = ts.index.to_series().diff().dt.days.fillna(1)

            if first:
                shape = ts.shape[1]
                first = False
            else:
                if ts.shape[1] != shape:
                    print('{} has {} cols, should have {}, skipped it'.format(f, ts.shape[1], shape))
                    continue

            outfile = os.path.join(out_dir, '{}.csv'.format(f))
            # write csv without dt index
            ts.to_csv(outfile, index=False)
            scaling['stations'].append(f)

            if removed_nan:
                print(f, ts.shape[0], '{} rows with NaN'.format(pre_ - post_))
            else:
                print(f, ts.shape[0])

    with open(scaling_json, 'w') as fp:
        json.dump(scaling, fp, indent=4)


def print_rsmse(o, n, g):
    r2_nl = r2_score(o, n)
    rmse_nl = root_mean_squared_error(o, n)
    print("r2_nldas", r2_nl)
    print('rmse_nldas', rmse_nl)

    r2_gm = r2_score(o, g)
    rmse_gm = root_mean_squared_error(o, g)
    print("r2_gridmet", r2_gm)
    print('rmse_gridmet', rmse_gm)

    print('{} records'.format(len(o)))


def get_sr_features(df):
    feats = []
    feats += [c for c in df.columns if c.startswith('B')]
    feats += [c for c in df.columns if c.startswith('evi')]
    feats += [c for c in df.columns if c.startswith('gi')]
    feats += [c for c in df.columns if c.startswith('nd')]
    feats += [c for c in df.columns if c.startswith('nw')]
    return feats


def apply_scaling_and_save(csv_dir, scaling_json, training_metadata, output_dir, train_frac=0.8,
                           chunk_size=10, chunks_per_file=1, target='rsds'):
    with open(scaling_json, 'r') as f:
        scaling_data = json.load(f)

    scaling_data['column_order'] = []
    scaling_data['scaling_status'] = []
    scaling_data['observation_count'] = 0

    padded_data = []
    file_count = 0
    first = True
    fate = None

    stations = scaling_data['stations']
    scaling_data['train_stations'] = []
    scaling_data['val_stations'] = []
    scaling_data['chunk_size'] = chunk_size
    scaling_data['chunks_per_file'] = chunks_per_file

    if not os.path.exists(os.path.join(output_dir, 'val')):
        os.mkdir(os.path.join(output_dir, 'val'))

    if not os.path.exists(os.path.join(output_dir, 'train')):
        os.mkdir(os.path.join(output_dir, 'train'))

    destiny = ['train' if random.random() < train_frac else 'val' for _ in stations]

    obs, gm, nl = [], [], []

    for fate, station in zip(destiny, stations):

        print(station)

        scaling_data['{}_stations'.format(fate)].append(station)

        filepath = os.path.join(csv_dir, '{}.csv'.format(station))
        df = pd.read_csv(filepath)

        if df.empty:
            print('{} is empty'.format(station))
            continue

        if fate == 'val':
            obs.extend(df[f'{target}_obs'].to_list())
            gm.extend(df[f'{target}_gm'].to_list())
            nl.extend(df[f'{target}_nl'].to_list())

        day_diff = df['doy_diff'].astype(int).to_list()
        df.drop(columns=['doy_diff'], inplace=True)

        for col in df.columns:

            if col in [f'{target}_obs', f'{target}_gm', 'doy_diff']:
                if first:
                    scaling_data['column_order'].append(col)
                    scaling_data['scaling_status'].append('unscaled')
                continue
            else:
                min_val = scaling_data[f'{col}_min']
                max_val = scaling_data[f'{col}_max']
                df[col] = (df[col] - min_val) / (max_val - min_val)
                if first:
                    scaling_data['column_order'].append(col)
                    scaling_data['scaling_status'].append('scaled')

        first = False

        data_tensor = torch.tensor(df.values, dtype=torch.float32)
        num_chunks = len(data_tensor) // chunk_size

        for i in range(num_chunks):

            chunk = data_tensor[i * chunk_size: (i + 1) * chunk_size]

            consecutive = np.array(day_diff[i * chunk_size: (i + 1) * chunk_size])
            sequence_check = np.all(consecutive == 1)
            if not sequence_check:
                continue

            if len(chunk) < chunk_size:
                padding = torch.full((chunk_size - len(chunk), chunk.shape[1]), fill_value=float('nan'))
                chunk = torch.cat([chunk, padding], dim=0)
            padded_data.append(chunk)

            if len(padded_data) == chunks_per_file:
                outfile = os.path.join(output_dir, fate, f'data_chunk_{file_count}.pth')
                torch.save(torch.stack(padded_data), outfile)
                scaling_data['observation_count'] += chunk_size * len(padded_data)
                file_count += 1
                # print(os.path.basename(outfile))
                padded_data = []

    if padded_data:
        outfile = os.path.join(output_dir, fate, f'data_chunk_{file_count}.pth')
        torch.save(torch.stack(padded_data), outfile)
        scaling_data['observation_count'] += chunk_size * len(padded_data)

    with open(training_metadata, 'w') as fp:
        json.dump(scaling_data, fp, indent=4)

    print('{} sites; {} observations'.format(len(scaling_data['stations']), scaling_data['observation_count']))
    print_rsmse(obs, nl, gm)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    target_var = 'vpd'

    fields = os.path.join(d, 'met', 'stations', 'dads_stations_WMT_mgrs.csv')
    sta = os.path.join(d, 'met', 'tables', 'obs_grid')
    rs = os.path.join(d, 'rs', 'dads_stations', 'landsat', 'dads_stations_WMT_500_2023.csv')
    solrad = os.path.join(d, 'dem', 'rsun_tables')
    out_csv = os.path.join(d, 'training', target_var, 'compiled_csv')
    if not os.path.exists(out_csv):
        os.mkdir(out_csv)
    scaling_ = os.path.join(d, 'training', target_var, 'scaling_metadata.json')

    join_training(fields, sta, rs, solrad, out_csv, scaling_json=scaling_, var=target_var)

    metadata = os.path.join(d, 'training', target_var, 'training_metadata.json')
    out_pth = os.path.join(d, 'training', target_var, 'scaled_pth')

    if not os.path.exists(out_pth):
        os.mkdir(out_pth)

    apply_scaling_and_save(out_csv, scaling_, metadata, out_pth, target=target_var)
# ========================= EOF ==============================================================================
