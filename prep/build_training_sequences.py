import os
import json

import numpy as np
import pandas as pd
import torch

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
        features = TERRAIN_FEATURES + get_sr_features(rs_df)

    features = ['fid'] + features
    rs_df = rs_df[features]
    rs_df.index = rs_df['fid']
    rs_df.drop(columns=['fid'], inplace=True)

    ts, ct, scaling, first, shape = None, 0, {}, True, None

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
            ts = ts[[f'{var}_obs', f'{var}_gm'] + [c for c in ts.columns if c.endswith('_nl')]]
            ts['doy'] = ts.index.dayofyear
            ts['rsun'] = ts['doy'].map(sol) * 0.0036
            ts[rs_df.columns] = np.ones((ts.shape[0], len(rs_df.columns))) * rs

            removed_nan = False
            if np.count_nonzero(np.isnan(ts.values)) > 1:
                pre_ = ts.shape[0]
                ts.dropna(how='any', axis=0, inplace=True)
                post_ = ts.shape[0]
                removed_nan = True

            if scaling_json:
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

            outfile = os.path.join(out_dir, '{}.csv'.format(f))

            if first:
                shape = ts.shape[1]
                first = False
            else:
                if ts.shape[1] != shape:
                    print('{} has {} cols, should have {}, skipped it'.format(f, ts.shape[1], shape))
                    continue

            # write csv without dt index
            ts.to_csv(outfile, index=False)
            if removed_nan:
                print(f, ts.shape[0], '{} rows with NaN'.format(pre_ - post_))
            else:
                print(f, ts.shape[0])

    if scaling_json:
        with open(scaling_json, 'w') as fp:
            json.dump(scaling, fp, indent=4)


def get_sr_features(df):
    feats = [c for c in df.columns if c.startswith('B')]
    feats += [c for c in df.columns if c.startswith('evi')]
    feats += [c for c in df.columns if c.startswith('gi')]
    feats += [c for c in df.columns if c.startswith('nd')]
    feats += [c for c in df.columns if c.startswith('nw')]
    return feats


def apply_scaling_and_save(csv_dir, scaling_json, training_metadata, output_dir,
                           target='rsds_obs', comparison='rsds_gm'):
    with open(scaling_json, 'r') as f:
        scaling_data = json.load(f)

    scaling_data['column_order'] = []
    scaling_data['scaling_status'] = []
    scaling_data['observation_count'] = 0
    scaling_data['stations'] = []

    first = True
    for filename in os.listdir(csv_dir):
        if filename.endswith(".csv"):

            scaling_data['stations'].append(filename.split('.csv')[0])

            filepath = os.path.join(csv_dir, filename)
            df = pd.read_csv(filepath)

            if df.empty:
                print('{} is empty'.format(filename))
                continue

            for col in df.columns:

                if col in [target, comparison]:
                    if first:
                        scaling_data['column_order'].append(col)
                        scaling_data['scaling_status'].append('unscaled')
                    continue
                else:
                    min_val = scaling_data[f"{col}_min"]
                    max_val = scaling_data[f"{col}_max"]
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                    if first:
                        scaling_data['column_order'].append(col)
                        scaling_data['scaling_status'].append('scaled')

            first = False

            scaling_data['observation_count'] += df.shape[0]
            data_tensor = torch.tensor(df.values, dtype=torch.float32)
            outfile = os.path.join(output_dir, os.path.splitext(filename)[0] + '.pth')
            torch.save(data_tensor, outfile)
            print(os.path.basename(outfile))

    with open(training_metadata, 'w') as fp:
        json.dump(scaling_data, fp, indent=4)

    print('{} sites; {} observations'.format(len(scaling_data['stations']), scaling_data['observation_count']))


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
    scaling_ = os.path.join(d, 'training', target_var, 'scaling.json')

    join_training(fields, sta, rs, solrad, out_csv, scaling_json=scaling_, var=target_var)

    metadata = os.path.join(d, 'training', target_var, 'training_metadata.json')
    out_pth = os.path.join(d, 'training', target_var, 'scaled_pth')

    target_col, comparison_col = f'{target_var}_obs', f'{target_var}_gm'
    apply_scaling_and_save(out_csv, scaling_, metadata, out_pth, target=target_col, comparison=comparison_col)
# ========================= EOF ==============================================================================
