import os
import json

import numpy as np
import pandas as pd
import torch


def join_training(stations, ts_dir, rs_file, dem_dir, out_dir, scaling_json, var='rsds'):
    """"""

    stations = pd.read_csv(stations)
    stations['fid'] = [f.strip() for f in stations['fid']]
    stations.index = stations['fid']
    stations.sort_index(inplace=True)

    stations = stations[stations['source'] == 'madis']

    rs_df = pd.read_csv(rs_file)
    rs_df = rs_df[['fid', 'slope', 'aspect', 'elevation', 'tpi_1250', 'tpi_250', 'tpi_150']]
    rs_df.index = rs_df['fid']
    rs_df.drop(columns=['fid'], inplace=True)

    ts, ct, scaling = None, 0, {}

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
            ts = ts[[f'{var}_obs', f'{var}_nl']]
            ts['doy'] = ts.index.dayofyear
            ts['rsun'] = ts['doy'].map(sol)
            ts[rs_df.columns] = np.ones((ts.shape[0], len(rs_df.columns))) * rs

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
            ts.to_csv(outfile)
            print(f, ts.shape[0])

    if scaling_json:
        with open(scaling_json, 'w') as fp:
            json.dump(scaling, fp, indent=4)


def apply_scaling_and_save(csv_dir, scaling_json, output_dir):
    with open(scaling_json, 'r') as f:
        scaling_data = json.load(f)

    for filename in os.listdir(csv_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(csv_dir, filename)
            df = pd.read_csv(filepath, index_col='Unnamed: 0', parse_dates=True)

            for col in df.columns:
                min_val = scaling_data[f"{col}_min"]
                max_val = scaling_data[f"{col}_max"]
                df[col] = (df[col] - min_val) / (max_val - min_val)

            data_tensor = torch.tensor(df.values, dtype=torch.float32)
            outfile = os.path.join(output_dir, os.path.splitext(filename)[0] + '.pth')
            torch.save(data_tensor, outfile)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    fields = os.path.join(d, 'met', 'stations', 'dads_stations_WMT_mgrs.csv')
    sta = os.path.join(d, 'met', 'tables', 'obs_grid')
    rs = os.path.join(d, 'rs', 'dads_stations', 'landsat', 'dads_stations_WMT_500_2023.csv')
    solrad = os.path.join(d, 'dem', 'rsun_tables')
    out_csv = os.path.join(d, 'training', 'compiled_csv')
    scaling_ = os.path.join(d, 'training', 'scaling.json')
    # join_training(fields, sta, rs, solrad, out_csv, scaling_json=scaling_, var='rsds')

    out_pth = os.path.join(d, 'training', 'scaled_pth')
    apply_scaling_and_save(out_csv, scaling_, out_pth)
# ========================= EOF ====================================================================
