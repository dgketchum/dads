import datetime
import os
import json
import warnings

import numpy as np
import pandas as pd
import ast

warnings.filterwarnings("ignore", category=FutureWarning)
VAR_MAP = {'rsds': 'Rs (w/m2)',
           'ea': 'Vapor Pres (kPa)',
           'min_temp': 'TMin (C)',
           'max_temp': 'TMax (C)',
           'mean_temp': 'TAvg (C)',
           'wind': 'ws_2m (m/s)',
           'eto': 'ETo (mm)'}

RENAME_MAP = {v: k for k, v in VAR_MAP.items()}

COMPARISON_VARS = ['rsds', 'mean_temp', 'vpd', 'rn', 'u2', 'eto']

STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']


def join_daily_timeseries(stations, sta_dir, nldas_dir, dst_dir, gridmet_dir=None, index='FID',
                          metric_json=None):
    """"""
    stations = pd.read_csv(stations)
    stations['fid'] = [f.strip() for f in stations['fid']]
    stations.index = stations['fid']
    stations.sort_index(inplace=True)

    stations = stations[stations['source'] == 'madis']

    df, ct = pd.DataFrame(), 0
    for i, (f, row) in enumerate(stations.iterrows(), start=1):

        nldas_file = os.path.join(nldas_dir, '{}.csv'.format(f))
        try:
            ndf = pd.read_csv(nldas_file, index_col=0, parse_dates=True)
        except FileNotFoundError:
            print('{} does not exist'.format(nldas_file))
            continue

        ndf.index = pd.DatetimeIndex([datetime.date(i.year, i.month, i.day) for i in ndf.index])
        ndf = ndf[COMPARISON_VARS]
        nld_cols = ['{}_nl'.format(c) for c in ndf.columns]
        ndf.columns = nld_cols

        gridmet_file = os.path.join(gridmet_dir, '{}.csv'.format(f))
        try:
            gdf = pd.read_csv(gridmet_file, index_col=0, parse_dates=True)
        except FileNotFoundError:
            print('{} does not exist'.format(gridmet_file))
            continue

        gdf.index = pd.DatetimeIndex([datetime.date(i.year, i.month, i.day) for i in gdf.index])
        gdf = gdf[COMPARISON_VARS]
        grd_cols = ['{}_gm'.format(c) for c in gdf.columns]
        gdf.columns = grd_cols

        sta_file = os.path.join(sta_dir, '{}.csv'.format(f))

        try:
            sdf = pd.read_csv(sta_file, index_col='Unnamed: 0', parse_dates=True)
        except ValueError:
            sdf = pd.read_csv(sta_file, index_col='date', parse_dates=True)
        except FileNotFoundError:
            print('{} does not exist'.format(sta_file))
            continue

        sdf.rename(columns=RENAME_MAP, inplace=True)
        sdf['doy'] = [i.dayofyear for i in sdf.index]
        sdf = sdf[COMPARISON_VARS]
        obs_cols = ['{}_obs'.format(c) for c in sdf.columns]
        sdf.columns = obs_cols

        all_cols = ['FID'] + obs_cols + grd_cols + nld_cols

        try:
            sdf = pd.concat([sdf, gdf, ndf], ignore_index=False, axis=1)
        except pd.errors.InvalidIndexError:
            print('Non-unique index in {}'.format(f))
            continue
        sdf['FID'] = f
        sdf = sdf[all_cols]
        df = pd.concat([df, sdf])
        df.dropna(subset=['rsds_obs', 'mean_temp_obs', 'vpd_obs',
                          'rn_obs', 'u2_obs', 'eto_obs'], inplace=True, axis=0)
        if df.empty:
            continue
        else:
            out = os.path.join(dst_dir, '{}.csv'.format(f))
            df = df.reindex(sorted(df.columns), axis=1)
            df.to_csv(out)
            ct += 1

        if metric_json:
            if os.path.exists(metric_json):
                with open(metric_json, 'r') as fp:
                    metrics = json.load(fp)
            else:
                metrics = {}

            if f not in metrics.keys():
                rmse = get_rmse(df)

                metrics[f] = rmse
                print(f, df.shape[0], 'records')
                print(f, 'RMSE rsds gridmet: {:.2f}, nldas: {:.2f}'.format(rmse['rsds_gm'], rmse['rsds_nl']))
                print(f, 'RMSE tmean gridmet: {:.2f}, nldas: {:.2f}'.format(rmse['mean_temp_gm'],
                                                                            rmse['mean_temp_nl']))
                print(f, 'Mean Temp: {:.2f}\n'.format(df['mean_temp_obs'].mean().item()))
                continue
        else:
            print(f, ct)

    if metric_json:
        with open(metric_json, 'w') as fp:
            json.dump(metrics, fp, indent=4)


def get_rmse(df):
    variables = ['rsds', 'mean_temp', 'vpd', 'rn', 'u2', 'eto']
    rmse_results = {}

    for variable in variables:
        rmse_gm = np.sqrt(np.mean((df[f'{variable}_obs'] - df[f'{variable}_gm']) ** 2))
        rmse_results[f'{variable}_gm'] = rmse_gm.item()

        rmse_nl = np.sqrt(np.mean((df[f'{variable}_obs'] - df[f'{variable}_nl']) ** 2))
        rmse_results[f'{variable}_nl'] = rmse_nl.item()
    return rmse_results


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    fields = os.path.join(d, 'met', 'stations', 'dads_stations_WMT_mgrs.csv')

    obs = os.path.join(d, 'met', 'obs', 'madis')
    gm = os.path.join(d, 'met', 'gridded', 'gridmet')
    nl = os.path.join(d, 'met', 'gridded', 'nldas2')
    # rs = os.path.join(d, 'rs', 'gwx_stations', 'concatenated', 'bands.csv')

    rs = os.path.join(d, 'rs', 'dads_stations', 'landsat')
    joined = os.path.join(d, 'met', 'tables', 'obs_grid')
    metrics_ = os.path.join(d, 'met', 'tables', 'metrics.json')
    plots = os.path.join(d, 'plots', 'gridmet')

    join_daily_timeseries(fields, obs, nl, joined, gm, metric_json=None, index='index')

# ========================= EOF ====================================================================
