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

NLDAS_COL_DROP = ['doy', 'year', 'date_str']


def join_daily_timeseries(stations, sta_dir, nldas_dir, dst_dir, gridmet_dir=None, overwrite=False, bounds=None,
                          shuffle=False):
    """"""
    stations = pd.read_csv(stations)
    stations.index = stations['fid']
    stations.sort_index(inplace=True)

    stations = stations[stations['source'] == 'madis']

    if shuffle:
        stations = stations.sample(frac=1)

    if bounds:
        w, s, e, n = bounds
        stations = stations[(stations['latitude'] < n) & (stations['latitude'] >= s)]
        stations = stations[(stations['longitude'] < e) & (stations['longitude'] >= w)]
    else:
        # NLDAS-2 extent
        ln = stations.shape[0]
        w, s, e, n = (-125.0, 25.0, -67.0, 53.0)
        stations = stations[(stations['latitude'] < n) & (stations['latitude'] >= s)]
        stations = stations[(stations['longitude'] < e) & (stations['longitude'] >= w)]
        print('dropped {} stations outside NLDAS-2 extent'.format(ln - stations.shape[0]))

    ct, empty_sdf = 0, []
    for i, (f, row) in enumerate(stations.iterrows(), start=1):

        out = os.path.join(dst_dir, '{}.csv'.format(f))
        if os.path.exists(out) and not overwrite:
            df = pd.read_csv(out)
            df.dropna(subset=['rsds_obs', 'mean_temp_obs', 'vpd_obs',
                              'rn_obs', 'u2_obs', 'eto_obs'], inplace=True, axis=0)
            if df.empty:
                empty_sdf.append(out)
                print('obs file has all nan in a column: {}'.format(os.path.basename(out)))
            print(os.path.basename(out), 'exists, skipping')
            continue

        nldas_file = os.path.join(nldas_dir, '{}.csv'.format(f))
        try:
            ndf = pd.read_csv(nldas_file, index_col=0, parse_dates=True)
        except FileNotFoundError:
            print('nldas_file {} does not exist'.format(os.path.basename(nldas_file)))
            continue

        ndf.index = pd.DatetimeIndex([datetime.date(i.year, i.month, i.day) for i in ndf.index])
        # ndf = ndf[COMPARISON_VARS]
        ndf.drop(columns=NLDAS_COL_DROP, inplace=True)
        nld_cols = ['{}_nl'.format(c) for c in ndf.columns]
        ndf.columns = nld_cols

        gridmet_file = os.path.join(gridmet_dir, '{}.csv'.format(f))
        try:
            gdf = pd.read_csv(gridmet_file, index_col=0, parse_dates=True)
        except FileNotFoundError:
            print('gridmet_file {} does not exist'.format(os.path.basename(gridmet_file)))
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
            print('sta_file {} does not exist'.format(os.path.basename(sta_file)))
            continue

        sdf.rename(columns=RENAME_MAP, inplace=True)
        sdf['doy'] = [i.dayofyear for i in sdf.index]
        sdf = sdf[COMPARISON_VARS]
        obs_cols = ['{}_obs'.format(c) for c in sdf.columns]
        sdf.columns = obs_cols

        data_cols = obs_cols + grd_cols + nld_cols
        all_cols = ['FID'] + data_cols

        try:
            sdf = pd.concat([sdf, gdf, ndf], ignore_index=False, axis=1)
        except pd.errors.InvalidIndexError:
            print('Non-unique index in {}'.format(f))
            continue

        sdf['FID'] = f
        sdf = sdf[all_cols]
        sdf.dropna(subset=['rsds_obs', 'mean_temp_obs', 'vpd_obs',
                           'rn_obs', 'u2_obs', 'eto_obs'], inplace=True, axis=0)
        if sdf.empty:
            empty_sdf.append(sta_file)
            print('obs file has all nan in a column: {}'.format(os.path.basename(sta_file)))
            continue
        else:
            sdf = sdf.reindex(sorted(sdf.columns), axis=1)
            sdf.to_csv(out)
            ct += 1

        print('wrote {} to {}, {} records'.format(f, os.path.basename(out), ct))
    print(empty_sdf)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    fields = os.path.join(d, 'met', 'stations', 'dads_stations_elev_mgrs.csv')

    obs = os.path.join(d, 'met', 'obs', 'madis')
    gm = os.path.join(d, 'met', 'gridded', 'gridmet')
    nl = os.path.join(d, 'met', 'gridded', 'nldas2')
    joined = os.path.join(d, 'met', 'tables', 'obs_grid')

    join_daily_timeseries(fields, obs, nl, joined, gm, overwrite=False, shuffle=True, bounds=(-125., 40., -103., 49.))

# ========================= EOF ====================================================================
