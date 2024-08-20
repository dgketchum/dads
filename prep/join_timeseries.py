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

COMPARISON_VARS = ['rsds', 'mean_temp', 'vpd', 'rn', 'u2']

NLDAS_COL_DROP = ['doy', 'year', 'date_str']


def join_daily_timeseries(stations, sta_dir, nldas_dir, dst_dir, gridmet_dir=None, overwrite=False, bounds=None,
                          shuffle=False, write_missing=None, hourly=False):
    """"""
    stations = pd.read_csv(stations, index_col='index')
    stations.index = stations['fid'].astype(str)
    stations['source'] = stations['source'].astype(str)
    stations.sort_index(inplace=True)

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

    ct, obs_dir, ndf_hr = 0, None, None
    empty, eidx = pd.DataFrame(columns=['fid', 'source', 'orig_netid']), 0

    if hourly:
        nldas_hr_dir = nldas_dir + '_hourly'

    for i, (f, row) in enumerate(stations.iterrows(), start=1):

        if 'snotel' in row['source']:
            continue
        elif row['source'].endswith('gwx'):
            obs_dir = 'gwx'
        elif row['source'] == 'madis':
            obs_dir = 'madis'
        else:
            print(f'Observation source unknown for {f}')
            continue

        if hourly:
            out = os.path.join(dst_dir, 'hourly', '{}.csv'.format(f))
        else:
            out = os.path.join(dst_dir, 'daily', '{}.csv'.format(f))

        if os.path.exists(out) and not overwrite:
            df = pd.read_csv(out)
            df.dropna(subset=['rsds_obs', 'mean_temp_obs', 'vpd_obs',
                              'rn_obs', 'u2_obs'], inplace=True, axis=0)
            if df.empty:
                empty.at[eidx, 'fid'] = f
                empty.at[eidx, 'source'] = row['source']
                eidx += 1
                print('obs file is empty: {}'.format(os.path.basename(out)))
                continue

            print('{} in {} exists, skipping'.format(os.path.basename(out), row['source']))
            continue

        nldas_file = os.path.join(nldas_dir, '{}.csv'.format(f))

        try:
            ndf = pd.read_csv(nldas_file, index_col=0, parse_dates=True)

            if hourly:
                nldas_hr_file = os.path.join(nldas_hr_dir, '{}.csv'.format(f))
                ndf_hr = pd.read_csv(nldas_hr_file, index_col=0, parse_dates=True)
                ndf_hr.index = pd.DatetimeIndex(
                    [datetime.datetime(i.year, i.month, i.day, i.hour) for i in ndf_hr.index])
                ndf_hr.drop(columns=['doy'], inplace=True)
                nld_hr_cols = ['{}_nl_hr'.format(c) for c in ndf_hr.columns]
                ndf_hr.columns = nld_hr_cols

        except FileNotFoundError:
            empty, eidx = add_empty_entry(empty, eidx, f, row, 'does not exist', nldas_file)
            print('nldas_file {} does not exist'.format(os.path.basename(nldas_file)))
            continue

        ndf.index = pd.DatetimeIndex([datetime.date(i.year, i.month, i.day) for i in ndf.index])
        ndf.drop(columns=NLDAS_COL_DROP, inplace=True)

        nld_cols = ['{}_nl'.format(c) for c in ndf.columns]
        ndf.columns = nld_cols

        gridmet_file = os.path.join(gridmet_dir, '{}.csv'.format(f))
        try:
            gdf = pd.read_csv(gridmet_file, index_col=0, parse_dates=True)
        except FileNotFoundError:
            empty, eidx = add_empty_entry(empty, eidx, f, row, 'does not exist', gridmet_file)
            print('gridmet_file {} does not exist'.format(os.path.basename(gridmet_file)))
            continue

        gdf.index = pd.DatetimeIndex([datetime.date(i.year, i.month, i.day) for i in gdf.index])
        gdf = gdf[COMPARISON_VARS]
        grd_cols = ['{}_gm'.format(c) for c in gdf.columns]
        gdf.columns = grd_cols

        sta_file = os.path.join(sta_dir, obs_dir, '{}.csv'.format(f))

        try:
            sdf = pd.read_csv(sta_file, index_col='Unnamed: 0', parse_dates=True)
        except ValueError:
            sdf = pd.read_csv(sta_file, index_col='date', parse_dates=True)
        except FileNotFoundError:
            empty, eidx = add_empty_entry(empty, eidx, f, row, 'does not exist', sta_file)
            print('sta_file {} does not exist'.format(os.path.basename(sta_file)))
            continue

        sdf.rename(columns=RENAME_MAP, inplace=True)
        sdf['doy'] = [i.dayofyear for i in sdf.index]
        sdf = sdf[COMPARISON_VARS]
        obs_cols = ['{}_obs'.format(c) for c in sdf.columns]
        sdf.columns = obs_cols

        data_cols = obs_cols + grd_cols + nld_cols
        all_cols = ['FID'] + data_cols

        if hourly:
            gdf = gdf.resample('H').ffill()
            sdf = sdf.resample('H').ffill()
            ndf = ndf.resample('H').ffill()
            ndf_hr = ndf_hr.loc[~ndf_hr.index.duplicated(keep='first')]
            data_cols = obs_cols + grd_cols + nld_cols + nld_hr_cols
            all_cols = ['FID'] + data_cols

        try:
            if hourly:
                sdf = pd.concat([sdf, gdf, ndf, ndf_hr], ignore_index=False, axis=1)
            else:
                sdf = pd.concat([sdf, gdf, ndf], ignore_index=False, axis=1)
        except pd.errors.InvalidIndexError:
            print('Non-unique index in {}'.format(f))
            continue

        sdf['FID'] = f
        sdf = sdf[all_cols].copy()
        sdf.dropna(subset=['rsds_obs', 'mean_temp_obs', 'vpd_obs',
                           'rn_obs', 'u2_obs'], inplace=True, axis=0)
        if sdf.empty:
            empty, eidx = add_empty_entry(empty, eidx, f, row, 'col all nan', sta_file)
            print('obs file has all nan in a column: {}'.format(os.path.basename(sta_file)))
            continue
        else:
            sdf = sdf[sorted(sdf.columns)]
            sdf.to_csv(out)

        print('wrote {} station {} to {}, {} records'.format(row['source'], f, os.path.basename(out), sdf.shape[0]))

    if write_missing:
        empty.to_csv(missing_list)
        print('wrote', missing_list)


def add_empty_entry(edf, idx, feat, row_, reason, file_):
    edf.at[idx, 'fid'] = feat
    edf.at[idx, 'source'] = row_['source']
    edf.at[idx, 'note'] = reason
    edf.at[idx, 'dataset'] = file_
    idx += 1
    return edf, idx


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    fields = os.path.join(d, 'met', 'stations', 'dads_stations_elev_mgrs.csv')

    obs = os.path.join(d, 'met', 'obs')
    gm = os.path.join(d, 'met', 'gridded', 'gridmet')
    joined = os.path.join(d, 'met', 'obs_grid')
    missing_list = os.path.join(d, 'met', 'obs_grid', 'missing_data.csv')

    hourly_ = True
    nl = os.path.join(d, 'met', 'gridded', 'nldas2')
    join_daily_timeseries(fields, obs, nl, joined, gm, overwrite=False, shuffle=True,
                          bounds=(-125., 25., -96., 49.), write_missing=missing_list, hourly=hourly_)

# ========================= EOF ====================================================================
