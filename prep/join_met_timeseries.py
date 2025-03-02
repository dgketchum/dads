import datetime
import os
import warnings

import numpy as np
import pandas as pd
import pyarrow

warnings.filterwarnings("ignore", category=FutureWarning)
VAR_MAP = {'rsds': 'Rs (w/m2)',
           'ea': 'Vapor Pres (kPa)',
           'min_temp': 'TMin (C)',
           'max_temp': 'TMax (C)',
           'mean_temp': 'TAvg (C)',
           'wind': 'ws_2m (m/s)',
           'eto': 'ETo (mm)'}

RENAME_MAP = {v: k for k, v in VAR_MAP.items()}

# this depends on comparison of existing data
COMPARISON_VARS = ['rsds', 'mean_temp', 'vpd']


def join_daily_timeseries(stations, sta_dir, nldas_dir, dst_dir, daymet_dir=None, overwrite=False, bounds=None,
                          shuffle=False, write_missing=None, hourly=False, clip_to_obs=True):
    """"""
    stations = pd.read_csv(stations, index_col=0)
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

    ct, obs_dir, ndf = 0, None, None
    empty, eidx = pd.DataFrame(columns=['fid', 'source', 'orig_netid']), 0
    station_ct = stations.shape[0]

    for i, (f, row) in enumerate(stations.iterrows(), start=1):

        if 'snotel' in row['source']:
            continue
        elif row['source'].endswith('gwx'):
            obs_dir = 'gwx'
        elif 'madis' in row['source']:
            obs_dir = 'madis_daily'
        else:
            print(f'Observation source unknown for {f}')
            continue

        if hourly:
            out = os.path.join(dst_dir, 'hourly', '{}.parquet'.format(f))
        else:
            if clip_to_obs:
                out = os.path.join(dst_dir, 'daily', '{}.parquet'.format(f))
            else:
                out = os.path.join(dst_dir, 'daily_untrunc', '{}.parquet'.format(f))

        if os.path.exists(out) and not overwrite:
            print('{} in {} exists, skipping'.format(os.path.basename(out), row['source']))
            continue

        nldas_file = os.path.join(nldas_dir, '{}.parquet'.format(f))

        try:
            if hourly:
                nldas_hr_file = os.path.join(nldas_dir, '{}.parquet.gzip'.format(f))
                ndf = pd.read_parquet(nldas_hr_file)
                nld_cols = ['{}_nl_hr'.format(c) for c in ndf.columns]
                ndf.columns = nld_cols
            else:
                ndf = pd.read_parquet(nldas_file)
                nld_cols = ['{}_nl'.format(c) for c in ndf.columns]
                ndf.columns = nld_cols

        except FileNotFoundError:
            empty, eidx = add_empty_entry(empty, eidx, f, row, 'does not exist', nldas_file)
            print('nldas_file {} does not exist'.format(os.path.basename(nldas_file)))
            continue

        except pyarrow.lib.ArrowInvalid:
            empty, eidx = add_empty_entry(empty, eidx, f, row, 'is empty', nldas_file)
            print('nldas_file {} is empty'.format(os.path.basename(nldas_file)))
            continue

        daymet_file = os.path.join(daymet_dir, '{}.parquet'.format(f))
        try:
            dmet = pd.read_parquet(daymet_file)
            dmet = dmet.rename(columns={'tmean': 'mean_temp'})
            dmet = dmet[COMPARISON_VARS]
            grd_cols = ['{}_dm'.format(c) for c in dmet.columns]
            dmet.columns = grd_cols

        except (FileNotFoundError, pyarrow.lib.ArrowInvalid):
            print('daymet_file {} does not exist'.format(os.path.basename(daymet_file)))
            dmet = pd.DataFrame(index=ndf.index, columns=ndf.columns, data=np.ones_like(ndf.values) * np.nan)
            grd_cols = [c.replace('_nl', '_dm') for c in dmet.columns]
            dmet.columns = grd_cols

        sta_file = os.path.join(sta_dir, obs_dir, '{}.csv'.format(f))

        try:
            sdf = pd.read_csv(sta_file, index_col='Unnamed: 0', parse_dates=True)
        except ValueError:
            sdf = pd.read_csv(sta_file, index_col='date', parse_dates=True)
        except FileNotFoundError:
            empty, eidx = add_empty_entry(empty, eidx, f, row, 'does not exist', sta_file)
            print('sta_file {} does not exist'.format(os.path.basename(sta_file)))
            continue

        check = sdf[COMPARISON_VARS].values
        check[check == 0.0] = np.nan
        if np.count_nonzero(np.isnan(check)) / check.size == 1.0:
            print(f'{f} all zero or nan, skipping')
            continue

        valid_obs = sdf.shape[0]
        obs_cols = ['{}_obs'.format(c) for c in sdf.columns]
        sdf.columns = obs_cols

        data_cols = obs_cols + grd_cols + nld_cols
        all_cols = ['FID'] + data_cols

        if hourly:
            dmet = dmet.resample('h').ffill()
            sdf = sdf.resample('h').ffill()
            data_cols = obs_cols + grd_cols + nld_cols
            all_cols = ['FID'] + data_cols

        try:
            sdf = pd.concat([sdf, dmet, ndf], ignore_index=False, axis=1)
        except pd.errors.InvalidIndexError:
            print('Non-unique index in {}'.format(f))
            continue

        sdf['FID'] = f
        sdf = sdf[all_cols].copy()

        if clip_to_obs:
            sdf.dropna(subset=['rsds_obs', 'mean_temp_obs', 'vpd_obs', 'prcp_obs',
                               'rn_obs', 'u2_obs'], how='all', inplace=True, axis=0)

        if sdf.empty:
            empty, eidx = add_empty_entry(empty, eidx, f, row, 'col all nan', sta_file)
            print('obs file has all nan in a column: {}'.format(os.path.basename(sta_file)))
            continue

        else:
            sdf = sdf[sorted(sdf.columns)]
            sdf.to_parquet(out)

        print('wrote {} station {} to {}, {} records'.format(row['source'], f, os.path.basename(out), valid_obs))
        ct += valid_obs
        print(f'{ct} days of observations, {i} of {station_ct}')

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

    fields = os.path.join(d, 'met', 'stations', 'dads_stations_10FEB2025.csv')
    # fields = os.path.join(d, 'met', 'stations', 'madis_mgrs_28OCT2024.csv')

    obs = os.path.join(d, 'met', 'obs')
    dm = os.path.join(d, 'met', 'gridded', 'processed_parquet', 'daymet')
    joined = os.path.join(d, 'met', 'joined')
    missing_list = os.path.join(d, 'met', 'joined', 'missing_data.csv')

    clip_to_obs = True
    hourly_ = True
    overwrite = False

    if hourly_:
        nl = os.path.join(d, 'met', 'gridded', 'raw_parquet', 'nldas2')
    else:
        nl = os.path.join(d, 'met', 'gridded', 'processed_parquet', 'nldas2')

    join_daily_timeseries(fields, obs, nl, joined, dm, overwrite=overwrite, shuffle=True, clip_to_obs=clip_to_obs,
                          bounds=(-180., 25., -60., 85.), write_missing=missing_list, hourly=hourly_)

# ========================= EOF ====================================================================
