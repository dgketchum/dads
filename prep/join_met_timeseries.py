import datetime
import os
import warnings

import numpy as np
import pandas as pd
import pyarrow

from utils.station_parameters import station_par_map

warnings.filterwarnings("ignore", category=FutureWarning)
VAR_MAP = {'rsds': 'Rs (w/m2)',
           'ea': 'Vapor Pres (kPa)',
           'min_temp': 'TMin (C)',
           'max_temp': 'TMax (C)',
           'mean_temp': 'TAvg (C)',
           'wind': 'ws_2m (m/s)',
           'eto': 'ETo (mm)'}

GHCN_MAP = {'TMAX': 'max_temp', 'TMIN': 'min_temp', 'PRCP': 'prcp'}

RENAME_MAP = {v: k for k, v in VAR_MAP.items()}

# this depends on comparison of existing data
COMPARISON_VARS = ['prcp', 'tmax', 'tmin', 'vpd', 'rsds']
OBS_TARGETS = ['rsds', 'max_temp', 'min_temp', 'vpd', 'prcp', 'u2']


def join_daily_timeseries(stations, sta_dir, nldas_dir, dst_dir, source, daymet_dir=None, overwrite=False,
                          bounds=None, shuffle=False, write_missing=None, hourly=False, clip_to_obs=True):
    """"""
    kw = station_par_map(source)
    stations = pd.read_csv(stations, index_col=kw['index'])
    stations.sort_index(inplace=True)

    if shuffle:
        stations = stations.sample(frac=1)

    if bounds:
        w, s, e, n = bounds
        stations = stations[(stations[kw['lat']] < n) & (stations[kw['lat']] >= s)]
        stations = stations[(stations[kw['lon']] < e) & (stations[kw['lon']] >= w)]
    else:
        # NLDAS-2 extent
        ln = stations.shape[0]
        w, s, e, n = (-125.0, 25.0, -67.0, 53.0)
        stations = stations[(stations[kw['lat']] < n) & (stations[kw['lat']] >= s)]
        stations = stations[(stations[kw['lon']] < e) & (stations[kw['lon']] >= w)]
        print('dropped {} stations outside NLDAS-2 extent'.format(ln - stations.shape[0]))

    ct, obs_dir, ndf = 0, None, None
    empty, eidx = pd.DataFrame(columns=['fid', 'source', 'orig_netid']), 0
    station_ct = stations.shape[0]

    for i, (f, row) in enumerate(stations.iterrows(), start=1):

        if f != 'D4993':
            continue

        if source == 'ghcn':
            obs_dir = None
        elif source == 'madis':
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
            print('{} in {} exists, skipping'.format(os.path.basename(out), source))
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

        if obs_dir:
            sta_file = os.path.join(sta_dir, obs_dir, '{}.csv'.format(f))
        else:
            sta_file = os.path.join(sta_dir, '{}.csv'.format(f))

        try:
            sdf = pd.read_csv(sta_file, index_col='Unnamed: 0', parse_dates=True)
        except ValueError:
            sdf = pd.read_csv(sta_file, index_col='date', parse_dates=True)
        except FileNotFoundError:
            empty, eidx = add_empty_entry(empty, eidx, f, source, 'does not exist', sta_file)
            print('sta_file {} does not exist'.format(os.path.basename(sta_file)))
            continue

        if source == 'ghcn':
            sdf = sdf.rename(columns=GHCN_MAP)
            for col in OBS_TARGETS:

                if col not in sdf.columns:
                    sdf[col] = np.nan
                    continue

                sdf[col] /= 10.
                if 'temp' in col:
                    sdf[sdf[col] > 43.0] = np.nan
                    sdf[sdf[col] < -40.0] = np.nan

        cols = [c for c in OBS_TARGETS if c in sdf.columns]
        check = sdf[cols].values
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
            sdf.dropna(subset=['rsds_obs', 'max_temp_obs', 'min_temp_obs', 'vpd_obs', 'prcp_obs', 'u2_obs'],
                       how='all', inplace=True, axis=0)

        if sdf.empty:
            empty, eidx = add_empty_entry(empty, eidx, f, row, 'col all nan', sta_file)
            print('obs file has all nan in a column: {}'.format(os.path.basename(sta_file)))
            continue

        else:
            sdf = sdf[sorted(sdf.columns)]
            sdf.to_parquet(out)

        print('wrote {} station {} to {}, {} records'.format(source, f, os.path.basename(out), valid_obs))
        ct += valid_obs
        print(f'{ct} days of observations, {i} of {station_ct}')

    if write_missing:
        empty.to_csv(missing_list)
        print('wrote', missing_list)


def add_empty_entry(edf, idx, feat, source_, reason, file_):
    edf.at[idx, 'fid'] = feat
    edf.at[idx, 'source'] = source_
    edf.at[idx, 'note'] = reason
    edf.at[idx, 'dataset'] = file_
    idx += 1
    return edf, idx


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    # fields = os.path.join(d, 'dads', 'met', 'stations', 'dads_stations_10FEB2025.csv')
    # obs = os.path.join(d, 'dads', 'met', 'obs')
    # src_ = 'madis'

    fields = os.path.join(d, 'dads', 'met', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    obs = os.path.join(d, 'climate', 'ghcn', 'station_data')
    src_ = 'ghcn'

    dm = os.path.join(d, 'dads', 'met', 'gridded', 'processed_parquet', 'daymet')
    joined = os.path.join(d, 'dads', 'met', 'joined')
    missing_list = os.path.join(d, 'dads', 'met', 'joined', 'missing_data.csv')

    clip_to_obs = True
    hourly_ = True
    overwrite = True

    if hourly_:
        nl = os.path.join(d, 'dads', 'met', 'gridded', 'raw_parquet', 'nldas2')
    else:
        nl = os.path.join(d, 'dads', 'met', 'gridded', 'processed_parquet', 'nldas2')

    join_daily_timeseries(fields, obs, nl, joined, source=src_, daymet_dir=dm, overwrite=overwrite,
                          bounds=(-180., 25., -60., 85.), shuffle=True, write_missing=missing_list, hourly=hourly_,
                          clip_to_obs=clip_to_obs)

# ========================= EOF ====================================================================
