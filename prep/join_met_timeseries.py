import datetime
import os
import warnings

import numpy as np
import pandas as pd
import pyarrow

from utils.station_parameters import station_par_map

warnings.filterwarnings("ignore", category=FutureWarning)

GHCN_MAP = {'TMAX': 'tmax', 'TMIN': 'tmin', 'PRCP': 'prcp'}
OBS_TARGETS = ['rsds', 'tmax', 'tmin', 'ea', 'prcp', 'wind']


def join_daily_timeseries(stations, sta_dir, gridded_met_dir, dst_dir, source,
                          overwrite=False, bounds=None, shuffle=False, write_missing=None,
                          hourly=False, clip_to_obs=True):
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

    ct, ndf = 0, None
    empty, eidx = pd.DataFrame(columns=['fid', 'source', 'orig_netid']), 0
    station_ct = stations.shape[0]

    for i, (f, row) in enumerate(stations.iterrows(), start=1):

        out = os.path.join(dst_dir, f'{f}.parquet')

        if os.path.exists(out) and not overwrite:
            print('{} in {} exists, skipping'.format(os.path.basename(out), source))
            continue

        gridded_met_file = os.path.join(gridded_met_dir, '{}.parquet'.format(f))

        try:
            if hourly:
                gridded_met_file = os.path.join(hourly_, '{}.parquet.gzip'.format(f))
                ndf = pd.read_parquet(gridded_met_file)
                nld_cols = ['{}_e5l_hr'.format(c) for c in ndf.columns]
                ndf.columns = nld_cols
            else:
                ndf = pd.read_parquet(gridded_met_file)
                nld_cols = ['{}_e5l'.format(c) for c in ndf.columns]
                ndf.columns = nld_cols

        except FileNotFoundError:
            empty, eidx = add_empty_entry(empty, eidx, f, row, 'does not exist', gridded_met_file)
            print('gridded_file {} does not exist'.format(os.path.basename(gridded_met_file)))
            continue

        except pyarrow.lib.ArrowInvalid:
            empty, eidx = add_empty_entry(empty, eidx, f, row, 'is empty', gridded_met_file)
            print('gridded_file {} is empty'.format(os.path.basename(gridded_met_file)))
            continue

        sta_file = os.path.join(sta_dir, '{}.parquet'.format(f))
        if not os.path.isfile(sta_file):
            sta_file = os.path.join(sta_dir, '{}.csv'.format(f))

        try:
            if sta_file.endswith('csv'):
                sdf = pd.read_csv(sta_file, index_col=0, parse_dates=True)
            else:
                sdf = pd.read_parquet(sta_file)

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

        if not clip_to_obs:
            cols = [c for c in OBS_TARGETS if c in sdf.columns]
            check = sdf[cols].values
            check[check == 0.0] = np.nan
            if np.count_nonzero(np.isnan(check)) / check.size == 1.0:
                print(f'{f} all zero or nan, skipping')
                continue

        valid_obs = sdf.shape[0]
        target_cols = [c for c in sdf.columns if c in OBS_TARGETS]
        sdf = sdf[target_cols]

        obs_cols = ['{}_obs'.format(c) for c in sdf.columns]
        sdf.columns = obs_cols

        # drop any observation columns where the entire column is 0 and/or NaN
        # noinspection PyUnresolvedReferences
        valid_obs_cols = sdf.columns[~((sdf[obs_cols] == 0) | sdf[obs_cols].isna()).all(axis=0)].to_list()

        if hourly:
            try:
                sdf = sdf.resample('h').ffill()
            except ValueError as exc:
                if 'duplicate labels' in exc.args[0]:
                    print(f'\n{sta_file} has duplicates: {exc}, removing\n')
                    os.remove(sta_file)
                    continue
                else:
                    print(f'\n{sta_file} error: {exc}\n')
                    continue

        data_cols = valid_obs_cols +  nld_cols
        all_cols = ['FID'] + data_cols

        try:
            sdf = pd.concat([sdf, ndf], ignore_index=False, axis=1)
        except pd.errors.InvalidIndexError:
            print('Non-unique index in {}'.format(f))
            continue

        sdf['FID'] = f
        sdf = sdf[all_cols].copy()

        if clip_to_obs:
            sdf.dropna(subset=valid_obs_cols,  how='all', inplace=True, axis=0)

        if sdf.empty:
            empty, eidx = add_empty_entry(empty, eidx, f, row, 'col all nan', sta_file)
            print('obs file has all nan in a column: {}'.format(os.path.basename(sta_file)))
            continue

        else:
            sdf.to_parquet(out)

        print('wrote {} station {} to {}, {} records'.format(source, f, os.path.basename(out), valid_obs))
        ct += valid_obs
        print(f'{ct} days of observations, {i} of {station_ct}')

    if write_missing:
        if len(empty) > 0:
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

    data = '/data/ssd2'
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    sites = os.path.join(root, 'dads', 'met', 'stations', 'madis_02JULY2025_mgrs.csv')
    obs = os.path.join(data, 'madis', 'daily')
    src_ = 'madis'

    # sites = os.path.join(root, 'climate', 'stations', 'ghcn_stations.csv')
    # obs = os.path.join(root, 'climate', 'ghcn', 'station_data')
    # src_ = 'ghcn'

    joined = os.path.join(data, 'dads', 'met', 'joined')
    now_str = datetime.datetime.now().strftime('%Y%m%d%H%M')
    missing_list = os.path.join(data, 'dads', 'met', f'join_missing_{now_str}.csv')

    clip_to_obs_ = True
    overwrite = False

    era5_land = os.path.join(data, 'dads', 'era5_land', 'processed_parquet')
    daily = os.path.join(era5_land, 'daily')
    hourly_ = os.path.join(era5_land, 'hourly')

    join_daily_timeseries(sites, obs, daily, joined, source=src_, overwrite=overwrite,
                          bounds=(-180., 25., -60., 85.), shuffle=True, write_missing=missing_list, hourly=False,
                          clip_to_obs=clip_to_obs_)

# ========================= EOF ====================================================================
