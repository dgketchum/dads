import datetime
import os
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

COMPARISON_VARS = ['mean_temp', 'vpd', 'rn', 'u2', 'eto']

STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']


def join_daily_timeseries(stations, sta_dir, nldas_dir, dst_dir, gridmet_dir=None, index='FID'):
    stations = pd.read_csv(stations, index_col=index)
    stations.sort_index(inplace=True)

    for year in range(2000, 2021):

        df = pd.DataFrame()
        for i, (f, row) in enumerate(stations.iterrows(), start=1):

            if f != 'MSLM8':
                continue

            nldas_file = os.path.join(nldas_dir, '{}.csv'.format(f))
            try:
                ndf = pd.read_csv(nldas_file, index_col=0, parse_dates=True)
            except FileNotFoundError:
                print('{} does not exist'.format(nldas_file))
                continue

            ndf.index = pd.DatetimeIndex([datetime.date(i.year, i.month, i.day) for i in ndf.index])
            ndf = ndf[COMPARISON_VARS]

            gridmet_file = os.path.join(gridmet_dir, '{}.csv'.format(f))
            try:
                gdf = pd.read_csv(gridmet_file, index_col=0, parse_dates=True)
            except FileNotFoundError:
                print('{} does not exist'.format(gridmet_file))
                continue

            gdf.index = pd.DatetimeIndex([datetime.date(i.year, i.month, i.day) for i in gdf.index])
            gdf = gdf[COMPARISON_VARS]
            grd_cols = ['{}_gm'.format(c) for c in gdf.columns]
            # TODO: remove this after running gridmet extract again
            gdf['vpd'] = gdf['vpd'].apply(ast.literal_eval).apply(lambda x: x[0])
            gdf.columns = grd_cols

            nld_cols = ['{}_nl'.format(c) for c in ndf.columns]
            ndf.columns = nld_cols

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

            params = None
            try:
                params = sdf.apply(clc, lat=row['latitude'], elev=row['elevation'], zw=2.0, axis=1)
                sdf[['rn', 'u2', 'vpd']] = pd.DataFrame(params.tolist(), index=sdf.index)
            except ValueError as e:
                print('{} error getting {} from returned value: {}'.format(e, ['rn', 'u2', 'vpd'], len(params)))
                continue

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

            idx = [i for i in sdf.index if i.year == year]
            sdf = sdf.loc[idx]

            df = pd.concat([df, sdf])
            fids = np.unique(df['FID'])
            for fid in fids:
                c = df[df['FID'] == fid]
                idx = [i for i in c.index if i.year == year]
                c = c.loc[idx]
                c.dropna(subset=['mean_temp_obs', 'vpd_obs', 'rn_obs', 'u2_obs', 'eto_obs'], inplace=True, axis=0)
                if c.empty:
                    continue
                else:
                    out = os.path.join(dst_dir, 'obs_grid', '{}_{}.csv'.format(fid, year))
                    c.to_csv(out)
                print(fid, year)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    fields = os.path.join(d, 'met', 'stations', 'dads_stations.csv')

    obs = os.path.join(d, 'met', 'obs', 'gwx')
    gm = os.path.join(d, 'met', 'gridded', 'gridmet')
    nl = os.path.join(d, 'met', 'gridded', 'nldas2')
    # rs = os.path.join(d, 'rs', 'gwx_stations', 'concatenated', 'bands.csv')

    rs = os.path.join(d, 'rs', 'dads_stations', 'landsat')
    joined = os.path.join(d, 'training')
    plots = os.path.join(d, 'plots', 'gridmet')

    join_daily_timeseries(fields, obs, nl, joined, gm, index='index')

# ========================= EOF ====================================================================
