import datetime
import os
import warnings

import pandas as pd
from refet import Daily, calcs
from extract.met_data.extract_met import calcs_ as clc

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


def join_daily_timeseries(stations, sta_dir, nldas_dir, rs_data, gridmet_dir, dst_dir, index='FID'):
    stations = pd.read_csv(stations, index_col=index)
    stations = stations.loc[[i for i, r in stations.iterrows() if r['State'] in STATES]]
    df = pd.DataFrame()
    for i, (f, row) in enumerate(stations.iterrows(), start=1):

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
        gdf.columns = grd_cols

        grd_cols = ['{}_gm'.format(c) for c in ndf.columns]
        ndf.columns = grd_cols

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
            params = sdf.apply(clc, lat=row['STATION_LAT'], elev=row['STATION_ELEV_M'], zw=row['Anemom_height_m'],
                               axis=1)
            sdf[['rn', 'u2', 'vpd']] = pd.DataFrame(params.tolist(), index=sdf.index)
        except ValueError as e:
            df.to_csv(os.path.join(dst_dir))
            print('{} error getting {} from returned value: {}'.format(e, ['rn', 'u2', 'vpd'], len(params)))
            continue

        sdf = sdf[COMPARISON_VARS]
        obs_cols = ['{}_obs'.format(c) for c in sdf.columns]
        sdf.columns = obs_cols

        all_cols = ['FID'] + obs_cols + grd_cols
        sdf = pd.concat([sdf, gdf, ndf], ignore_index=False, axis=1)
        sdf['FID'] = f
        sdf = sdf[all_cols]
        df = pd.concat([df, sdf])
        print(f, '{} of {}'.format(i, stations.shape[0]))

    df.to_csv(os.path.join(dst_dir))


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    fields = os.path.join(d, 'met', 'stations', 'gwx_stations.csv')
    sta = os.path.join(d, 'met', 'obs', 'gwx')
    gm = os.path.join(d, 'met', 'gridded', 'gridmet')
    nl = os.path.join(d, 'met', 'gridded', 'nldas2')
    # rs = os.path.join(d, 'rs', 'gwx_stations', 'concatenated', 'bands.csv')

    rs = os.path.join(d, 'rs', 'gwx_stations')
    joined = os.path.join(d, 'tables', 'gridmet', 'western_lst_metvars_all.csv')
    plots = os.path.join(d, 'plots', 'gridmet')

    join_daily_timeseries(fields, sta, nl, rs, gm, joined, index='STATION_ID')

# ========================= EOF ====================================================================
