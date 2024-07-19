import datetime
import os
import warnings

import pandas as pd
from refet import Daily, calcs

warnings.filterwarnings("ignore", category=FutureWarning)
VAR_MAP = {'rsds': 'Rs (w/m2)',
           'ea': 'Vapor Pres (kPa)',
           'min_temp': 'TMin (C)',
           'max_temp': 'TMax (C)',
           'mean_temp': 'TAvg (C)',
           'wind': 'ws_2m (m/s)',
           'eto': 'ETo (mm)'}

RENAME_MAP = {v: k for k, v in VAR_MAP.items()}

COMPARISON_VARS = ['vpd', 'rsds', 'min_temp', 'max_temp', 'mean_temp', 'wind', 'eto']

STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']


def join_daily_timeseries(stations, gridmet_dir, rs_dir, sta_dir, dst_dir, index='FID'):
    stations = pd.read_csv(stations, index_col=index)
    # stations = stations.loc[[i for i, r in stations.iterrows() if r['State'] in STATES]]
    df = pd.DataFrame()
    for i, (f, row) in enumerate(stations.iterrows(), start=1):

        lst = join_landsat(rs_dir, f)
        if lst is None:
            continue
        lst_cols = lst.columns

        gridmet_file = os.path.join(gridmet_dir, '{}.csv'.format(f))

        try:
            # TODO: remove the tz application in the data extract code
            gdf = pd.read_csv(gridmet_file, index_col=0, parse_dates=True)
        except FileNotFoundError:
            print('{} does not exist'.format(gridmet_file))
            continue

        gdf.index = pd.DatetimeIndex([datetime.date(i.year, i.month, i.day) for i in gdf.index])
        match_idx = [i for i in lst.index if i in gdf.index]
        gdf = gdf.loc[match_idx]
        gdf['rsds'] *= 0.0864
        gdf['mean_temp'] = (gdf['min_temp'] + gdf['max_temp']) * 0.5
        gdf['vpd'] = gdf.apply(_vpd, axis=1)
        gdf['rn'] = gdf.apply(_rn, lat=row['STATION_LAT'], elev=row['STATION_ELEV_M'],
                              zw=row['Anemom_height_m'], axis=1)
        gdf = gdf[COMPARISON_VARS]
        grd_cols = ['{}_gm'.format(c) for c in gdf.columns]
        gdf.columns = grd_cols
        gdf.loc[match_idx, lst.columns] = lst.loc[match_idx]

        sta_file = os.path.join(sta_dir, '{}.csv'.format(f))

        try:
            sdf = pd.read_csv(sta_file, index_col='Unnamed: 0', parse_dates=True)
        except ValueError:
            sdf = pd.read_csv(sta_file, index_col='date', parse_dates=True)
        except FileNotFoundError:
            print('{} does not exist'.format(sta_file))
            continue

        match_idx = [i for i in match_idx if i in sdf.index]
        sdf = sdf.loc[match_idx]
        sdf.rename(columns=RENAME_MAP, inplace=True)
        sdf['doy'] = [i.dayofyear for i in sdf.index]
        sdf['rsds'] *= 0.0864
        sdf['vpd'] = sdf.apply(_vpd, axis=1)
        sdf['rn'] = sdf.apply(_rn, lat=row['STATION_LAT'], elev=row['STATION_ELEV_M'],
                              zw=row['Anemom_height_m'], axis=1)

        sdf = sdf[COMPARISON_VARS]
        obs_cols = ['{}_obs'.format(c) for c in sdf.columns]
        sdf.columns = obs_cols
        all_cols = ['FID'] + obs_cols + grd_cols + list(lst_cols)
        sdf = pd.concat([sdf, gdf], ignore_index=False, axis=1)
        sdf['FID'] = f
        sdf = sdf[all_cols]
        df = pd.concat([df, sdf])
        print(f, '{} of {}'.format(i, stations.shape[0]))

    df.to_csv(os.path.join(dst_dir))


def join_landsat(dir_, glob):
    try:
        l = [os.path.join(dir_, x) for x in os.listdir(dir_) if glob in x]
        df = pd.concat([pd.read_csv(f).T for f in l])
        df.dropna(how='any', axis=0, inplace=True)
        splt = [i.split('_') for i in df.index]
        df['band'] = [i[-1] for i in splt]
    except Exception as e:
        print(glob, e)
        return None

    df.index = [pd.to_datetime(i[-2]) for i in splt]

    try:
        df = df.pivot(columns=['band'])
    except ValueError:
        df['index'] = [i.strftime('%Y%m%d') for i in df.index]
        df = df.groupby(['index', 'band']).mean().reset_index()
        df.index = pd.DatetimeIndex(df['index'])
        df.drop(columns=['index'], inplace=True)
        df = df.pivot(columns=['band'])

    df.columns = [c[1] for c in df.columns]
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]
    return df


def _vpd(r):
    es = calcs._sat_vapor_pressure(r['mean_temp'])
    vpd = es - r['ea']
    return vpd[0]


def _rn(r, lat, elev, zw):
    asce = Daily(tmin=r['min_temp'],
                 tmax=r['max_temp'],
                 rs=r['rsds'],
                 ea=r['ea'],
                 uz=r['wind'],
                 zw=zw,
                 doy=r['doy'],
                 elev=elev,
                 lat=lat)

    rn = asce.rn[0]
    return rn


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    fields = os.path.join(d, 'met', 'stations', 'gwx_stations.csv')
    sta = os.path.join(d, 'met', 'obs', 'gwx')
    gm = os.path.join(d, 'met', 'gridded', 'gridmet')
    rs = os.path.join(d, 'rs', 'gwx_stations')
    joined = os.path.join(d, 'tables', 'gridmet', 'western_lst_metvars_all.csv')
    plots = os.path.join(d, 'plots', 'gridmet')

    join_daily_timeseries(fields, gm, rs, sta, joined, index='STATION_ID')

# ========================= EOF ====================================================================
