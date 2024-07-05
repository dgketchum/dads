import json
import os
import pytz
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import geopandas as gpd
import pynldas2 as nld
from refet import Daily, calcs
from scipy.stats import skew, kurtosis

from extract.gridmet.thredds import GridMet
from extract.gridmet.thredds import air_pressure, actual_vapor_pressure
from extract.agrimet import Agrimet

warnings.simplefilter(action='ignore', category=FutureWarning)

VAR_MAP = {'rsds': 'Rs (w/m2)',
           'ea': 'Compiled Ea (kPa)',
           'min_temp': 'TMin (C)',
           'max_temp': 'TMax (C)',
           'temp': 'TAvg (C)',
           'wind': 'Windspeed (m/s)',
           'eto': 'ETo (mm)'}

RESAMPLE_MAP = {'rsds': 'mean',
                'humidity': 'mean',
                'min_temp': 'min',
                'max_temp': 'max',
                'wind': 'mean',
                'doy': 'first'}

RENAME_MAP = {v: k for k, v in VAR_MAP.items()}

COMPARISON_VARS = ['vpd', 'rn', 'tmean', 'u2', 'eto']

STR_MAP = {
    'rn': r'Net Radiation [MJ m$^{-2}$ d$^{-1}$]',
    'vpd': r'Vapor Pressure Deficit [kPa]',
    'tmean': r'Mean Daily Temperature [K]',
    'u2': r'Wind Speed at 2 m [m s$^{-1}$]',
    'eto': r'ASCE Grass Reference Evapotranspiration [mm day$^{-1}$]'
}

STR_MAP_SIMPLE = {
    'rn': r'Rn',
    'vpd': r'VPD',
    'tmean': r'Mean Temp',
    'u2': r'Wind Speed',
    'eto': r'ETo'
}

LIMITS = {'vpd': 3,
          'rn': 0.8,
          'u2': 12,
          'tmean': 12.5,
          'eto': 5}

PACIFIC = pytz.timezone('US/Pacific')


def residuals(stations, resids, out_data, check_dir=None, overwrite=False):
    kw = station_par_map('agri')
    stations = gpd.read_file(stations)
    stations.index = stations['FID']

    s, e = '1990-01-01', '2023-12-31'

    errors, all_res_dict = {}, {v: [] for v in COMPARISON_VARS}
    for i, (fid, row) in enumerate(stations.iterrows()):

        sta_res = {v: [] for v in COMPARISON_VARS}
        print('{} of {}: {}'.format(i + 1, stations.shape[0], fid))

        # try:
        station_file = os.path.join(out_data, '{}.csv'.format(fid))

        if not os.path.exists(station_file) or overwrite:
            ag = Agrimet(start_date=s, end_date=e, station='lmmm')
            ag.region = row['region']
            sdf = ag.fetch_met_data()

            sdf['ea'] = calcs._sat_vapor_pressure(sdf['t_dew'])

            sdf.index = sdf.index.tz_localize(PACIFIC)
            sdf = sdf.rename(RENAME_MAP, axis=1)
            sdf['doy'] = [i.dayofyear for i in sdf.index]
            sdf.dropna(how='any', inplace=True, axis=0)

            # TODO: find metadata for anemometer height
            _zw = 2.0

            # rs in MJ m-2 day-1
            def calc_asce_params(r, zw):
                asce = Daily(tmin=r['min_temp'],
                             tmax=r['max_temp'],
                             ea=r['ea'],
                             rs=r['rsds'],
                             uz=r['wind'],
                             zw=zw,
                             doy=r['doy'],
                             elev=row[kw['elev']],
                             lat=row[kw['lat']])

                vpd = asce.vpd[0]
                rn = asce.rn[0]
                u2 = asce.u2[0]
                mean_temp = asce.tmean[0]
                eto = asce.eto()[0]

                return vpd, rn, u2, mean_temp, eto

            asce_params = sdf.apply(calc_asce_params, zw=_zw, axis=1)
            sdf_cols = ['vpd_ob', 'rn_ob', 'u2_ob', 'tmean_ob', 'eto_ob']
            sdf[sdf_cols] = pd.DataFrame(asce_params.tolist(), index=sdf.index)
            gs, ge = sdf.index[0].strftime('%Y-%m-%d'), sdf.index[-1].strftime('%Y-%m-%d')

            nld = get_nldas(row[kw['lon']], row[kw['lat']], row[kw['elev']], start=gs, end=ge)
            asce_params = nld.apply(calc_asce_params, zw=_zw, axis=1)
            nld_cols = ['vpd_nl', 'rn_nl', 'u2_nl', 'tmean_nl', 'eto_nl']
            nld[nld_cols] = pd.DataFrame(asce_params.tolist(), index=nld.index)

            gmt = get_gridmet(row[kw['lon']], row[kw['lat']], start=gs, end=ge)
            asce_params = gmt.apply(calc_asce_params, zw=_zw, axis=1)
            gmt_cols = ['vpd_gm', 'rn_gm', 'u2_gm', 'tmean_gm', 'eto_gm']
            gmt[gmt_cols] = pd.DataFrame(asce_params.tolist(), index=gmt.index)

            grid = pd.concat([sdf, nld, gmt], axis=1, ignore_index=False)
            target_cols = [[s, n, g] for s, n, g in zip(sdf_cols, nld_cols, gmt_cols)]
            target_cols = [item for sublist in target_cols for item in sublist]
            grid = grid[target_cols]

            # TODO: gridmet ETo is not right
            res_df = sdf[['eto']].copy()

            if check_dir:
                check_file = os.path.join(check_dir, '{}_nldas_daily.csv'.format(fid))
                cdf = pd.read_csv(check_file, parse_dates=True, index_col='date')
                cdf.index = cdf.index.tz_localize(PACIFIC)
                indx = [i for i in cdf.index if i in grid.index]
                rsq = np.corrcoef(grid.loc[indx, 'eto'], cdf.loc[indx, 'eto_asce'])[0, 0]
                print('{} PyNLDAS/Earth Engine r2: {:.3f}'.format(row['station_name'], rsq))

            for var in COMPARISON_VARS:
                s_var, n_var = '{}_station'.format(var), '{}_nldas'.format(var)
                df = pd.DataFrame(columns=[s_var], index=sdf.index, data=sdf[var].values)
                df.dropna(how='any', axis=0, inplace=True)
                df[n_var] = grid.loc[df.index, var].values
                residuals = df[s_var] - df[n_var]
                res_df[var] = residuals
                sta_res[var] = list(residuals)
                all_res_dict[var] += list(residuals)

            grid = grid.loc[sdf.index]
            grid['obs_eto'] = sdf['eto']
            grid.to_csv(station_file)

            res_df['eto'] = sdf['eto'] - grid['eto']
            res_df.to_csv(resids)

            # except Exception as e:
            #     print('Exception at {}: {}'.format(fid, e))
            #     errors[fid] = 'exception'

    with open(resids, 'w') as dst:
        json.dump(all_res_dict, dst, indent=4)


def station_par_map(station_type):
    if station_type == 'ec':
        return {'index': 'SITE_ID',
                'lat': 'LATITUDE',
                'lon': 'LONGITUDE',
                'elev': 'ELEVATION (METERS)',
                'start': 'START DATE',
                'end': 'END DATE'}
    elif station_type == 'agri':
        return {'index': 'FID',
                'lat': 'lat',
                'lon': 'lon',
                'elev': 'elev',
                'start': 'record_start',
                'end': 'record_end'}
    else:
        raise NotImplementedError


def gridmet_par_map():
    return {
        'pet': 'eto',
        'srad': 'rsds',
        'tmmx': 'max_temp',
        'tmmn': 'min_temp',
        'vs': 'wind',
        'vpd': 'vpd',
    }


def get_nldas(lon, lat, elev, start, end):
    nldas = nld.get_bycoords((lon, lat), start_date=start, end_date=end,
                             variables=['temp', 'wind_u', 'wind_v', 'humidity', 'rsds'])

    nldas = nldas.tz_convert(PACIFIC)

    wind_u = nldas['wind_u']
    wind_v = nldas['wind_v']
    nldas['wind'] = np.sqrt(wind_v ** 2 + wind_u ** 2)

    nldas['min_temp'] = nldas['temp'] - 273.15
    nldas['max_temp'] = nldas['temp'] - 273.15
    nldas['doy'] = [i.dayofyear for i in nldas.index]

    # W m-2 to MJ m-2 day-1
    nldas['rsds'] = (nldas['rsds'] * 86400) / 1000000

    nldas = nldas.resample('D').agg(RESAMPLE_MAP)
    nldas['ea'] = calcs._actual_vapor_pressure(pair=calcs._air_pressure(elev),
                                               q=nldas['humidity'])

    return nldas


def get_gridmet(lon, lat, start, end):
    first = True
    df, cols = pd.DataFrame(), gridmet_par_map()

    for thredds_var, variable in cols.items():

        if not thredds_var:
            continue

        try:
            g = GridMet(thredds_var, start=start, end=end, lat=lat, lon=lon)
            s = g.get_point_timeseries()
        except OSError as e:
            print('Error on {}, {}'.format(variable, e))

        df[variable] = s[thredds_var]

        if first:
            df['date'] = [i.strftime('%Y-%m-%d') for i in df.index]
            df['year'] = [i.year for i in df.index]
            df['month'] = [i.month for i in df.index]
            df['day'] = [i.day for i in df.index]
            df['centroid_lat'] = [lat for _ in range(df.shape[0])]
            df['centroid_lon'] = [lon for _ in range(df.shape[0])]
            g = GridMet('elev', lat=lat, lon=lon)
            elev = g.get_point_elevation()
            df['elev_m'] = [elev for _ in range(df.shape[0])]
            first = False

    df['doy'] = [i.dayofyear for i in df.index]

    df['min_temp'] = df['min_temp'] - 273.15
    df['max_temp'] = df['max_temp'] - 273.15

    # W m-2 to MJ m-2 day-1
    df['rsds'] = (df['rsds'] * 86400) / 1000000

    es = 0.5 * (calcs._sat_vapor_pressure(df['min_temp']) +
                calcs._sat_vapor_pressure(df['max_temp']))
    df['ea'] = es - df['vpd']

    df.index = df.index.tz_localize(PACIFIC)

    return df


def concatenate_station_residuals(error_json, out_file):
    with open(error_json, 'r') as f:
        meta = json.load(f)

    eto_residuals = []
    first, df = True, None
    for sid, data in meta.items():

        if data == 'exception':
            continue

        _file = data['resid']
        c = pd.read_csv(_file, parse_dates=True, index_col='date')

        eto = c['eto'].copy()
        eto.dropna(how='any', inplace=True)
        eto_residuals += list(eto.values)

        c.dropna(how='any', axis=0, inplace=True)
        if first:
            df = c.copy()
            first = False
        else:
            df = pd.concat([df, c], ignore_index=True, axis=0)

    df.to_csv(out_file)


if __name__ == '__main__':

    r = '/media/research/IrrigationGIS'
    if not os.path.isdir(r):
        home = os.path.expanduser('~')
        r = os.path.join(home, 'data', 'IrrigationGIS')

    fields = os.path.join(r, 'climate', 'agrimet', 'agrimet_mt_aea_elev.shp')

    d = os.path.join(r, 'dads')
    comp_data = os.path.join(d, 'obs', 'agrimet', 'station_data')
    error_json = os.path.join(d, 'obs', 'agrimet', 'error_distributions.json')
    res_json = os.path.join(d, 'obs', 'agrimet', 'residuals.json')
    residuals(fields, res_json, comp_data)

# ========================= EOF ====================================================================
