import os
import pytz
import configparser

import numpy as np
import pandas as pd
import geopandas as gpd
import pynldas2 as nld
from refet import calcs, Daily
from pandarallel import pandarallel

from thredds import GridMet
from qaqc.agweatherqaqc import WeatherQC
from agrimet import Agrimet

PACIFIC = pytz.timezone('US/Pacific')

GRIDMET_RESAMPLE_MAP = {'rsds': 'mean',
                        'humidity': 'mean',
                        'min_temp': 'min',
                        'max_temp': 'max',
                        'wind': 'mean'}

NLDAS_RESAMPLE_MAP = {'rsds': 'sum',
                      'rlds': 'sum',
                      'humidity': 'mean',
                      'min_temp': 'min',
                      'max_temp': 'max',
                      'wind': 'mean',
                      'ea': 'mean'}

REQUIRED_GRID_COLS = ['tmean', 'vpd', 'rn', 'u2', 'eto']


def extract_met_data(stations, obs_dir, gridded_dir, overwrite=False, transfer=None):
    kw = station_par_map('openet')

    if stations.endswith('.csv'):
        station_list = pd.read_csv(stations, index_col=kw['index'])

    else:
        station_list = gpd.read_file(stations, index_col=kw['index'])
        station_list.drop(columns=['url', 'title', 'install', 'geometry'], inplace=True)

        station_list.index = station_list[kw['index']]

    for i, (fid, row) in enumerate(station_list.iterrows()):

        sta_file = os.path.join(obs_dir, '{}_data.xlsx'.format(fid))

        if transfer:
            try:
                dst_file = os.path.join(transfer, '{}.csv'.format(fid))
                if not os.path.exists(dst_file):
                    df = pd.read_excel(sta_file, index_col='date')
                    df.to_csv(dst_file)
                    print(fid)
            except FileNotFoundError as e:
                print(fid, e)

        lat, lon, elv = row[kw['lon']], row[kw['lat']], row[kw['elev']]

        _file = os.path.join(gridded_dir, 'nldas2', '{}.csv'.format(fid))
        if not os.path.exists(_file) or overwrite:
            df = get_nldas(lat, lon, elv)
            df.to_csv(_file)

        _file = os.path.join(gridded_dir, 'gridmet', '{}.csv'.format(fid))
        if not os.path.exists(_file) or overwrite:
            df = get_gridmet(lat, lon, elv)
            df.to_csv(_file)

        print(fid)


def get_nldas(lon, lat, elev, start='2000-01-01', end='2023-12-31'):
    df = nld.get_bycoords((lon, lat), start_date=start, end_date=end, source='grib',
                          variables=['temp', 'wind_u', 'wind_v', 'rlds', 'rsds', 'humidity'])

    df = df.tz_convert(PACIFIC)
    wind_u = df['wind_u']
    wind_v = df['wind_v']
    df['wind'] = np.sqrt(wind_v ** 2 + wind_u ** 2)

    df['temp'] = df['temp'] - 273.15

    df['rsds'] *= 0.0036
    df['rlds'] *= 0.0036

    df['hour'] = [i.hour for i in df.index]

    df['ea'] = calcs._actual_vapor_pressure(pair=calcs._air_pressure(elev),
                                            q=df['humidity'])

    df['max_temp'] = df['temp'].copy()
    df['min_temp'] = df['temp'].copy()

    df = df.resample('D').agg(NLDAS_RESAMPLE_MAP)

    df['doy'] = [i.dayofyear for i in df.index]

    def calc_asce_params(r, zw, lat, lon, elev):
        asce = Daily(tmin=r['min_temp'],
                     tmax=r['max_temp'],
                     rs=r['rsds'],
                     ea=r['ea'],
                     uz=r['wind'],
                     zw=zw,
                     doy=r['doy'],
                     elev=elev,
                     lat=lat,
                     method='asce')

        vpd = asce.vpd[0]
        rn = asce.rn[0]
        u2 = asce.u2[0]
        tmean = asce.tmean[0]
        eto = asce.eto()[0]

        return tmean, vpd, rn, u2, eto

    asce_params = df.parallel_apply(calc_asce_params, lat=lat, lon=lon, elev=elev, zw=10, axis=1)
    df[['tmean', 'vpd', 'rn', 'u2', 'eto']] = pd.DataFrame(asce_params.tolist(),
                                                           index=df.index)
    df['year'] = [i.year for i in df.index]
    df['date_str'] = [i.strftime('%Y-%m-%d') for i in df.index]

    df = df[REQUIRED_GRID_COLS]

    return df


def get_gridmet(lon, lat, elev, start='2000-01-01', end='2023-12-31'):
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

    df['min_temp'] = df['min_temp'] - 273.15
    df['max_temp'] = df['max_temp'] - 273.15

    df['year'] = [i.year for i in df.index]
    df['doy'] = [i.dayofyear for i in df.index]
    df.index = df.index.tz_localize(PACIFIC)

    df['ea'] = calcs._actual_vapor_pressure(pair=calcs._air_pressure(elev),
                                            q=df['q'])
    es = calcs._sat_vapor_pressure(df['mean_temp'])
    df['vpd'] = es - df['ea']

    df = df[REQUIRED_GRID_COLS]

    return df


def modify_config(template_file, output_file, data_file_path, latitude, longitude, elevation):
    config = configparser.ConfigParser()
    assert os.path.exists(template_file)
    config.read(template_file)

    config.set('METADATA', 'DATA_FILE_PATH', data_file_path)
    config.set('METADATA', 'LATITUDE', str(latitude))
    config.set('METADATA', 'LONGITUDE', str(longitude))
    config.set('METADATA', 'ELEVATION', str(elevation))

    with open(output_file, 'w') as configfile:
        config.write(configfile)


def gridmet_par_map():
    return {
        'pet': 'eto',
        'srad': 'rsds',
        'tmmx': 'max_temp',
        'tmmn': 'min_temp',
        'vs': 'wind',
        'sph': 'q',
    }


def station_par_map(station_type):
    if station_type == 'openet':
        return {'index': 'STATION_ID',
                'lat': 'LAT',
                'lon': 'LON',
                'elev': 'ELEV_M',
                'start': 'START DATE',
                'end': 'END DATE'}
    elif station_type == 'agri':
        return {'index': 'id',
                'lat': 'lat',
                'lon': 'lon',
                'elev': 'elev',
                'start': 'record_start',
                'end': 'record_end'}
    else:
        raise NotImplementedError


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/dads'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS', 'dads')

    pandarallel.initialize(nb_workers=8)

    station_meta = os.path.join(d, 'met', 'stations', 'gwx_stations.csv')

    obs = os.path.join(d, 'met', 'obs', 'corrected_station_data')
    dst_obs = os.path.join(d, 'met', 'obs', 'gwx')

    grid_dir = os.path.join(d, 'met', 'gridded')

    extract_met_data(station_meta, obs, grid_dir, transfer=dst_obs, overwrite=True)

# ========================= EOF ====================================================================
