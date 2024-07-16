import os
import pytz
import configparser

import numpy as np
import pandas as pd
import geopandas as gpd
import pynldas2 as nld
from refet import calcs

from thredds import GridMet
from qaqc.agweatherqaqc import WeatherQC
from agrimet import Agrimet

PACIFIC = pytz.timezone('US/Pacific')

RESAMPLE_MAP = {'rsds': 'mean',
                'humidity': 'mean',
                'min_temp': 'min',
                'max_temp': 'max',
                'wind': 'mean'}


def extract_met_data(stations, obs_dir, gridded_dir, source='agrimet', overwrite=False):
    kw = station_par_map('agri')

    if stations.endswith('.csv'):
        station_list = pd.read_csv(stations, index_col=kw['index'])
    else:
        station_list = gpd.read_file(stations, index_col=kw['index'])
        station_list.drop(columns=['url', 'title', 'install', 'geometry'], inplace=True)
        station_list.index = station_list['siteid']

    for i, (fid, row) in enumerate(station_list.iterrows()):

        lat, lon, elv = row[kw['lon']], row[kw['lat']], row[kw['elev']]

        df = get_nldas(lat, lon, elv)
        _file = os.path.join(gridded_dir, 'nldas2', '{}.csv'.format(fid))
        df.to_csv(_file)

        df = get_gridmet(lat, lon, elv)
        _file = os.path.join(gridded_dir, 'gridmet', '{}.csv'.format(fid))
        df.to_csv(_file)


def get_nldas(lon, lat, elev, start='2000-01-01', end='2023-12-31'):
    df = nld.get_bycoords((lon, lat), start_date=start, end_date=end, source='grib',
                          variables=['temp', 'wind_u', 'wind_v', 'humidity', 'rsds'])

    df = df.tz_convert(PACIFIC)
    wind_u = df['wind_u']
    wind_v = df['wind_v']
    df['wind'] = np.sqrt(wind_v ** 2 + wind_u ** 2)

    df['min_temp'] = df['temp'] - 273.15
    df['max_temp'] = df['temp'] - 273.15

    df = df.resample('D').agg(RESAMPLE_MAP)

    df['doy'] = [i.dayofyear for i in df.index]
    df['year'] = [i.year for i in df.index]

    df['ea'] = calcs._actual_vapor_pressure(pair=calcs._air_pressure(elev),
                                            q=df['humidity'])
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
    if station_type == 'ec':
        return {'index': 'SITE_ID',
                'lat': 'LATITUDE',
                'lon': 'LONGITUDE',
                'elev': 'ELEVATION (METERS)',
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

    station_meta = os.path.join(d, 'met', 'stations', 'openet_gridwxcomp_input.csv')

    obs = os.path.join(d, 'met', 'obs')

    grid_dir = os.path.join(d, 'met', 'gridded')

    extract_met_data(station_meta, obs, grid_dir, source='agrimet')
# ========================= EOF ====================================================================
