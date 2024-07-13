import os
import pytz
import configparser

import numpy as np
import pandas as pd
import pynldas2 as nld
from refet import calcs

from thredds import GridMet
from nldas_eto_error import station_par_map
from qaqc.agweatherqaqc import WeatherQC

PACIFIC = pytz.timezone('US/Pacific')

RESAMPLE_MAP = {'rsds': 'mean',
                'humidity': 'mean',
                'min_temp': 'min',
                'max_temp': 'max',
                'wind': 'mean'}


def extract_gridded(stations, proc_dir, model='nldas'):
    kw = station_par_map('agri')
    station_list = pd.read_csv(stations, index_col=kw['index'])

    template = '{}_template.ini'.format(model)

    for i, (fid, row) in enumerate(station_list.iterrows()):

        if fid != 'bfam':
            continue

        lat, lon, elv = row[kw['lon']], row[kw['lat']], row[kw['elev']]

        station_dir = os.path.join(proc_dir, model, fid)
        if not os.path.isdir(station_dir):
            os.mkdir(station_dir)

        if model == 'nldas2':
            df = get_nldas(lat, lon, elv)
        elif model == 'gridmet':
            df = get_gridmet(lat, lon, elv)
        else:
            raise NotImplementedError('Choose "nldas2" or "gridmet" model')

        file_unproc = os.path.join(station_dir, '{}_input.csv'.format(fid))
        df.to_csv(file_unproc, index=False)

        ini_unproc = os.path.join(station_dir, '{}_input.ini'.format(fid))
        modify_config(template, ini_unproc, file_unproc, lat, lon, elv)

        qaqc = WeatherQC(ini_unproc)
        qaqc.process_station()


def get_nldas(lon, lat, elev, start='1989-01-01', end='2023-12-31'):
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


def get_gridmet(lon, lat, elev, start='1989-01-01', end='2023-12-31'):
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


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS', 'milk')

    station_meta = os.path.join(d, 'bias_ratio_data_processing/ETo/'
                                   'final_milk_river_metadata_nldas_eto_bias_ratios_long_term_mean.csv')

    data_proc = os.path.join(d, 'weather_station_data_processing', 'gridded')

    extract_gridded(station_meta, data_proc)
# ========================= EOF ====================================================================
