import os

import pandas as pd
import pynldas2 as nld

from extract.met_data.thredds import GridMet
from utils.station_parameters import station_par_map

REQUIRED_GRID_COLS = ['prcp', 'mean_temp', 'vpd', 'rn', 'u2', 'eto']


def extract_met_data(stations, gridded_dir, overwrite=False, station_type='openet', gridmet=False, shuffle=True,
                     bounds=None):
    kw = station_par_map(station_type)

    station_list = pd.read_csv(stations, index_col='index')

    if shuffle:
        station_list = station_list.sample(frac=1)

    if bounds:
        w, s, e, n = bounds
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]
    else:
        # NLDAS-2 extent
        ln = station_list.shape[0]
        w, s, e, n = (-125.0, 25.0, -67.0, 53.0)
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]
        print('dropped {} stations outside NLDAS-2 extent'.format(ln - station_list.shape[0]))

    record_ct = station_list.shape[0]
    for i, (fid, row) in enumerate(station_list.iterrows(), start=1):

        # if fid != 'TMPF7':
        #     continue

        lon, lat, elv = row[kw['lon']], row[kw['lat']], row[kw['elev']]
        print('{}: {} of {}; {:.2f}, {:.2f}'.format(fid, i, record_ct, lat, lon))

        _file = os.path.join(gridded_dir, 'nldas2_raw', '{}.csv'.format(fid))
        if not os.path.exists(_file) or overwrite:
            df = get_nldas(lon, lat)
            if df is None:
                continue
            df.to_csv(_file)
            print('nldas', fid)

        else:
            print('nldas {} exists'.format(fid))

        if gridmet:
            _file = os.path.join(gridded_dir, 'gridmet_raw', '{}.csv'.format(fid))
            if not os.path.exists(_file) or overwrite:
                df = get_gridmet(lon=lon, lat=lat)
                df.to_csv(_file)
                print('gridmet', fid)

            else:
                print('gridmet {} exists'.format(fid))


def get_nldas(lon, lat, start='2000-01-01', end='2023-12-31'):
    df = nld.get_bycoords((lon, lat), start_date=start, end_date=end, source='grib',
                          variables=['prcp', 'temp', 'wind_u', 'wind_v', 'rlds', 'rsds', 'humidity'])

    if df.empty:
        return None
    else:
        return df


def get_gridmet(lon, lat, start='2000-01-01', end='2023-12-31'):
    s = None
    df, cols = pd.DataFrame(), gridmet_par_map()
    for thredds_var, variable in cols.items():

        if not thredds_var:
            continue

        try:
            g = GridMet(thredds_var, start=start, end=end, lat=lat, lon=lon)
            s = g.get_point_timeseries()
        except OSError as e:
            print('Error on {}, {}'.format(variable, e))
        except RuntimeError as e:
            print('Error on {}, {}'.format(variable, e))
        except Exception as e:
            print('Error on {}, {}'.format(variable, e))
        try:
            df[variable] = s[thredds_var]
        except KeyError:
            continue

    return df


def gridmet_par_map():
    return {
        'pr': 'prcp',
        'pet': 'eto',
        'srad': 'rsds',
        'tmmx': 'max_temp',
        'tmmn': 'min_temp',
        'vs': 'wind',
        'sph': 'q',
    }


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS')

    # pandarallel.initialize(nb_workers=6)

    madis_data_dir_ = os.path.join(d, 'climate', 'madis')
    sites = os.path.join(d, 'dads', 'met', 'stations', 'dads_stations_res_elev_mgrs.csv')

    grid_dir = os.path.join(d, 'dads', 'met', 'gridded')

    extract_met_data(sites, grid_dir, overwrite=False, station_type='dads',
                     shuffle=True, bounds=(-125., 25., -66., 49.), gridmet=True)

# ========================= EOF ====================================================================
