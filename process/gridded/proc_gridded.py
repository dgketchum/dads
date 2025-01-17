import os
import pytz

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from refet import calcs

from utils.station_parameters import station_par_map
from utils.calc_eto import calc_asce_params

PACIFIC = pytz.timezone('US/Pacific')

NLDAS_RESAMPLE_MAP = {'rsds': 'sum',
                      'rlds': 'sum',
                      'prcp': 'sum',
                      'q': 'mean',
                      'tmin': 'min',
                      'tmax': 'max',
                      'tmean': 'mean',
                      'wind': 'mean',
                      'ea': 'mean'}


def extract_met_data(stations, gridded_dir, overwrite=False, station_type='openet',
                     shuffle=True, bounds=None, hourly=False, **targets):
    """"""
    kw = station_par_map(station_type)

    station_list = pd.read_csv(stations, index_col=kw['index'])

    if shuffle:
        station_list = station_list.sample(frac=1)

    if bounds:
        w, s, e, n = bounds
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]

    for i, (fid, row) in enumerate(station_list.iterrows(), start=1):

        # if fid != 'COVM':
        #     continue

        lon, lat, elv = row[kw['lon']], row[kw['lat']], row[kw['elev']]
        if np.isnan(elv):
            print('{} has nan elevation'.format(fid))
            continue

        if targets['nldas2']:
            in_file_ = os.path.join(gridded_dir, 'raw_parquet', 'nldas2', '{}.parquet.gzip'.format(fid))
            if not os.path.exists(in_file_):
                print('Input NLDAS at {} does not exist, skipping'.format(os.path.basename(in_file_)))
                continue

            if hourly:
                out_file_ = os.path.join(gridded_dir, 'processed_parquet', 'nldas2_hourly', '{}.parquet'.format(fid))
            else:
                out_file_ = os.path.join(gridded_dir, 'processed_parquet', 'nldas2', '{}.parquet'.format(fid))

            if not os.path.exists(out_file_) or overwrite:
                proc_nldas(in_file_=in_file_, lat=lat, elev=elv, out_file_=out_file_, hourly_=hourly)
                print('nldas', fid)
            else:
                print('nldas {} exists'.format(fid))
                pass

        if targets['gridmet']:

            in_file_ = os.path.join(gridded_dir, 'raw_parquet', 'gridmet', '{}.parquet.gzip'.format(fid))
            if not os.path.exists(in_file_):
                print('Input GridMET at {} does not exist, skipping'.format(os.path.basename(in_file_)))
                continue

            out_file_gm = os.path.join(gridded_dir, 'processed_parquet', 'gridmet', '{}.parquet'.format(fid))

            if not os.path.exists(out_file_gm) or overwrite:
                proc_gridmet(in_file_=in_file_, lat=lat, elev=elv, out_file_=out_file_gm)
                print('gridmet', fid)
            else:
                pass

        if targets['prism']:

            in_file_ = os.path.join(gridded_dir, 'raw_parquet', 'prism', '{}.parquet.gzip'.format(fid))
            if not os.path.exists(in_file_):
                print('Input PRISM at {} does not exist, skipping'.format(os.path.basename(in_file_)))
                continue

            out_file_gm = os.path.join(gridded_dir, 'processed_parquet', 'prism', '{}.parquet'.format(fid))

            if not os.path.exists(out_file_gm) or overwrite:
                proc_prism(in_file_=in_file_, elev=elv, out_file_=out_file_gm)
                print('prism', fid)
            else:
                pass

        if targets['daymet']:

            in_file_ = os.path.join(gridded_dir, 'raw_parquet', 'daymet', '{}.parquet.gzip'.format(fid))
            if not os.path.exists(in_file_):
                print('Input DAYMET at {} does not exist, skipping'.format(os.path.basename(in_file_)))
                continue

            out_file_gm = os.path.join(gridded_dir, 'processed_parquet', 'daymet', '{}.parquet'.format(fid))

            if not os.path.exists(out_file_gm) or overwrite:
                proc_daymet(in_file_=in_file_, lat=lat, elev=elv, out_file_=out_file_gm)
                print('daymet', fid)
            else:
                pass

def proc_nldas(in_file_, lat, elev, out_file_, hourly_=False):
    df = pd.read_parquet(in_file_)

    try:
        wind_u = df['Wind_E']
        wind_v = df['Wind_N']
        df['wind'] = np.sqrt(wind_v ** 2 + wind_u ** 2)

        df['temp'] = df['Tair'] - 273.15

        df['rsds'] = df['SWdown'] * 0.0036
        df['rlds'] = df['LWdown'] * 0.0036
        df['prcp'] = df['Rainf']
        df['q'] = df['Qair']

        df['hour'] = [i.hour for i in df.index]

        df['ea'] = calcs._actual_vapor_pressure(pair=calcs._air_pressure(elev),
                                                q=df['q'])

        df['tmean'] = df['temp'].copy()

        if hourly_:
            df['doy'] = [i.dayofyear for i in df.index]
            df.to_parquet(out_file_)
            return

        df['tmax'] = df['temp'].copy()
        df['tmin'] = df['temp'].copy()

        df = df.resample('D').agg(NLDAS_RESAMPLE_MAP)

        df['doy'] = [i.dayofyear for i in df.index]

        asce_params = df.parallel_apply(calc_asce_params, lat=lat, elev=elev, zw=10, axis=1)
        # asce_params = df.apply(calc_asce_params, lat=lat, elev=elev, zw=10, axis=1)
        df[['tmean', 'vpd', 'rn', 'u2', 'eto']] = pd.DataFrame(asce_params.tolist(),
                                                               index=df.index)
        df['year'] = [i.year for i in df.index]
        df['date_str'] = [i.strftime('%Y-%m-%d') for i in df.index]

    except KeyError as exc:
        bad_files = os.path.join(os.path.dirname(__file__), 'bad_files.txt')
        with open(bad_files, 'a') as f:
            f.write(in_file_ + '\n')
        return None

    df.to_parquet(out_file_)


def proc_gridmet(in_file_, lat, elev, out_file_):
    df = pd.read_parquet(in_file_)

    try:
        df['tmin'] = df['tmmn'] - 273.15
        df['tmax'] = df['tmmx'] - 273.15
        df['q'] = df['sph']

        df.index = pd.to_datetime(df.index)
        df['year'] = [i.year for i in df.index]
        df['doy'] = [i.dayofyear for i in df.index]

        df['ea'] = calcs._actual_vapor_pressure(pair=calcs._air_pressure(elev),
                                                q=df['q'])

        df['rsds'] = df['srad'] * 0.0864
        df['wind'] = df['vs']

        df['prcp'] = df['pr']

        asce_params = df.parallel_apply(calc_asce_params, lat=lat, elev=elev, zw=10, axis=1)
        # asce_params = df.apply(calc_asce_params, lat=lat, elev=elev, zw=10, axis=1)
        df[['tmean', 'vpd', 'rn', 'u2', 'eto']] = pd.DataFrame(asce_params.tolist(),
                                                               index=df.index)

    except KeyError as exc:
        bad_files = os.path.join(os.path.dirname(__file__), 'bad_files.txt')
        with open(bad_files, 'a') as f:
            f.write(in_file_ + '\n')
        return None

    df.to_parquet(out_file_)


def proc_prism(in_file_, elev, out_file_):
    """
    https://daymet.ornl.gov/overview
    """
    df = pd.read_parquet(in_file_)

    try:
        df.rename(columns={'vp': 'vpd'}, inplace=True)
        df['vpd'] /= 1000.
        es = 0.6108 * np.exp(17.27 * df['tmean'] / (df['tmean'] + 237.3))
        ea = es - df['vpd']
        df['q'] = (0.622 * ea) / (calcs._air_pressure(elev) - (0.378 * ea))

        # ((srad(W / m2) * dayl(s / df['dayl'])) / 1e6)

        df.index = pd.to_datetime(df.index)
        df['year'] = [i.year for i in df.index]
        df['doy'] = [i.dayofyear for i in df.index]

    except KeyError as exc:
        bad_files = os.path.join(os.path.dirname(__file__), 'bad_files.txt')
        with open(bad_files, 'a') as f:
            f.write(in_file_ + '\n')
        return None

    df.to_parquet(out_file_)


def proc_daymet(in_file_, lat, elev, out_file_):
    df = pd.read_parquet(in_file_)

    try:

        es = 0.6108 * np.exp(17.27 * df['tmean'] / (df['tmean'] + 237.3))
        df['vpd'] = (df['vpdmin'] + df['vpdmax']) * 0.5
        ea = es - df['vpd']
        df['q'] = (0.622 * ea) / (calcs._air_pressure(elev) - (0.378 * ea))

        df.index = pd.to_datetime(df.index)
        df['year'] = [i.year for i in df.index]
        df['doy'] = [i.dayofyear for i in df.index]

        df['ea'] = calcs._actual_vapor_pressure(pair=calcs._air_pressure(elev),
                                                q=df['q'])

        df['rsds'] = df['srad'] * 0.0864
        df['wind'] = df['vs']

        asce_params = df.parallel_apply(calc_asce_params, lat=lat, elev=elev, zw=10, axis=1)
        # asce_params = df.apply(calc_asce_params, lat=lat, elev=elev, zw=10, axis=1)
        df[['tmean', 'vpd', 'rn', 'u2', 'eto']] = pd.DataFrame(asce_params.tolist(),
                                                               index=df.index)

    except KeyError as exc:
        bad_files = os.path.join(os.path.dirname(__file__), 'bad_files.txt')
        with open(bad_files, 'a') as f:
            f.write(in_file_ + '\n')
        return None

    df.to_parquet(out_file_)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS')

    overwrite = True
    processing_targets = {'nldas2': False, 'gridmet': False,
                          'prism': False, 'daymet': True}

    pandarallel.initialize(nb_workers=4)

    sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_29OCT2024.csv')

    grid_dirs = os.path.join(d, 'dads', 'met', 'gridded')

    extract_met_data(sites, grid_dirs, overwrite=overwrite, station_type='madis', shuffle=True,
                     bounds=None, hourly=False, **processing_targets)

# ========================= EOF ====================================================================
