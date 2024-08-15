import os
import pytz

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from refet import calcs

from extract.met_data.down_gridded import station_par_map
from utils.calc_eto import calc_asce_params

PACIFIC = pytz.timezone('US/Pacific')

NLDAS_RESAMPLE_MAP = {'rsds': 'sum',
                      'rlds': 'sum',
                      'prcp': 'sum',
                      'humidity': 'mean',
                      'min_temp': 'min',
                      'max_temp': 'max',
                      'mean_temp': 'mean',
                      'wind': 'mean',
                      'ea': 'mean'}


def extract_met_data(stations, gridded_dir, overwrite=False, station_type='openet',
                     gridmet=False, shuffle=True, bounds=None):
    kw = station_par_map(station_type)

    station_list = pd.read_csv(stations, index_col=kw['index'])

    if shuffle:
        station_list = station_list.sample(frac=1)

    if bounds:
        w, s, e, n = bounds
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]

    record_ct = station_list.shape[0]
    for i, (fid, row) in enumerate(station_list.iterrows(), start=1):

        lon, lat, elv = row[kw['lon']], row[kw['lat']], row[kw['elev']]
        if np.isnan(elv):
            print('{} has nan elevation'.format(fid))
            continue

        print('{}: {} of {}; {:.2f}, {:.2f}'.format(fid, i, record_ct, lat, lon))

        in_file_ = os.path.join(gridded_dir, 'nldas2_raw', '{}.csv'.format(fid))
        if not os.path.exists(in_file_):
            print('Input NLDAS at {} does not exist, skipping'.format(os.path.basename(in_file_)))
            # TODO: write missing stations
            continue

        out_file_ = os.path.join(gridded_dir, 'nldas2', '{}.csv'.format(fid))

        if not os.path.exists(out_file_) or overwrite:
            proc_nldas(in_csv=in_file_, lon=lon, lat=lat, elev=elv, out_csv=out_file_)
            print('nldas', fid)
        else:
            print('nldas {} exists'.format(fid))

        if gridmet:

            in_file_gm = os.path.join(gridded_dir, 'gridmet_raw', '{}.csv'.format(fid))
            out_file_gm = os.path.join(gridded_dir, 'gridmet', '{}.csv'.format(fid))

            if not os.path.exists(out_file_gm) or overwrite:
                proc_gridmet(in_csv=in_file_gm, lat=lat, elev=elv, out_csv=out_file_gm)
                print('gridmet', fid)
            else:
                print('gridmet {} exists'.format(fid))


def proc_nldas(in_csv, lat, lon, elev, out_csv):
    df = pd.read_csv(in_csv, index_col='time', parse_dates=True)

    try:
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
        df['mean_temp'] = df['temp'].copy()

        df = df.resample('D').agg(NLDAS_RESAMPLE_MAP)

        df['doy'] = [i.dayofyear for i in df.index]

        asce_params = df.parallel_apply(calc_asce_params, lat=lat, elev=elev, zw=10, axis=1)
        # asce_params = df.apply(calc_asce_params, lat=lat, elev=elev, zw=10, axis=1)
        df[['mean_temp', 'vpd', 'rn', 'u2', 'eto']] = pd.DataFrame(asce_params.tolist(),
                                                                   index=df.index)
        df['year'] = [i.year for i in df.index]
        df['date_str'] = [i.strftime('%Y-%m-%d') for i in df.index]

    except KeyError:
        bad_files = os.path.join(os.path.dirname(__file__), 'bad_files.txt')
        with open(bad_files, 'a') as f:
            f.write(in_csv + '\n')
        return None

    df.to_csv(out_csv)


def proc_gridmet(in_csv, lat, elev, out_csv):
    df = pd.read_csv(in_csv, index_col=0, parse_dates=True)

    try:
        df['min_temp'] = df['min_temp'] - 273.15
        df['max_temp'] = df['max_temp'] - 273.15

        df['year'] = [i.year for i in df.index]
        df['doy'] = [i.dayofyear for i in df.index]
        df.index = df.index.tz_localize(PACIFIC)

        df['ea'] = calcs._actual_vapor_pressure(pair=calcs._air_pressure(elev),
                                                q=df['q'])

        df['rsds'] *= 0.0864

        asce_params = df.parallel_apply(calc_asce_params, lat=lat, elev=elev, zw=10, axis=1)
        # asce_params = df.apply(calc_asce_params, lat=lat, elev=elev, zw=10, axis=1)
        df[['mean_temp', 'vpd', 'rn', 'u2', 'eto']] = pd.DataFrame(asce_params.tolist(),
                                                                   index=df.index)

    except KeyError:
        bad_files = os.path.join(os.path.dirname(__file__), 'bad_files.txt')
        with open(bad_files, 'a') as f:
            f.write(in_csv + '\n')
        return None

    df.to_csv(out_csv)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS')

    pandarallel.initialize(nb_workers=6)

    sites = os.path.join(d, 'dads', 'met', 'stations', 'dads_stations_elev_mgrs.csv')

    grid_dir = os.path.join(d, 'dads', 'met', 'gridded')

    extract_met_data(sites, grid_dir, overwrite=False, station_type='dads',
                     shuffle=True, bounds=(-125., 25., -96., 49.), gridmet=True)

# ========================= EOF ====================================================================
