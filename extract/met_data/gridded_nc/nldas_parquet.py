import calendar
import os
import gc
import tempfile
import shutil
import calendar
import concurrent.futures
from datetime import datetime

import earthaccess
from earthaccess.results import DataGranule
import numpy as np
import pandas as pd
import xarray as xr


def process_and_concat_csv(stations, root, start_date, end_date, outdir, workers):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    required_months = pd.date_range(start=start, end=end, freq='MS').strftime('%Y%m').tolist()
    expected_index = pd.date_range(start=start, end=end, freq='h')
    strdt = [d.strftime('%Y%m%d%H') for d in expected_index]

    station_list = pd.read_csv(stations)
    w, s, e, n = (-125.0, 25.0, -67.0, 53.0)
    station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
    station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]

    if 'LAT' in station_list.columns:
        station_list = station_list.rename(columns={'STAID': 'fid', 'LAT': 'latitude', 'LON': 'longitude'})

    station_list = station_list.sample(frac=1)
    subdirs = station_list['fid'].to_list()

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        executor.map(process_parquet, [root] * len(subdirs), subdirs,
                     [required_months] * len(subdirs),
                     [expected_index] * len(subdirs), [strdt] * len(subdirs),
                     [outdir] * len(subdirs))


def process_parquet(root_, subdir_, required_months_, expected_index_, strdt_, outdir_):
    subdir_path = os.path.join(root_, subdir_)
    if os.path.isdir(subdir_path):

        out_file = os.path.join(outdir_, f'{subdir_}.parquet.gzip')
        csv_files_ = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]

        if os.path.exists(out_file) and csv_files_:
            shutil.rmtree(subdir_path)
            return

        dtimes = [f.split('_')[-1].replace('.csv', '') for f in csv_files_]
        rm_files = csv_files_.copy()

        if len(dtimes) < len(required_months_):
            missing = [m for m in required_months_ if m not in dtimes]
            if len(missing) > 0:
                print(f'{subdir_} missing {len(missing)} months: {missing}')
                return

        dfs = []
        for file in csv_files_:
            c = pd.read_csv(os.path.join(subdir_path, file), parse_dates=['dt'],
                            date_format='%Y%m%d%H')
            dfs.append(c)
        df = pd.concat(dfs)

        df = df.set_index('dt').sort_index()
        df = df.drop(columns=['fid', 'time_bnds'])

        missing = len(expected_index_) - df.shape[0]
        if missing > 10:
            counts = {}
            missing_idx = [i for i in expected_index_ if i not in df.index]
            for midx in missing_idx:
                dt = f'{midx.year}-{midx.month:02}'
                if dt not in counts.keys():
                    counts[dt] = 1
                else:
                    counts[dt] += 1

            print(f'{subdir_} is missing {missing} rows')
            [print(k, v) for k, v in counts.items()]
            return

        elif missing > 0:
            df = df.reindex(expected_index_)
            df = df.interpolate(method='linear')

        df['dt'] = strdt_

        df.to_parquet(out_file, compression='gzip')
        shutil.rmtree(subdir_path)
        print(f'wrote {subdir_}, removed {len(rm_files)} .csv files,'
              f' {datetime.strftime(datetime.now(), '%Y%m%d %H:%M')}')
        return
    else:
        print(f'{subdir_} not found')
        return


def get_quadrants(b):
    mid_longitude = (b[0] + b[2]) / 2
    mid_latitude = (b[1] + b[3]) / 2
    quadrant_nw = (b[0], mid_latitude, mid_longitude, b[3])
    quadrant_ne = (mid_longitude, mid_latitude, b[2], b[3])
    quadrant_sw = (b[0], b[1], mid_longitude, mid_latitude)
    quadrant_se = (mid_longitude, b[1], b[2], mid_latitude)
    quadrants = [quadrant_nw, quadrant_ne, quadrant_sw, quadrant_se]
    return quadrants


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS'

    if not os.path.isdir(d):
        d = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS')

    if not os.path.isdir(d):
        d = os.path.join('/home', 'ec2-user', 'data', 'IrrigationGIS')

    if not os.path.isdir(d):
        d = os.path.join('/home', 'dketchum', 'data', 'IrrigationGIS')

    # sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_29OCT2024.csv')

    csv_files = '/data/ssd1/nldas2/station_data/'
    # csv_files = os.path.join(d, 'dads', 'met', 'gridded', 'nldas2', 'station_data')

    p_files = os.path.join(d, 'dads', 'met', 'gridded', 'nldas2_parquet')

    process_and_concat_csv(sites, csv_files, start_date='1990-01-01', end_date='2023-12-31', outdir=p_files,
                           workers=1)

# ========================= EOF ====================================================================
