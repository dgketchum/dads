import concurrent.futures
import os
import json
import shutil
from datetime import datetime

import pandas as pd
import numpy as np


def process_and_concat_csv(stations, root, source, start_date, end_date, outdir, workers, missing_file=None):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    required_months = pd.date_range(start=start, end=end, freq='MS').strftime('%Y%m').tolist()
    required_years = pd.date_range(start=start, end=end, freq='Y').strftime('%Y').tolist()

    if source == 'gridmet':
        expected_index = pd.date_range(start=start, end=end, freq='d')
    elif source == 'nldas2':
        expected_index = pd.date_range(start=start, end=end, freq='h')

    strdt = [d.strftime('%Y%m%d%H') for d in expected_index]

    station_list = pd.read_csv(stations)
    if 'LAT' in station_list.columns:
        station_list = station_list.rename(columns={'STAID': 'fid', 'LAT': 'latitude', 'LON': 'longitude'})
    w, s, e, n = (-125.0, 25.0, -67.0, 53.0)
    station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
    station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]

    station_list = station_list.sample(frac=1)
    subdirs = station_list['fid'].to_list()
    subdirs = [s for s in subdirs if s in ['COVM', '1896P']]
    # subdirs.sort()

    if missing_file:
        for sd in subdirs:
            if source == 'nldas2':
                nldas2_parquet(root, sd, required_months, expected_index, strdt, outdir, missing_file)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        if source == 'nldas2':
            executor.map(nldas2_parquet, [root] * len(subdirs), subdirs,
                         [required_months] * len(subdirs),
                         [expected_index] * len(subdirs), [strdt] * len(subdirs),
                         [outdir] * len(subdirs))

        if source == 'gridmet':
            executor.map(gridmet_parquet, [root] * len(subdirs), subdirs,
                         [required_years] * len(subdirs),
                         [expected_index] * len(subdirs), [strdt] * len(subdirs))

def nldas2_parquet(root_, subdir_, required_months_, expected_index_, strdt_, outdir_, write_missing=None):
    subdir_path = os.path.join(root_, subdir_)
    out_file = os.path.join(outdir_, f'{subdir_}.parquet.gzip')

    if os.path.isdir(subdir_path):

        csv_files_ = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]

        if os.path.exists(out_file) and csv_files_:
            shutil.rmtree(subdir_path)
            print(f'{os.path.basename(out_file)} exists, removing {len(csv_files)} csv files')
            return

        dtimes = [f.split('_')[-1].replace('.csv', '') for f in csv_files_]
        rm_files = csv_files_.copy()

        if len(dtimes) < len(required_months_):
            missing = [m for m in required_months_ if m not in dtimes]
            if len(missing) > 0:
                print(f'{subdir_} missing {len(missing)} months: {np.random.choice(missing, size=5, replace=False)}')
                return

        dfs = []
        for file in csv_files_:
            c = pd.read_csv(os.path.join(subdir_path, file), parse_dates=['dt'],
                            date_format='%Y%m%d%H')
            dfs.append(c)
        df = pd.concat(dfs)
        df = df.drop_duplicates(subset='dt', keep='first')
        df = df.set_index('dt').sort_index()
        df = df.drop(columns=['fid', 'time_bnds'])

        missing = len(expected_index_) - df.shape[0]
        if missing > 15:
            counts, missing_list = {}, []
            missing_idx = [i for i in expected_index_ if i not in df.index]
            for midx in missing_idx:
                dt = f'{midx.year}{midx.month:02}'
                if dt not in counts.keys():
                    counts[dt] = 1
                else:
                    counts[dt] += 1
                p = f'NLDAS_FORA0125_H.A{midx.year}{midx.month:02}{midx.day:02}.{midx.hour}00.020.nc'
                f = os.path.join('/data/ssd1/nldas2/netcdf', p)
                if os.path.exists(f):
                    missing_list.append(1)

            print(f'{subdir_} is missing {missing} rows')
            # [print(k, v) for k, v in counts.items()]

            counts = {k: v for k, v in counts.items() if v > 1}

            if write_missing:
                with open(write_missing, 'w') as fp:
                    json.dump({'missing': list(counts.keys())}, fp, indent=4)
                print(f'wrote missing dates to {write_missing}, exiting')
                exit()
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
        if os.path.exists(out_file):
            print(f'{os.path.basename(out_file)} exists, skipping')
        else:
            print(f'{subdir_} not found')
        return


def gridmet_parquet(root_, subdir_, required_years_, expected_index_, strdt_, outdir_):
    subdir_path = os.path.join(root_, subdir_)
    out_file = os.path.join(outdir_, f'{subdir_}.parquet.gzip')

    if os.path.isdir(subdir_path):

        csv_files_ = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]

        # if os.path.exists(out_file) and csv_files_:
        #     shutil.rmtree(subdir_path)
        #     print(f'{os.path.basename(out_file)} exists, removing {len(csv_files)} csv files')
        #     return

        dtimes = [f.split('_')[-1].replace('.csv', '') for f in csv_files_]
        rm_files = csv_files_.copy()

        if len(dtimes) < len(required_years_):
            missing = [m for m in required_years_ if m not in dtimes]
            if len(missing) > 0:
                print(f'{subdir_} missing {len(missing)} months: {np.random.choice(missing, size=5, replace=False)}')
                return

        dfs = []
        for file in csv_files_:
            c = pd.read_csv(os.path.join(subdir_path, file), parse_dates=['dt'],
                            date_format='%Y%m%d%H')
            dfs.append(c)
        df = pd.concat(dfs)
        df = df.drop_duplicates(subset='dt', keep='first')
        df = df.set_index('dt').sort_index()

        missing = len(expected_index_) - df.shape[0]
        if missing > 15:
            print(f'{subdir_} is missing {missing} records')

        df['dt'] = strdt_

        df.to_parquet(out_file, compression='gzip')
        shutil.rmtree(subdir_path)
        print(f'wrote {subdir_}, removed {len(rm_files)} .csv files,'
              f' {datetime.strftime(datetime.now(), '%Y%m%d %H:%M')}')
        return
    else:
        if os.path.exists(out_file):
            print(f'{os.path.basename(out_file)} exists, skipping')
        else:
            print(f'{subdir_} not found')
        return

if __name__ == '__main__':

    d = '/media/research/IrrigationGIS'
    home = os.path.expanduser('~')

    if not os.path.isdir(d):
        d = os.path.join(home, 'data', 'IrrigationGIS')

    if not os.path.isdir(d):
        d = os.path.join('/data', 'IrrigationGIS')

    sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_29OCT2024.csv')
    # sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')

    source = 'gridmet'

    if source == 'gridmet':
        csv_files = '/data/gridmet/station_data'
        p_files = '/data/gridmet/parquet'

    elif source == 'nldas2':
        csv_files = '/data/ssd1/nldas2/station_data/'
        p_files = os.path.join(d, 'dads', 'met', 'gridded', 'nldas2_parquet')

    process_and_concat_csv(sites, csv_files, source, start_date='1992-01-01', end_date='2023-12-31', outdir=p_files,
                           workers=1, missing_file=None)

# ========================= EOF ====================================================================
