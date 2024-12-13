import concurrent.futures
import os
import json
import shutil
from datetime import datetime

import pandas as pd
import numpy as np


def process_and_concat_csv(root, source, start_date, end_date, outdir, workers, missing_file=None,
                           debug=False):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    required_months = pd.date_range(start=start, end=end, freq='MS').strftime('%Y%m').tolist()
    required_years = pd.date_range(start=start, end=end, freq='YE').strftime('%Y').tolist()

    if source in ['gridmet', 'prism', 'daymet']:
        expected_index = pd.date_range(start=start, end=end, freq='d')
        str_dtime = [d.strftime('%Y%m%d') for d in expected_index]
    elif source == 'nldas2':
        expected_index = pd.date_range(start=start, end=end, freq='h')
        str_dtime = [d.strftime('%Y%m%d%H') for d in expected_index]
    else:
        raise ValueError('Unknown source')

    subdirs = list(os.listdir(root))
    print(f'{len(subdirs)} directories to check')

    if missing_file:
        for sd in subdirs:
            if source == 'nldas2':
                nldas2_parquet(root, sd, required_months, expected_index, str_dtime, outdir, missing_file)

    if debug:
        for sd in subdirs:
            if source == 'nldas2':
                nldas2_parquet(root, sd, required_months, expected_index, str_dtime, outdir)
            elif source in ['gridmet', 'prism', 'daymet']:
                general_parquet(root, sd, required_years, str_dtime, outdir)
            else:
                raise ValueError(f"Invalid source: {source}")
            print(sd)

    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            if source == 'nldas2':
                executor.map(nldas2_parquet, [root] * len(subdirs), subdirs,
                             [required_months] * len(subdirs),
                             [expected_index] * len(subdirs), [str_dtime] * len(subdirs),
                             [outdir] * len(subdirs))

            if source in ['gridmet', 'prism', 'daymet']:
                executor.map(general_parquet, [root] * len(subdirs), subdirs,
                             [required_years] * len(subdirs), [str_dtime] * len(subdirs),
                             [outdir] * len(subdirs))


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


def general_parquet(root_, subdir_, required_years_, expected_index_, outdir_):
    """Use for PRISM, Daymet, and GridMET"""
    subdir_path = os.path.join(root_, subdir_)
    out_file = os.path.join(outdir_, f'{subdir_}.parquet.gzip')

    if os.path.isdir(subdir_path):
        csv_files_ = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
        if os.path.exists(out_file) and csv_files_:
            shutil.rmtree(subdir_path)
            print(f'{os.path.basename(out_file)} exists, removing {len(csv_files)} csv files')
            return

        dtimes = [f.split('_')[-1].replace('.csv', '') for f in csv_files_]

        if len(dtimes) < len(required_years_):
            missing = [m for m in required_years_ if m not in dtimes]
            if len(missing) > 0:
                if missing >= 5:
                    print(f'{subdir_} missing {len(missing)} years: {np.random.choice(missing, size=5, replace=False)}')
                else:
                    print(f'{subdir_} missing {len(missing)} years: {missing}')
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

        if missing > 10:
            print(f'{subdir_} is missing {missing} records')
            print([i for i in expected_index_ if i not in df.index])
            return

        df['dt'] = df.index
        df.to_parquet(out_file, compression='gzip')
        shutil.rmtree(subdir_path)

    else:
        if os.path.exists(out_file):
            print(f'{os.path.basename(out_file)} exists, skipping')
        else:
            print(f'{subdir_} not found')


if __name__ == '__main__':

    home = os.path.expanduser('~')
    r = os.path.join('/data')

    source_ = 'daymet'

    if source_ in ['gridmet', 'prism', 'daymet']:
        csv_files = os.path.join(r, source_, 'station_data')
        p_files = os.path.join(r, source_, 'parquet')

    elif source_ == 'nldas2':
        csv_files = '/data/ssd1/nldas2/station_data/'
        d = os.path.join(home, 'data', 'IrrigationGIS')
        p_files = os.path.join(d, 'dads', 'met', 'gridded', 'nldas2_parquet')

    else:
        raise ValueError

    print(f'{csv_files} exists: {os.path.exists(csv_files)}')
    print(f'{p_files} exists: {os.path.exists(p_files)}')
    process_and_concat_csv(csv_files, source_, start_date='1990-01-01', end_date='2023-12-31', outdir=p_files,
                           workers=10, missing_file=None, debug=False)

# ========================= EOF ====================================================================
