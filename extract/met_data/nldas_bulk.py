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


def get_nldas(start_date, end_date, down_dst=None):
    results = earthaccess.search_data(
        doi='10.5067/THUF4J1RLSYG',
        temporal=(start_date, end_date))
    if down_dst:
        earthaccess.download(results, down_dst)
    else:
        return results


def extract_nldas(stations, out_data, nc_data=None, workers=8, overwrite=False, bounds=None, debug=False,
                  parquet_check=None):
    station_list = pd.read_csv(stations)
    if 'LAT' in station_list.columns:
        station_list = station_list.rename(columns={'STAID': 'fid', 'LAT': 'latitude', 'LON': 'longitude'})
    station_list.index = station_list['fid']

    if bounds:
        w, s, e, n = bounds
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]
    else:
        ln = station_list.shape[0]
        w, s, e, n = (-125.0, 25.0, -67.0, 53.0)
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]
        print('dropped {} stations outside NLDAS-2 extent'.format(ln - station_list.shape[0]))

    station_list = station_list.head(n=100)
    print(f'{len(station_list)} stations to write')

    station_list = station_list.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    indexer = station_list[['lat', 'lon']].to_xarray()
    fids = np.unique(indexer.fid.values).tolist()

    yrmo, files = [], []

    for year in range(2023, 2024):

        for month in range(1, 13):

            month_start = datetime(year, month, 1)
            date_string = month_start.strftime('%Y%m')

            if nc_data:
                nc_files = [f for f in os.listdir(nc_data) if date_string == f.split('.')[1][1:7]]
                nc_files.sort()
                nc_files = [os.path.join(nc_data, f) for f in nc_files]
            else:
                m_str = str(month).rjust(2, '0')
                month_end = calendar.monthrange(year, month)[1]
                nc_files = get_nldas(f'{year}-{m_str}-01', f'{year}-{m_str}-{month_end}')

            if not nc_files:
                print(f'No NetCDF files found for {year}-{month}')
                continue

            yrmo.append(date_string)
            files.append(nc_files)

    print(f'{len(yrmo)} months to write')
    file_packs = [(yrmo[i], len(files[i])) for i in range(0, len(yrmo))]
    file_packs.reverse()
    [print(fp) for fp in file_packs]

    if debug:
        for fileset, dts in zip(files, yrmo):
            proc_time_slice(fileset, indexer, dts, fids, out_data, overwrite, parquet_check)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(proc_time_slice, fileset, indexer.copy(), dts, fids, out_data, overwrite)
                   for fileset, dts in zip(files, yrmo)]
        concurrent.futures.wait(futures)


def proc_time_slice(nc_files_, indexer_, date_string_, fids_, out_, overwrite_, par_check=None):
    """"""
    try:
        if isinstance(nc_files_[0], DataGranule):
            tmpdir = tempfile.gettempdir()
            ges_files = earthaccess.download(nc_files_, tmpdir, threads=4)
            # [print(f) for f in ges_files]
            ds = xr.open_mfdataset(ges_files, engine='netcdf4')
        else:
            ds = xr.open_mfdataset(nc_files_, engine='netcdf4')

        ds = ds.chunk({'time': len(nc_files_), 'lat': 28, 'lon': 29})

    except Exception as exc:
        print(f'{exc} on {date_string_}')
        return
    try:
        ds = ds.sel(lat=indexer_.lat, lon=indexer_.lon, method='nearest')
        df_all = ds.to_dataframe()
        ct, skip = 0, 0
        print(f'prepare to write {date_string_}: {datetime.strftime(datetime.now(), '%Y%m%d %H:%M')}')
    except Exception as exc:
        print(f'{exc} on {date_string_}')
        return

    for fid in fids_:

        try:
            if par_check:
                parquet = os.path.join(par_check, f'{fid}.parquet.gzip')
                if os.path.exists(parquet):
                    skip += 1
                    continue

            dst_dir = os.path.join(out_, fid)
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)

            _file = os.path.join(dst_dir, '{}_{}.csv'.format(fid, date_string_))
            if os.path.exists(_file.replace('ssd2', 'ssd1')):
                continue

            if os.path.exists(_file) and os.path.getsize(_file) == 0:
                os.remove(_file)

            if not os.path.exists(_file) or overwrite_:
                df_station = df_all.loc[(slice(None), 0, fid)].copy()
                df_station['dt'] = [i.strftime('%Y%m%d%H') for i in df_station.index]
                df_station.to_csv(_file, index=False)
                ct += 1
                if ct % 1000 == 0.:
                    print(f'{ct} of {len(fids_)} for {date_string_}')
            else:
                skip += 1
        except Exception as exc:
            print(f'{exc} on {fid}')
            return
    del ds, df_all
    gc.collect()
    print(f'wrote {ct} for {date_string_}, skipped {skip}, {datetime.strftime(datetime.now(), '%Y%m%d %H:%M')}')


def process_and_concat_csv(stations, root, start_date, end_date, outdir, workers, alt_dir):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    required_months = pd.date_range(start=start, end=end, freq='MS').strftime('%Y%m').tolist()
    expected_index = pd.date_range(start=start, end=end, freq='h')
    strdt = [d.strftime('%Y%m%d%H') for d in expected_index]

    station_list = pd.read_csv(stations)
    if 'LAT' in station_list.columns:
        station_list = station_list.rename(columns={'STAID': 'fid', 'LAT': 'latitude', 'LON': 'longitude'})

    station_list = station_list.sample(frac=1)
    subdirs = station_list['fid'].to_list()

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        executor.map(process_parquet, [root] * len(subdirs), subdirs,
                     [required_months] * len(subdirs),
                     [expected_index] * len(subdirs), [strdt] * len(subdirs),
                     [outdir] * len(subdirs), [alt_dir] * len(subdirs))


def process_parquet(root_, subdir_, required_months_, expected_index_, strdt_, outdir_, alt_dir_):
    subdir_path = os.path.join(root_, subdir_)
    if os.path.isdir(subdir_path):

        csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
        dtimes = [f.split('_')[-1].replace('.csv', '') for f in csv_files]
        rm_files = csv_files.copy()

        if len(dtimes) < len(required_months_):

            alt_subdir = os.path.join(alt_dir_, subdir_)
            if not os.path.exists(alt_subdir):
                return
            alt_csv_files = [os.path.join(alt_subdir, f) for f in os.listdir(alt_subdir) if f.endswith('.csv')]
            alt_dtimes = [f.split('_')[-1].replace('.csv', '') for f in alt_csv_files]

            for acf, adt in zip(alt_csv_files, alt_dtimes):
                if adt in required_months_ and adt not in dtimes:
                    csv_files.append(acf)
                    dtimes.append(adt)

                rm_files.append(acf)

            missing = [m for m in required_months_ if m not in dtimes]
            if len(missing) > 0:
                print(f'{subdir_} missing {len(missing)} months: {missing}')
                return
        else:
            alt_subdir = None

        dfs = []
        for file in csv_files:
            c = pd.read_csv(os.path.join(subdir_path, file), parse_dates=['dt'],
                            date_format='%Y%m%d%H')
            dfs.append(c)
        df = pd.concat(dfs)

        df = df.set_index('dt').sort_index()
        df = df.drop(columns=['fid', 'time_bnds'])

        missing = len(expected_index_) - df.shape[0]
        if missing > 100:
            print(f'{subdir_} is missing {missing} rows')
            return

        elif missing > 0:
            df = df.reindex(expected_index_)
            df = df.interpolate(method='linear')

        df['dt'] = strdt_
        out_file = os.path.join(outdir_, f'{subdir_}.parquet.gzip')
        df.to_parquet(out_file, compression='gzip')
        shutil.rmtree(subdir_path)
        if alt_subdir:
            shutil.rmtree(alt_subdir)
        print(f'wrote {subdir_}, removed {len(rm_files)} .csv files,'
              f' {datetime.strftime(datetime.now(), '%Y%m%d %H:%M')}')
        return
    else:
        print(f'{subdir_} not found')
        return


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'

    if not os.path.isdir(d):
        d = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS')

    if not os.path.isdir(d):
        d = os.path.join('/home', 'ec2-user', 'data', 'IrrigationGIS')

    if not os.path.isdir(d):
        d = os.path.join('/home', 'dketchum', 'data', 'IrrigationGIS')

    # nc_data_ = '/data/ssd1/nldas2/netcdf'
    nc_data_ = None
    # get_nldas(nc_data_)

    sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    # sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_29OCT2024.csv')

    # csv_files = '/data/ssd2/nldas2/station_data/'
    csv_files = os.path.join(d, 'dads', 'met', 'gridded', 'nldas2', 'station_data')

    p_files = os.path.join(d, 'dads', 'met', 'gridded', 'nldas2_parquet')

    if not nc_data_:
        earthaccess.login()
        print('earthdata access authenticated')
    
    extract_nldas(sites, csv_files, nc_data=None, workers=16, overwrite=False, bounds=None,
                  debug=False, parquet_check=p_files)

    # process_and_concat_csv(sites, csv_files, start_date='1990-01-01', end_date='2023-12-31', outdir=p_files,
    #                        workers=10, alt_dir='/data/ssd1/nldas2/station_data/')

# ========================= EOF ====================================================================
