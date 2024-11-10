import os
import concurrent.futures
import shutil
from datetime import datetime
import os
import shutil
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

import earthaccess
import numpy as np
import pandas as pd
import xarray as xr


def get_nldas(dst):
    earthaccess.login()
    print('earthdata access authenticated')
    results = earthaccess.search_data(
        doi='10.5067/THUF4J1RLSYG',
        temporal=('2003-01-01', '2003-06-23'))
    earthaccess.download(results, dst)


def extract_nldas(stations, nc_data, out_data, workers=8, overwrite=False, bounds=None,
                  debug=False):
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

    print(f'{len(station_list)} stations to write')

    station_list = station_list.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    indexer = station_list[['lat', 'lon']].to_xarray()
    fids = np.unique(indexer.fid.values).tolist()

    yrmo, files = [], []

    for year in range(2010, 2016):

        for month in range(1, 13):

            month_start = datetime(year, month, 1)
            date_string = month_start.strftime('%Y%m')

            nc_files = [f for f in os.listdir(nc_data) if date_string == f.split('.')[1][1:7]]
            nc_files.sort()
            nc_files = [os.path.join(nc_data, f) for f in nc_files]

            if not nc_files:
                print(f'No NetCDF files found for {year}-{month}')
                continue

            yrmo.append(date_string)
            files.append(nc_files)

    print(f'{len(yrmo)} months to write')
    file_packs = [(yrmo[i], len(files[i])) for i in range(0, len(yrmo))]
    [print(fp) for fp in file_packs]

    if debug:
        for fileset, dts in zip(files, yrmo):
            proc_time_slice(fileset, indexer.copy(), dts, fids, out_data, overwrite)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(proc_time_slice, fileset, indexer.copy(), dts, fids, out_data, overwrite)
                   for fileset, dts in zip(files, yrmo)]
        concurrent.futures.wait(futures)


def proc_time_slice(nc_files_, indexer_, date_string_, fids_, out_, overwrite_):
    datasets, complete = [], True
    for f in nc_files_:
        try:
            ds = xr.open_dataset(f, engine='netcdf4', decode_cf=False)
            datasets.append(ds)
        except Exception as exc:
            if f == '/data/ssd1/nldas2/netcdf/NLDAS_FORA0125_H.A20030622.0900.020.nc':
                continue
            else:
                print(f'{exc} on {f}')
                return

    ds = xr.concat(datasets, dim='time')
    ds = ds.sel(lat=indexer_.lat, lon=indexer_.lon, method='nearest')
    time_values = pd.to_datetime(ds['time'].values, unit='h', origin=pd.Timestamp('1979-01-01'))
    ds = ds.assign_coords(time=time_values).set_index(time='time')
    ct, skip = 0, 0
    print(f'prepare to write {date_string_}')
    for fid in fids_:
        dst_dir = os.path.join(out_, fid)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        _file = os.path.join(dst_dir, '{}_{}.csv'.format(fid, date_string_))

        if os.path.exists(_file) and os.path.getsize(_file) == 0:
            os.remove(_file)

        if not os.path.exists(_file) or overwrite_:
            df_station = ds.sel(fid=fid).to_dataframe()
            df_station = df_station.groupby(df_station.index.get_level_values('time')).first()
            df_station['dt'] = [i.strftime('%Y%m%d%H') for i in df_station.index]
            df_station.to_csv(_file, index=False)
            ct += 1
        else:
            skip += 1
    print(f'wrote {ct} for {date_string_}, skipped {skip}')


def process_and_concat_csv(stations, root, start_date, end_date, outdir, workers):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    required_months = pd.date_range(start=start, end=end, freq='MS').strftime('%Y%m').tolist()
    expected_index = pd.date_range(start=start, end=end, freq='h')
    strdt = [d.strftime('%Y%m%d%H') for d in expected_index]

    station_list = pd.read_csv(stations)
    if 'LAT' in station_list.columns:
        station_list = station_list.rename(columns={'STAID': 'fid', 'LAT': 'latitude', 'LON': 'longitude'})
    subdirs = sorted(station_list['fid'].to_list())

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        executor.map(process_parquet, [root] * len(subdirs), subdirs,
                     [required_months] * len(subdirs),
                     [expected_index] * len(subdirs), [strdt] * len(subdirs),
                     [outdir] * len(subdirs))


def process_parquet(root_, subdir_, required_months_, expected_index_, strdt_, outdir_):
    subdir_path = os.path.join(root_, subdir_)
    if os.path.isdir(subdir_path):
        csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
        dtimes = [f.split('_')[-1].replace('.csv', '') for f in csv_files]

        missing = [m for m in required_months_ if m not in dtimes]
        if len(missing) > 1:
            print(f'{subdir_} missing {len(missing)} months')
            return

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
        [os.remove(os.path.join(subdir_path, file)) for file in csv_files]
        shutil.rmtree(subdir_path)
        print(f'wrote {subdir_}')


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.isdir(d):
        d = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS')

    nc_data_ = '/data/ssd1/nldas2/netcdf'
    # get_nldas(nc_data_)

    # sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_29OCT2024.csv')
    csv_files = '/data/ssd2/nldas2/station_data/'
    p_files = '/data/ssd2/nldas2/parquet/'

    # extract_nldas(sites, nc_data_, csv_files, workers=3, overwrite=True, bounds=None, debug=False)
    process_and_concat_csv(sites, csv_files, start_date='1990-01-01', end_date='2023-12-31', outdir=p_files,
                           workers=40)

# ========================= EOF ====================================================================
