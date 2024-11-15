import os
import json
from datetime import datetime
import concurrent.futures

import dask
import earthaccess
import pandas as pd
import numpy as np
import xarray as xr
from dask.distributed import Client


def get_daymet(dst):
    earthaccess.login()
    print('earthdata access authenticated')
    results = earthaccess.search_data(
        doi='10.3334/ORNLDAAC/2129',
        temporal=('1990-01-01', '1990-01-03'))
    earthaccess.download(results, dst)


def extract_daymet(stations, nc_data, out_data, workers=8, overwrite=False, bounds=None,
                   debug=False):
    station_list = pd.read_csv(stations)
    if 'LAT' in station_list.columns:
        station_list = station_list.rename(columns={'STAID': 'fid', 'LAT': 'latitude', 'LON': 'longitude'})
    station_list.index = station_list['fid']

    if bounds:
        w, s, e, n = bounds
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]

    station_list = station_list.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    indexer = station_list[['lat', 'lon']].to_xarray()
    fids = np.unique(indexer.fid.values).tolist()

    yrmo, files = [], []

    for year in range(1990, 1991):

        all_files = [f for f in os.listdir(nc_data) if f.split('.')[-2][-4:] == str(year)]
        all_files.sort()
        nc_files = []
        for file in all_files:
            split = file.split('_')
            region, param = split[3], split[4]
            if param in ['tmax', 'tmin', 'vp', 'prcp', 'srad'] and region == 'na':
                nc_files.append(os.path.join(nc_data, file))

        if not nc_files:
            print(f"No NetCDF files found for {year}")
            continue

        yrmo.append(year)
        files.append(nc_files)

    print(f'{len(yrmo)} years to write')
    file_packs = [(yrmo[i], len(files[i])) for i in range(0, len(yrmo))]

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
            return

    ds = xr.concat(datasets, dim='time')
    ds = ds.sel(lat=indexer_.lat, lon=indexer_.lon, method='nearest')
    time_values = pd.to_datetime(ds['time'].values, unit='h', origin=pd.Timestamp('1979-01-01'))
    ds = ds.assign_coords(time=time_values).set_index(time='time')
    ct = 0
    for fid in fids_:
        dst_dir = os.path.join(out_, fid)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        _file = os.path.join(dst_dir, '{}_{}.csv'.format(fid, date_string_))

        if not os.path.exists(_file) or overwrite_:
            df_station = ds.sel(fid=fid).to_dataframe()
            df_station = df_station.groupby(df_station.index.get_level_values('time')).first()
            df_station['dt'] = [i.strftime('%Y%m%d%H') for i in df_station.index]
            df_station.to_csv(_file, index=False)
            ct += 1
    print(f'wrote {ct} for {date_string_}')


def main():
    d = '/media/research/IrrigationGIS'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS')

    nc_data_ = '/data/ssd2/daymet/netcdf'
    # nc_data_ = '/home/dgketchum/Downloads/daymet/netcdf'
    get_daymet(nc_data_)

    # sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_29OCT2024.csv')
    out_files = '/data/ssd2/daymet/station_data/'

    # extract_daymet(sites, nc_data_, out_files, workers=20, overwrite=False, bounds=None, debug=False)


if __name__ == '__main__':
    main()
# ========================= EOF ====================================================================