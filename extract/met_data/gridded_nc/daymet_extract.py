import calendar
import concurrent.futures
import gc
import os
import tempfile
import json
from datetime import datetime
import concurrent.futures

import earthaccess
import numpy as np
import pandas as pd
import xarray as xr
from earthaccess.results import DataGranule


def get_daymet(start_date, end_date, down_dst=None):
    print('earthdata access authenticated')
    results = earthaccess.search_data(
        doi='10.3334/ORNLDAAC/2129',
        temporal=(start_date, end_date))
    if down_dst:
        earthaccess.download(results, down_dst)
    else:
        return results


def extract_daymet(stations, out_data, nc_data=None, workers=8, overwrite=False, bounds=None, debug=False,
                   parquet_check=None, missing_list=None, tmpd=None):
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
        print('dropped {} stations outside DAYMET extent'.format(ln - station_list.shape[0]))

    print(f'{len(station_list)} stations to write')

    station_list = station_list.sample(frac=1)
    station_list = station_list.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    indexer = station_list[['lat', 'lon']].to_xarray()
    fids = np.unique(indexer.fid.values).tolist()

    yrmo, files = [], []

    for year in range(1990, 2024):
        nc_files = []
        all_files = get_daymet(f'{year}-01-01', f'{year}-12-31')
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
    [print(fp) for fp in file_packs]

    if debug:
        for fileset, dts in zip(files, yrmo):
            proc_time_slice(fileset, indexer, dts, fids, out_data, overwrite, par_check=parquet_check, tmpdir=tmpd)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(proc_time_slice, fileset, indexer, dts, fids, out_data, overwrite, tmpdir=tmpd)
                   for fileset, dts in zip(files, yrmo)]
        concurrent.futures.wait(futures)


def proc_time_slice(nc_files_, indexer_, date_string_, fids_, out_, overwrite_, par_check=None, tmpdir=None):
    try:
        if not tmpdir:
            tmpdir = tempfile.gettempdir()
        ges_files = earthaccess.download(nc_files_, tmpdir, threads=4)
        ds = xr.open_mfdataset(ges_files, engine='netcdf4')
        [os.remove(f) for f in ges_files]
    except Exception as exc:
        return

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


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS')

    earthaccess.login()
    print('earthdata access authenticated')

    # sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_29OCT2024.csv')
    out_files = '/data/ssd2/daymet/station_data/'

    # extract_daymet(sites, nc_data_, out_files, workers=20, overwrite=False, bounds=None, debug=False)

# ========================= EOF ====================================================================
