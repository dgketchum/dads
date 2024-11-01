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


def get_nldas(dst):
    earthaccess.login()
    print('earthdata access authenticated')
    results = earthaccess.search_data(
        doi='10.5067/THUF4J1RLSYG',
        temporal=('1990-01-01', '2023-12-31'))
    earthaccess.download(results, dst)


def extract_nldas(stations, nc_data, incomplete_out, out_data, workers=8, overwrite=False, bounds=None,
                  index_col='fid', debug=False):

    if os.path.exists(incomplete_out):
        with open(incomplete_out, 'r') as f:
            incomplete = json.load(f)
    else:
        incomplete = {'missing': []}

    station_list = pd.read_csv(stations, index_col=index_col)

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

    station_list = station_list.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    indexer = station_list[['lat', 'lon']].to_xarray()
    fids = np.unique(indexer.fid.values).tolist()

    for year in range(1990, 2024):

        for month in range(1, 13):

            month_start = datetime(year, month, 1)
            date_string = month_start.strftime('%Y%m')

            nc_files = [f for f in os.listdir(nc_data) if date_string == f.split('.')[1][1:7]]
            nc_files.sort()

            if not nc_files:
                print(f"No NetCDF files found for {year}-{month}")
                continue

            datasets, complete = [], True
            for f in nc_files[:49]:
                try:
                    nc_file = os.path.join(nc_data, f)
                    ds = xr.open_dataset(nc_file, engine='netcdf4', decode_cf=False)
                    datasets.append(ds)

                except Exception as exc:
                    incomplete['missing'].append(f)
                    print(f"Unreadable NetCDF files found for {year}-{month}")
                    complete = False
                    break

            if not complete:
                continue

            ds = xr.concat(datasets, dim='time')
            ds = ds.sel(lat=indexer.lat, lon=indexer.lon, method='nearest')
            time_values = pd.to_datetime(ds['time'].values, unit='h', origin=pd.Timestamp('1979-01-01'))
            ds = ds.assign_coords(time=time_values).set_index(time='time')

            if debug:
                for fid in fids:
                    process_fid(fid, ds, date_string, out_data, overwrite)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(process_fid, fid, ds, date_string, out_data, overwrite) for fid in fids]
                    concurrent.futures.wait(futures)

    if len(incomplete) > 0:
        with open(incomplete_out, 'w') as fp:
            json.dump(incomplete, fp, indent=4)

def process_fid(fid, ds, yearmo, out_data, overwrite):
    dst_dir = os.path.join(out_data, fid)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    _file = os.path.join(dst_dir, '{}_{}.csv'.format(fid, yearmo))

    if not os.path.exists(_file) or overwrite:
        df_station = ds.sel(fid=fid).to_dataframe()
        df_station = df_station.groupby(df_station.index.get_level_values('time')).first()
        df_station['dt'] = [i.strftime('%Y%m%d%H') for i in df_station.index]
        df_station.to_csv(_file, index=False)
        print(_file)


def main():

    d = '/media/research/IrrigationGIS'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS')

    nc_data_ = '/data/ssd1/nldas2'
    # get_nldas(nc_data)

    # sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_mgrs_28OCT2024.csv')
    grid_dir = '/data/ssd1/nldas_raw/'

    incomp = os.path.join(d, 'dads', 'met', 'gridded', 'incomplete_files_nldas.json')

    extract_nldas(sites, nc_data_, incomp, grid_dir, workers=20,
                  overwrite=False, bounds=None, debug=True)


if __name__ == '__main__':
    main()
# ========================= EOF ====================================================================
