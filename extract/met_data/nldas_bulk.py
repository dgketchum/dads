import concurrent.futures
import json
import os
from datetime import datetime

import earthaccess
import numpy as np
import pandas as pd
import xarray as xr

def get_nldas(dst):
    earthaccess.login()
    print('earthdata access authenticated')
    results = earthaccess.search_data(
        doi='10.5067/THUF4J1RLSYG',
        temporal=('1990-01-01', '2023-12-31'))
    earthaccess.download(results, dst)


def extract_nldas(stations, gridded_dir, incomplete_out, out_data, overwrite=False, bounds=None,
                  num_workers=1, index_col='fid', debug=False):
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

    start, end = datetime(2000, 1, 1), datetime(2024, 8, 1)

    station_list = station_list.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    indexer = station_list[['lat', 'lon']].to_xarray()

    for year in range(start.year, end.year + 1):
        for month in range(1, 13):

            month_start = datetime(year, month, 1)
            date_string = month_start.strftime('%Y%m')

            nc_files = sorted([f for f in os.listdir(gridded_dir) if date_string == f.split('.')[1][1:7]])
            if not nc_files:
                print(f"No NetCDF files found for {year}")
                continue

            datasets, complete = [], True
            for f in nc_files:
                try:
                    nc_file = os.path.join(gridded_dir, f)
                    ds = xr.open_dataset(nc_file)
                    if bounds:
                        datasets.append(ds.sel(lat=slice(s, n), lon=slice(w, e)))
                    else:
                        datasets.append(ds)

                except Exception as exc:
                    incomplete['missing'].append(f)
                    print(f"Unreadable NetCDF files found for {year}")
                    complete = False
                    break

            if not complete:
                continue

            ds = xr.concat(datasets, dim='time')
            ds = ds.sel(lat=indexer.lat, lon=indexer.lon, method='nearest')
            fids = np.unique(indexer.fid.values).tolist()

            if debug:
                for fid in fids:
                    process_fid(fid, year, month, ds, out_data, overwrite)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(process_fid, fid, year, ds,
                                               out_data, overwrite) for fid in fids]
                    concurrent.futures.wait(futures)

    if len(incomplete) > 0:
        with open(incomplete_out, 'w') as fp:
            json.dump(incomplete, fp, indent=4)


def process_fid(fid, year, month, ds, out_data, overwrite):
    dst_dir = os.path.join(out_data, fid)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    _file = os.path.join(dst_dir, '{}_{}_{}.csv'.format(fid, year, month))

    if not os.path.exists(_file) or overwrite:
        df_station = ds.sel(fid=fid).to_dataframe()
        df_station = df_station.groupby(df_station.index.get_level_values('time')).first()
        df_station['dt'] = [i.strftime('%Y%m%d%H') for i in df_station.index]
        df_station.to_csv(_file)
        print(_file)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS')

    nc_data = '/data/ssd1/nldas2'
    # get_nldas(nc_data)

    # sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_mgrs_28OCT2024.csv')
    grid_dir = os.path.join(d, 'dads', 'met', 'gridded', 'nldas2_monthly')
    incomp = os.path.join(d, 'dads', 'met', 'gridded', 'incomplete_nldas_files.json')

    extract_nldas(sites, nc_data, incomp, grid_dir, overwrite=False, bounds=None, num_workers=1, debug=True)
# ========================= EOF ====================================================================
