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

    print(f'{len(station_list)} stations to write')

    station_list = station_list.sample(frac=1)
    station_list = station_list.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

    # exist = [sbd for sbd in os.listdir(out_data) if os.path.exists(os.path.join(out_data, sbd))]

    indexer = station_list[['lat', 'lon']].to_xarray()
    fids = np.unique(indexer.fid.values).tolist()

    yrmo, files = [], []

    for year in range(2007, 2008):

        for month in range(6, 7):

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
    [print(fp) for fp in file_packs]

    if debug:
        for fileset, dts in zip(files, yrmo):
            proc_time_slice(fileset, indexer, dts, fids, out_data, overwrite, parquet_check)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(proc_time_slice, fileset, indexer, dts, fids, out_data, overwrite)
                   for fileset, dts in zip(files, yrmo)]
        concurrent.futures.wait(futures)


def proc_time_slice(nc_files_, indexer_, date_string_, fids_, out_, overwrite_, par_check=None):
    """"""
    try:
        if isinstance(nc_files_[0], DataGranule):
            tmpdir = tempfile.gettempdir()
            ges_files = earthaccess.download(nc_files_, tmpdir, threads=4)
            ds = xr.open_mfdataset(ges_files, engine='netcdf4')
            [os.remove(f) for f in ges_files]
        else:
            # could not find a way to get a good file for NLDAS_FORA0125_H.A20030622.0900.020.nc
            if date_string_ == '200306':
                dataset = []
                for f in nc_files_:
                    try:
                        sds = xr.open_dataset(f, engine='netcdf4')
                    except Exception as exc:
                        print(exc, os.path.basename(f))
                        pass
                    dataset.append(sds)
                ds = xr.concat(dataset, dim='time')
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

    nc_data_ = '/data/ssd1/nldas2/netcdf'
    # nc_data_ = None
    # get_nldas(nc_data_)

    # sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_29OCT2024.csv')

    csv_files = '/data/ssd1/nldas2/station_data/'
    # csv_files = os.path.join(d, 'dads', 'met', 'gridded', 'nldas2', 'station_data')

    if not nc_data_:
        earthaccess.login()
        print('earthdata access authenticated')

    bounds = (-125.0, 25.0, -67.0, 53.0)
    quadrants = get_quadrants(bounds)

    # for e, quad in enumerate(quadrants, start=1):

    # print(f'\n\n\n\n Quadrant {e} \n\n\n\n')
    # extract_nldas(sites, csv_files, nc_data=nc_data_, workers=1, overwrite=False, bounds=bounds,
    #               debug=False, parquet_check=p_files)
# ========================= EOF ====================================================================
