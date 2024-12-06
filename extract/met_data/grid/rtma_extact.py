import calendar
import concurrent.futures
import gc
import os
import tempfile
from datetime import datetime, timedelta

import earthaccess
import numpy as np
import pandas as pd
import xarray as xr
from earthaccess.results import DataGranule

from extract.met_data.grid.to_parquet import process_and_concat_csv


def get_rtma(start_date, end_date, down_dst=None):
    # Construct the URL for the RTMA data
    base_url = "https://noaa-urma-pds.s3.amazonaws.com/urma2p5.{date}/urma2p5.t14z.2dvarges_ndfd.grb2_ext"
    date_str = start_date.strftime("%Y%m%d")
    url = base_url.format(date=date_str)

    if down_dst:
        # Download the file using earthaccess if down_dst is provided
        earthaccess.download(url, down_dst)
    else:
        # Return the URL if down_dst is not provided
        return url


def extract_rtma(stations, out_data, grb_data=None, workers=8, overwrite=False, bounds=None, debug=False,
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
        print('dropped {} stations outside RTMA extent'.format(ln - station_list.shape[0]))

    print(f'{len(station_list)} stations to write')

    station_list = station_list.sample(frac=1)
    station_list = station_list.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

    if 'END' in station_list.columns:
        station_list['end_dt'] = [pd.to_datetime(r['END']) for i, r in station_list.iterrows()]
        station_list = station_list[station_list['end_dt'] > pd.to_datetime('2016-01-01')]

    indexer = station_list[['lat', 'lon']].to_xarray()
    fids = np.unique(indexer.fid.values).tolist()

    yrmo, files = [], []

    for year in range(2016, 2024):

        for month in range(1, 13):

            month_start = datetime(year, month, 1)
            date_string = month_start.strftime('%Y%m')

            if missing_list and date_string not in missing_list:
                continue

            if grb_data:
                grb_files = [f for f in os.listdir(grb_data) if date_string == f.split('.')[1][1:7]]
                grb_files.sort()
                grb_files = [os.path.join(grb_data, f) for f in grb_files]
            else:
                m_str = str(month).rjust(2, '0')
                month_end = calendar.monthrange(year, month)[1]
                grb_files = []
                for day in range(1, month_end + 1):
                    start_date = datetime(year, month, day)
                    grb_files.append(get_rtma(start_date, start_date))

            if not grb_files:
                print(f'No GRIB files found for {year}-{month}')
                continue

            yrmo.append(date_string)
            files.append(grb_files)

    print(f'{len(yrmo)} months to write')
    file_packs = [(yrmo[i], len(files[i])) for i in range(0, len(yrmo))]
    [print(fp) for fp in file_packs]

    if debug:
        for fileset, dts in zip(files, yrmo):
            proc_time_slice(fileset, indexer, dts, fids, out_data, overwrite, par_check=parquet_check, tmpdir=tmpd)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(proc_time_slice, fileset, indexer, dts, fids, out_data, overwrite, tmpdir=tmpd)
                   for fileset, dts in zip(files, yrmo)]
        concurrent.futures.wait(futures)


def proc_time_slice(grb_files_, indexer_, date_string_, fids_, out_, overwrite_, par_check=None, tmpdir=None):
    try:
        if isinstance(grb_files_[0], DataGranule):
            if not tmpdir:
                tmpdir = tempfile.gettempdir()
            ges_files = earthaccess.download(grb_files_, tmpdir, threads=4)
            ds = xr.open_mfdataset(ges_files, engine='cfgrib')
            [os.remove(f) for f in ges_files]
        else:
            ds = xr.open_mfdataset(grb_files_, engine='cfgrib', combine='nested', concat_dim='time')

        ds = ds.chunk({'time': len(grb_files_), 'latitude': 28, 'longitude': 29})

    except Exception as exc:
        print(f'{exc} on {date_string_}')
        return
    try:
        ds = ds.sel(latitude=indexer_.lat, longitude=indexer_.lon, method='nearest')
        df_all = ds.to_dataframe()
        ct, skip = 0, 0
        now = datetime.strftime(datetime.now(), '%Y%m%d %H:%M')

        print(f'prepare to write {date_string_}: {now}')
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
                df_station = df_all.loc[(slice(None), fid)].copy()
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
    now = datetime.strftime(datetime.now(), '%Y%m%d %H:%M')
    print(f'wrote {ct} for {date_string_}, skipped {skip}, {now}')


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
    temp_directory = None

    if not os.path.isdir(d):
        d = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS')

    if not os.path.isdir(d):
        d = os.path.join('/data', 'IrrigationGIS')
        temp_directory = os.path.join('/data', 'temp')

    if not os.path.isdir(d):
        d = os.path.join('/home', 'dketchum', 'data', 'IrrigationGIS')

    grb_data_ = '/data/ssd2/rtma/grb'
    # grb_data_ = None
    # get_rtma(grb_data_)

    sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    # sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_29OCT2024.csv')

    csv_files = '/data/ssd1/rtma/station_data/'
    # csv_files = os.path.join(d, 'dads', 'met', 'gridded', 'rtma', 'station_data')
    p_files = os.path.join(d, 'dads', 'met', 'gridded', 'rtma_parquet')

    if not grb_data_:
        earthaccess.login()
        print('earthdata access authenticated')

    bounds = (-125.0, 25.0, -67.0, 53.0)
    quadrants = get_quadrants(bounds)

    for e, quad in enumerate(quadrants, start=1):

        print(f'\n\n\n\n Quadrant {e} \n\n\n\n')

        extract_rtma(sites, csv_files, grb_data=grb_data_, workers=16, overwrite=False, missing_list=None,
                     bounds=quad, debug=False, parquet_check=p_files)

        process_and_concat_csv(sites, csv_files, start_date='2016-01-01', end_date='2023-12-31', outdir=p_files,
                               workers=16, missing_file=None)

# ========================= EOF ====================================================================
