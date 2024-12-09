import calendar
import concurrent.futures
import gc
import os
from datetime import datetime, date

import boto3
import numpy as np
import pandas as pd
import xarray as xr
from botocore.exceptions import ClientError
from botocore import UNSIGNED
from botocore.client import Config
from dateutil.rrule import rrule, HOURLY


def extract_rtma(stations, out_data, grb_data=None, workers=8, overwrite=False, bounds=None, debug=False,
                 start_yr=1990, end_yr=2024):
    station_list = pd.read_csv(stations, nrows=100)
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

    yrmo = [(datetime(year, month, 1).strftime('%Y%m')) for year in range(start_yr, end_yr) for month in range(1, 13)]
    yrmo = [yms for yms in yrmo if int(yms) > 201401][:1]

    print(f'{len(yrmo)} months to write')

    if debug:
        for dts in yrmo:
            proc_time_slice(indexer, dts, fids, out_data, overwrite, grb_dir=grb_data)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(proc_time_slice, indexer, dts, fids, out_data, overwrite, tmpdir=grb_data)
                   for dts in zip(yrmo)]
        concurrent.futures.wait(futures)


def proc_time_slice(indexer_, date_string_, fids_, out_, overwrite_, grb_dir=None):
    """"""
    grb_files = get_grb_files(date_string_, dst=grb_dir, overwrite=overwrite_)
    ds = xr.open_mfdataset(grb_files, engine='cfgrib', concat_dim='time', combine='nested')
    ds = ds.chunk({'time': len(grb_files), 'latitude': 28, 'longitude': 29})

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


def get_grb_files(yrmo_str, dst, overwrite=False):
    """"""
    year, month = int(yrmo_str[:4]), int(yrmo_str[-2:])
    bucket_name = 'noaa-urma-pds'
    date_dir, hr_file = 'urma2p5.{date}', 'urma2p5.t{hour}z.2dvaranl_ndfd.grb2{ext}'
    start_date = date(year, month, 1)
    month_end = calendar.monthrange(year, month)[1]
    end_date = date(year, month, month_end)
    files = []
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    for dt in rrule(HOURLY, dtstart=start_date, until=end_date):
        date_str = dt.strftime('%Y%m%d')
        hr_str = dt.strftime('%H')

        day_dir = date_dir.format(date=date_str)
        par_dir = os.path.join(dst, day_dir)
        if not os.path.exists(par_dir):
            os.mkdir(par_dir)

        file_path = os.path.join(par_dir, hr_file.format(hour=hr_str, ext=''))
        files.append(file_path)
        if os.path.exists(file_path) and not overwrite:
            continue

        object_key = f'{date_dir.format(date=date_str)}/{hr_file.format(hour=hr_str, ext='_wexp')}'
        try:
            s3.download_file(bucket_name, object_key, file_path)

        except ClientError as exc:
            try:
                object_key = f'{date_dir.format(date=date_str)}/{hr_file.format(hour=hr_str, ext='_ext')}'
                s3.download_file(bucket_name, object_key, file_path)

            except Exception as e:
                files.remove(file_path)
                print(f"Error downloading s3://{bucket_name}/{object_key}: {e}")

    return files


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

    home = os.path.expanduser('~')
    d = os.path.join(home, 'data', 'IrrigationGIS')
    rtma = os.path.join(home, 'data', 'rtma')

    if not os.path.isdir(d):
        d = os.path.join(home, 'data', 'IrrigationGIS')
        rtma = os.path.join(home, 'data', 'rtma')

    if not os.path.isdir(d):
        d = os.path.join('/data', 'IrrigationGIS')
        rtma = os.path.join('/data', 'rtma')

    sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_29OCT2024.csv')
    # sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')

    out_files = os.path.join(rtma, 'station_data')
    grb_files_ = os.path.join(rtma, 'grib')

    bounds_ = (-124.0, 23.0, -66.0, 52.0)
    quadrants = get_quadrants(bounds_)

    for e, quad in enumerate(quadrants, start=1):
        print(f'\n\n\n\n Quadrant {e} \n\n\n\n')

        extract_rtma(sites, out_files, grb_data=grb_files_, workers=16, overwrite=False,
                     bounds=quad, debug=True, start_yr=2014, end_yr=2024)

        # process_and_concat_csv(sites, csv_files, start_date='2016-01-01', end_date='2023-12-31', outdir=p_files,
        #                        workers=16, missing_file=None)

# ========================= EOF ====================================================================
