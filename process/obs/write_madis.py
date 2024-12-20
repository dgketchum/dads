import glob
import gzip
import json
import multiprocessing
import os
import shutil
import time
import warnings
from datetime import datetime, timedelta

import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=xr.SerializationWarning)

DESIRED_PARAMS = ['stationId', 'reportTime', 'elevation', 'stationType', 'latitude', 'longitude',
                  'precipAccum',
                  'solarRadiation',
                  'temperature',
                  'dewpoint', 'relHumidity',
                  'windSpeed', 'windDir']

SUBHOUR_RESAMPLE_MAP = {
    'longitude': 'first',
    'latitude': 'first',
    'elevation': 'first',
    'stationType': 'first',
    'relHumidity': 'mean',
    'dewpoint': 'mean',
    'precipAccum': 'sum',
    'solarRadiation': 'mean',
    'temperature': 'mean',
    'windSpeed': 'mean',
    'windDir': 'mean',
}

MET_PARAMS = DESIRED_PARAMS[6:]

METADATA = ['elevation', 'stationType', 'latitude', 'longitude']


def copy_file(source_file, dest_file):
    if os.path.exists(dest_file):
        return
    try:
        shutil.copy(source_file, dest_file)
        if dest_file.endswith('01_0000.gz'):
            print(dest_file)
    except Exception as e:
        print(e, os.path.basename(source_file))


def transfer_list(data_directory, dst, progress_json=None, yrmo_str=None, workers=2):
    if progress_json:
        with open(progress_json, 'r') as f:
            progress = json.load(f)
            yrmo_str = progress['complete']

        files_ = os.listdir(data_directory)
        yrmo = [str(f[:6]) for f in files_]
        file_list = [os.path.join(data_directory, f) for f, ym in zip(files_, yrmo) if ym not in yrmo_str]
        dst_list = [os.path.join(dst, f) for f, ym in zip(files_, yrmo) if ym not in yrmo_str]
        print(len(file_list), 'files')

    else:
        files_ = os.listdir(data_directory)
        yrmo = [str(f[:6]) for f in files_]
        file_list = [os.path.join(data_directory, f) for f, ym in zip(files_, yrmo) if ym in yrmo_str]
        dst_list = [os.path.join(dst, f) for f, ym in zip(files_, yrmo) if ym in yrmo_str]
        print(len(file_list), 'files')

    with multiprocessing.Pool(processes=workers) as pool:
        pool.starmap(copy_file, zip(file_list, dst_list))


def generate_monthly_time_tuples(start_year, end_year, check_dir=None):
    idxs = []
    if check_dir:
        file_list, dct = [], {}
        for root, dirs, files in os.walk(check_dir):
            for file in files:
                file_list.append(os.path.join(root, file))

        yrmos = [f.split('.')[0][-6:] for f in file_list]
        idxs = sorted(list(set(yrmos)))

    time_tuples = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            mstr = str(month).rjust(2, '0')

            if check_dir and f'{year}{mstr}' in idxs:
                print('{}/{} exists'.format(mstr, year))
                continue

            start_day = 1
            end_day = ((datetime(year, month + 1, 1) - timedelta(days=1)).day if month < 12 else 31)

            start_time_str = f"{year}{month:02}{start_day:02} 00"
            end_time_str = f"{year}{month:02}{end_day:02} 23"

            time_tuples.append((start_time_str, end_time_str))

    return time_tuples


def open_nc(f):
    temp_nc_file = None

    try:
        with gzip.open(f) as fp:
            ds_ = xr.open_dataset(fp, engine='scipy')
    except Exception as e:
        try:
            temp_nc_file = f.replace('.gz', '.nc')
            with gzip.open(f, 'rb') as f_in, open(temp_nc_file, 'wb') as f_out:
                f_out.write(f_in.read())
            ds_ = xr.open_dataset(temp_nc_file, engine='netcdf4')
        except Exception as e2:
            print(f"Error writing to temporary .nc file or reading with netCDF4 for {f}: {e2}")
            return None
        finally:
            if temp_nc_file and os.path.exists(temp_nc_file):
                os.remove(temp_nc_file)
    return ds_


def read_madis_hourly(data_directory, year_mo_str, output_directory, bounds=(-125., 25., -66., 49.)):
    """"""

    file_pattern = os.path.join(data_directory, f"*{year_mo_str}*.gz")
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        print(f"No files found for date: {year_mo_str}")
        return

    data, sites = {}, pd.DataFrame().to_dict()

    start = time.perf_counter()
    for j, filename in enumerate(file_list):

        dt_str = os.path.basename(filename).split('.')[0].replace('_', '')

        ds = open_nc(filename)
        if ds is None:
            continue

        valid_data = ds[DESIRED_PARAMS]
        df = valid_data.to_dataframe()
        df['stationId'] = df['stationId'].astype(str)
        df = df[(df['latitude'] < bounds[3]) & (df['latitude'] >= bounds[1])]
        df = df[(df['longitude'] < bounds[2]) & (df['longitude'] >= bounds[0])]
        df.dropna(how='all', inplace=True)
        df.set_index('stationId', inplace=True, drop=True)
        df = df.groupby(df.index).agg(SUBHOUR_RESAMPLE_MAP)

        df['v'] = df.apply(lambda row: [float(row[v]) for v in MET_PARAMS], axis=1)
        df.drop(columns=METADATA, inplace=True)
        dct = df.to_dict(orient='index')

        for k, v in dct.items():
            if not k in data.keys():
                data[k] = {dt_str: v['v']}
            else:
                data[k][dt_str] = v['v']

    for k, v in data.items():

        out_dir = os.path.join(output_directory, k)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        out_file = os.path.join(out_dir, '{}_{}.csv'.format(k, year_mo_str))
        df = pd.DataFrame.from_dict(v, orient='index')
        df.columns = MET_PARAMS
        df['datetime'] = df.index
        df = df[['datetime'] + MET_PARAMS]
        df.to_csv(out_file, index=False)

    end = time.perf_counter()
    print(f"Processing {len(file_list)} files took {end - start:0.4f} seconds")


def process_time_chunk(args):
    time_tuple, meso_dir, out_dir = args
    start_time, end_time = time_tuple
    read_madis_hourly(meso_dir, start_time[:6], out_dir, bounds=(-180., 25., -60., 85.))


if __name__ == "__main__":

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    mesonet_dir = os.path.join(d, 'climate', 'madis', 'LDAD', 'mesonet')
    tracker_ = os.path.join(mesonet_dir, 'stations.json')

    if os.path.exists('/data/ssd1/madis'):
        netcdf_src = os.path.join(mesonet_dir, 'netCDF')
        netcdf_dst = os.path.join('/data/ssd1/madis', 'netCDF')
        out_dir_ = os.path.join('/data/ssd1/madis', 'inclusive_csv')
        print('operating on zoran data')
    else:
        netcdf_src = os.path.join(mesonet_dir, 'netCDF')
        netcdf_dst = os.path.join(mesonet_dir, 'netCDF')
        out_dir_ = os.path.join(mesonet_dir, 'inclusive_csv')
        print('operating on network drive data')

    # get_station_metadata(netcdf_src, tracker_)
    shp_ = os.path.join(mesonet_dir, 'stations_25OCT2024.shp')
    # write_stations_to_shapefile(tracker_, shp_)

    dt = pd.date_range('2001-01-01', '2009-12-31', freq='MS')
    dts = [d.strftime('%Y%m') for d in dt]
    transfer_list(netcdf_src, netcdf_dst, progress_json=None, yrmo_str=dts, workers=20)

    times = generate_monthly_time_tuples(2001, 2009, check_dir=out_dir_)
    [print(t) for t in times]
    args_ = [(t, netcdf_dst, out_dir_) for t in times]

    # num_processes = 5
    num_processes = 20
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_time_chunk, args_)

# ========================= EOF ====================================================================
