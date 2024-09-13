import glob
import gzip
import json
import multiprocessing
import os
import shutil
import time
import warnings
from datetime import datetime, timedelta

import geopandas as gpd
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=xr.SerializationWarning)

SUBHOUR_RESAMPLE_MAP = {'relHumidity': 'mean',
                        'precipAccum': 'sum',
                        'solarRadiation': 'mean',
                        'temperature': 'mean',
                        'windSpeed': 'mean',
                        'longitude': 'mean',
                        'latitude': 'mean'}

params = ['relHumidity', 'precipAccum', 'solarRadiation', 'temperature', 'windSpeed']
COLS = ['datetime'] + params


def transfer_list(data_directory, dst, progress_json=None, yrmo_str=None):
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

    for i, (source_file, dest_file) in enumerate(zip(file_list, dst_list)):
        if os.path.exists(dest_file):
            continue
        try:
            shutil.copy(source_file, dest_file)
            if dest_file.endswith('01_0000.gz'):
                print(dest_file)
        except Exception as e:
            print(e, os.path.basename(source_file))
            continue


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

            start_day = 1
            end_day = ((datetime(year, month + 1, 1) - timedelta(days=1)).day if month < 12 else 31)

            start_time_str = f"{year}{month:02}{start_day:02} 00"
            end_time_str = f"{year}{month:02}{end_day:02} 23"

            time_tuples.append((start_time_str, end_time_str))

    return time_tuples


def read_madis_hourly(data_directory, year_mo_str, output_directory, shapedir=None, bounds=(-125., 25., -66., 49.)):
    """"""

    file_pattern = os.path.join(data_directory, f"*{year_mo_str}*.gz")
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        print(f"No files found for date: {year_mo_str}")
        return
    required_vars = ['stationId', 'relHumidity', 'precipAccum', 'solarRadiation', 'temperature',
                     'windSpeed', 'latitude', 'longitude']

    first = True
    data, sites = {}, pd.DataFrame().to_dict()

    start = time.perf_counter()
    for j, filename in enumerate(file_list):

        dt_str = os.path.basename(filename).split('.')[0].replace('_', '')
        temp_nc_file = None

        # following exception handler reads both new-style and classic NetCDF
        try:
            with gzip.open(filename) as fp:
                ds = xr.open_dataset(fp, engine='scipy')
        except Exception as e:
            try:
                temp_nc_file = filename.replace('.gz', '.nc')
                with gzip.open(filename, 'rb') as f_in, open(temp_nc_file, 'wb') as f_out:
                    f_out.write(f_in.read())
                ds = xr.open_dataset(temp_nc_file, engine='netcdf4')
            except Exception as e2:
                print(f"Error writing to temporary .nc file or reading with netCDF4 for {filename}: {e2}")
                continue
            finally:
                if temp_nc_file and os.path.exists(temp_nc_file):
                    os.remove(temp_nc_file)

        valid_data = ds[required_vars]
        df = valid_data.to_dataframe()
        df['stationId'] = df['stationId'].astype(str)
        df = df[(df['latitude'] < bounds[3]) & (df['latitude'] >= bounds[1])]
        df = df[(df['longitude'] < bounds[2]) & (df['longitude'] >= bounds[0])]
        df.dropna(how='any', inplace=True)
        df.set_index('stationId', inplace=True, drop=True)

        if first and shapedir:
            sites = df[['latitude', 'longitude']]
            sites = sites.groupby(sites.index).mean()
            sites = sites.to_dict(orient='index')
        elif shapedir:
            for i, r in df.iterrows():
                if i not in sites.keys():
                    sites[i] = {'latitude': r['latitude'], 'longitude': r['longitude']}

        df = df.groupby(df.index).agg(SUBHOUR_RESAMPLE_MAP)
        df['v'] = df.apply(lambda row: [float(row[v]) for v in required_vars[1:6]], axis=1)
        df.drop(columns=required_vars[1:], inplace=True)
        dct = df.to_dict(orient='index')

        for k, v in dct.items():
            if not k in data.keys():
                data[k] = {dt_str: v['v']}
            else:
                data[k][dt_str] = v['v']

        if first and shapedir:
            write_locations(sites, shapedir, dt_str[:6])
            first = False
        else:
            first = False

    for k, v in data.items():

        out_dir = os.path.join(output_directory, k)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        out_file = os.path.join(out_dir, '{}_{}.csv'.format(k, year_mo_str))
        df = pd.DataFrame.from_dict(v, orient='index')
        df.columns = params
        df['datetime'] = df.index
        df = df[['datetime'] + params]
        df.to_csv(out_file, index=False)

    end = time.perf_counter()
    print(f"Processing {len(file_list)} files took {end - start:0.4f} seconds")


def write_locations(loc, shp_dir, dt_):
    try:
        df = pd.DataFrame.from_dict(loc, orient='index')
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')
        shp = os.path.join(shp_dir, f'integrated_mesonet_{dt_}.shp')
        gdf.to_file(shp)
    except:
        pass


def process_time_chunk(args):
    time_tuple, meso_dir, out_dir, out_shp = args
    start_time, end_time = time_tuple
    read_madis_hourly(meso_dir, start_time[:6], out_dir,
                      shapedir=out_shp, bounds=(-180., 25., -60., 85.))


if __name__ == "__main__":

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    mesonet_dir = os.path.join(d, 'climate', 'madis', 'LDAD', 'mesonet')
    outshp = os.path.join(mesonet_dir, 'shapes')

    if os.path.exists('/data/ssd1/madis'):
        netcdf_src = os.path.join(mesonet_dir, 'netCDF')
        netcdf_dst = os.path.join('/data/ssd1/madis', 'netCDF')
        out_dir_ = os.path.join('/data/ssd1/madis', 'yrmo_csv')
        print('operating on zoran data')
    else:
        netcdf_src = os.path.join(mesonet_dir, 'netCDF')
        netcdf_dst = os.path.join(mesonet_dir, 'netCDF')
        out_dir_ = os.path.join(mesonet_dir, 'yrmo_csv')
        print('operating on network drive data')

    # dt = pd.date_range('2001-01-01', '2010-12-31', freq='MS')
    # dts = [d.strftime('%Y%m') for d in dt]
    # transfer_list(netcdf_src, netcdf_dst, progress_json=None, yrmo_str=dts)

    times = generate_monthly_time_tuples(2001, 2010, check_dir=out_dir_)
    args_ = [(t, netcdf_dst, out_dir_, outshp) for t in times]
    # random.shuffle(args_)
    # args_.reverse()

    # debug
    # for t in args_[:1]:
    #     process_time_chunk(t)

    # num_processes = 1
    num_processes = 20
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_time_chunk, args_)

# ========================= EOF ====================================================================
