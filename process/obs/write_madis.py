import glob
import gzip
import json
import multiprocessing
import os
import random
import time
import warnings
from datetime import datetime, timedelta

import geopandas as gpd
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=xr.SerializationWarning)

BASE_URL = "https://madis-data.ncep.noaa.gov/madisResearch/data"

SUBHOUR_RESAMPLE_MAP = {'relHumidity': 'mean',
                        'precipAccum': 'sum',
                        'solarRadiation': 'mean',
                        'temperature': 'mean',
                        'windSpeed': 'mean',
                        'longitude': 'mean',
                        'latitude': 'mean'}


def generate_monthly_time_tuples(start_year, end_year, check_dir=None):
    idxs = []
    if check_dir:
        idx = []
        for f in os.listdir(check_dir):
            station_path = os.path.join(check_dir, f)
            df = pd.read_csv(station_path, index_col=0, parse_dates=True)
            try:
                dts = list(set((i.year, i.month) for i in df.index))
            except Exception:
                continue
            idx.extend(dts)

        idxs = list(set(idx))

    time_tuples = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):

            if check_dir and (year, month) in idxs:
                print('{}/{} exists'.format(month, year))

            start_day = 1
            end_day = ((datetime(year, month + 1, 1) - timedelta(days=1)).day if month < 12 else 31)

            start_time_str = f"{year}{month:02}{start_day:02} 00"
            end_time_str = f"{year}{month:02}{end_day:02} 23"

            time_tuples.append((start_time_str, end_time_str))

    return time_tuples


def read_madis_hourly(data_directory, year_mo_str, output_directory, shapedir=None, bounds=(-125., 25., -66., 49.),
                      progress_json=None):
    """"""

    if progress_json:
        with open(progress_json, 'r') as f:
            progress = json.load(f)

        if year_mo_str in progress['complete']:
            print(year_mo_str, 'exists in progress file, skipping')
            return

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
    for filename in file_list:

        dt_str = os.path.basename(filename).split('.')[0].replace('_', '')
        temp_nc_file = None

        # following exception handler reads both new-style and classic NetCDF
        try:
            with gzip.open(filename) as fp:
                ds = xr.open_dataset(fp, engine='scipy')
        except Exception as e:  # Catch any exceptions during xarray open
            if first:
                print(f"Error opening {filename} with xarray: {e}")
            try:
                temp_nc_file = filename.replace('.gz', '.nc')
                with gzip.open(filename, 'rb') as f_in, open(temp_nc_file, 'wb') as f_out:
                    f_out.write(f_in.read())

                ds = xr.open_dataset(temp_nc_file, engine='netcdf4')
                if first:
                    print(f"Successfully read {filename} after writing to temporary .nc file")
            except Exception as e2:
                print(f"Error writing to temporary .nc file or reading with netCDF4 for {filename}: {e2}")
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

        print('Wrote {} records from {}'.format(len(df), dt_str))

    for k, v in data.items():
        f = os.path.join(output_directory, '{}.csv'.format(k))
        df = pd.DataFrame.from_dict(v, orient='index')
        df.columns = required_vars[1:6]
        if os.path.exists(f):
            df.to_csv(f, mode='a', header=False)
        else:
            df.to_csv(f)

    if progress_json:
        # re-open tracker may have updated during execution
        with open(progress_json, 'r') as f:
            progress = json.load(f)
        progress['complete'].append(year_mo_str)
        with open(progress_json, 'w') as fp:
            json.dump(progress, fp, indent=4)

    end = time.perf_counter()
    print(f"Processing {len(file_list)} files took {end - start:0.4f} seconds")


def write_locations(loc, shp_dir, dt_):
    df = pd.DataFrame.from_dict(loc, orient='index')
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')
    shp = os.path.join(shp_dir, f'integrated_mesonet_{dt_}.shp')
    gdf.to_file(shp)
    print('Wrote {}'.format(os.path.basename(shp)))


def process_time_chunk(args):
    time_tuple, meso_dir, out_dir, out_shp, prog_ = args
    start_time, end_time = time_tuple
    read_madis_hourly(meso_dir, start_time[:6], out_dir, progress_json=prog_,
                      shapedir=out_shp, bounds=(-180., 25., -60., 85.))


if __name__ == "__main__":

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    mesonet_dir = os.path.join(d, 'climate', 'madis', 'LDAD', 'mesonet')
    netcdf = os.path.join(mesonet_dir, 'netCDF')
    out_dir_ = os.path.join(mesonet_dir, 'csv')
    outshp = os.path.join(mesonet_dir, 'shapes')
    progress_ = os.path.join(mesonet_dir, 'madis_progress.json')
    # progress_ = None

    # num_processes = 1
    num_processes = 10

    times = generate_monthly_time_tuples(2001, 2024, check_dir=None)
    args_ = [(t, netcdf, out_dir_, outshp, progress_) for t in times]
    random.shuffle(args_)
    # debug
    # for t in args_:
    #     process_time_chunk(t)

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_time_chunk, args_)

# ========================= EOF ====================================================================
