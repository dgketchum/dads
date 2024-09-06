import os
import io
import json
import glob
import gzip
import random
import subprocess
import multiprocessing
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr
import netCDF4 as nc

from utils.elevation import elevation_from_coordinate

warnings.filterwarnings("ignore", category=xr.SerializationWarning)

BASE_URL = "https://madis-data.ncep.noaa.gov/madisResearch/data"

SUBHOUR_RESAMPLE_MAP = {'relHumidity': 'mean',
                        'precipAccum': 'sum',
                        'solarRadiation': 'mean',
                        'temperature': 'mean',
                        'windSpeed': 'mean',
                        'longitude': 'mean',
                        'latitude': 'mean'}

credentials_file = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'dads', 'extract', 'met_data',
                                'credentials.json')

with open(credentials_file, 'r') as fp:
    creds = json.load(fp)
USR = creds['usr']
PSWD = creds['pswd']

print('Requesting data for User: {}'.format(USR))


def generate_monthly_time_tuples(start_year, end_year, directory=None):

    if directory:
        idx = []
        for f in os.listdir(directory):
            station_path = os.path.join(directory, f)
            df = pd.read_csv(station_path, index_col=0, parse_dates=True)
            dts = list(set((i.year, i.month) for i in df.index))
            idx.extend(dts)

    idxs = list(set(idx))

    time_tuples = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if directory and (year, month) in idxs:
                print('{}/{} exists'.format(month, year))

            start_day = 1
            end_day = ((datetime(year, month + 1, 1) - timedelta(days=1)).day if month < 12 else 31)

            start_time_str = f"{year}{month:02}{start_day:02} 00"
            end_time_str = f"{year}{month:02}{end_day:02} 23"

            time_tuples.append((start_time_str, end_time_str))

    return time_tuples


def read_madis_hourly(data_directory, date, output_directory, shapefile=None, bounds=(-125., 25., -96., 49.),
                      only_write_bad=False):
    file_pattern = os.path.join(data_directory, f"*{date}*.gz")
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        print(f"No files found for date: {date}")
        return
    required_vars = ['stationId', 'relHumidity', 'precipAccum', 'solarRadiation', 'temperature',
                     'windSpeed', 'latitude', 'longitude']

    first = True
    data, sites = {}, pd.DataFrame().to_dict()

    start = time.perf_counter()
    for filename in file_list:

        dt = os.path.basename(filename).split('.')[0].replace('_', '')
        temp_nc_file = None

        # following exception handler reads both new-style and classic NetCDF
        try:
            with gzip.open(filename) as fp:
                ds = xr.open_dataset(fp, engine='scipy')
                if only_write_bad:
                    continue
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

        if first and shapefile:
            sites = df[['latitude', 'longitude']]
            sites = sites.groupby(sites.index).mean()
            sites = sites.to_dict(orient='index')
        elif shapefile:
            for i, r in df.iterrows():
                if i not in sites.keys():
                    sites[i] = {'latitude': r['latitude'], 'longitude': r['longitude']}

        df = df.groupby(df.index).agg(SUBHOUR_RESAMPLE_MAP)
        df['v'] = df.apply(lambda row: [float(row[v]) for v in required_vars[1:6]], axis=1)
        df.drop(columns=required_vars[1:], inplace=True)
        dct = df.to_dict(orient='index')

        for k, v in dct.items():
            if not k in data.keys():
                data[k] = {dt: v['v']}
            else:
                data[k][dt] = v['v']

        if first and shapefile:
            write_locations(sites, shapefile)
            first = False
        else:
            first = False

        print('Wrote {} records from {}'.format(len(df), dt))

    for k, v in data.items():
        d = os.path.join(output_directory, k)
        if not os.path.exists(d):
            os.makedirs(d)
        f = os.path.join(d, '{}.csv'.format(k))
        df = pd.DataFrame.from_dict(v, orient='index')
        df.columns = required_vars[1:6]
        if os.path.exists(f):
            df.to_csv(f, mode='a', header=False)
        else:
            df.to_csv(f)

    end = time.perf_counter()
    print(f"Processing {len(file_list)} files took {end - start:0.4f} seconds")


def write_locations(loc, shp):
    df = pd.DataFrame.from_dict(loc, orient='index')
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')
    gdf.to_file(shp)
    print('Wrote {}'.format(os.path.basename(shp)))


def process_time_chunk(time_tuple, meso_dir, out_dir, out_shp):
    start_time, end_time = time_tuple
    read_madis_hourly(meso_dir, start_time[:6], out_dir, shapefile=out_shp, bounds=(-180., 25., -60., 85.))


def consolidate_station_data(directory, outdir):

    max_len = 0
    for station_id in os.listdir(directory):
        all_data = []
        station_path = os.path.join(directory, station_id)
        if os.path.isdir(station_path):
            for filename in os.listdir(station_path):
                if filename.endswith(".csv"):
                    filepath = os.path.join(station_path, filename)
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=False, axis=0)
        combined_df = combined_df.groupby(combined_df.index).agg('first')
        output_filename = os.path.join(outdir, f'{station_id}.csv')
        combined_df.to_csv(output_filename)
        shape = combined_df.shape[0]
        if shape > max_len:
            print(station_id, 'has greatest length', shape)
            max_len = shape


def madis_station_shapefile(mesonet_dir, meta_file, outfile):
    """"""
    meta = pd.read_csv(meta_file, index_col='STAID')
    meta = meta.groupby(meta.index).first()
    unique_ids = set()
    unique_id_gdf = gpd.GeoDataFrame()

    shapefiles = [os.path.join(mesonet_dir, f) for f in os.listdir(mesonet_dir) if f.endswith('.shp')]
    for shapefile in shapefiles:
        print(os.path.basename(shapefile))
        gdf = gpd.read_file(shapefile)
        if 'index' in gdf.columns:
            unique_rows = gdf[~gdf['index'].isin(unique_ids)].copy()
            unique_rows.index = unique_rows['index']
            idx = [i for i in meta.index if i in unique_rows.index]

            unique_rows.loc[idx, 'ELEV'] = meta.loc[idx, 'ELEV']
            for i, r in unique_rows.iterrows():
                if isinstance(r['ELEV'], type(None)):
                    r['ELEV'] = elevation_from_coordinate(r['longitude'], r['latitude'])

            unique_rows['ELEV'] = unique_rows['ELEV'].astype(float)
            unique_rows.loc[idx, 'NET'] = meta.loc[idx, 'NET']
            unique_rows.loc[idx, 'NAME'] = meta.loc[idx, 'NAME']
            unique_ids.update(unique_rows['index'].unique())

            if unique_id_gdf.empty:
                unique_id_gdf = unique_rows
            else:
                unique_id_gdf = pd.concat([unique_id_gdf, unique_rows])

    unique_id_gdf.drop(columns=['index'], inplace=True)
    unique_id_gdf.to_file(outfile)
    print(outfile)


if __name__ == "__main__":

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    mesonet_dir = os.path.join(d, 'climate', 'madis', 'LDAD', 'mesonet')
    netcdf = os.path.join(mesonet_dir, 'netCDF')
    out_dir_ = os.path.join(mesonet_dir, 'csv')
    outshp = os.path.join(mesonet_dir, 'shapes')

    # the FTP we're currently using has from 2001-07-01
    times = generate_monthly_time_tuples(2005, 2013, out_dir_)
    times = [t for t in times if int(t[0][:6]) > 200501]
    times = [t for t in times if int(t[0][:6]) < 201312]
    # random.shuffle(times)

    num_processes = 1
    # num_processes = 20

    # messy_csv = os.path.join(mesonet_dir, 'csv_')
    # consolidate_station_data(messy_csv, out_dir_)

    # debug
    # for t in times:
    #     process_time_chunk(t)

    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     pool.map(process_time_chunk, times)

    # sites = os.path.join(madis_data_dir_, 'mesonet_sites.shp')
    # madis_station_shapefile(madis_shapes, stn_meta, sites)

# ========================= EOF ====================================================================
