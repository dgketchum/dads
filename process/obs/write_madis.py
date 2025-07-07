import glob
import gzip
import json
import multiprocessing
import os
import shutil
import time
import warnings
import concurrent.futures
from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd
import xarray as xr
import numpy as np

warnings.filterwarnings("ignore", category=xr.SerializationWarning)

DESIRED_PARAMS = {
    # --- Core Identification and Location ---
    'stationId': str,
    'providerId': str,
    'dataProvider': str,
    'observationTime': str,
    'stationType': str,
    'elevation': float,
    'latitude': float,
    'longitude': float,

    # --- Add PST codes to be merged and saved ---
    'code1PST': str, # Defines precipAccum period
    'code2PST': str, # Defines solarRadiation period/type

    # --- Temperature and its full QC suite ---
    'temperature': float,
    'temperatureDD': str,

    # --- Dewpoint and its full QC suite ---
    'dewpoint': float,
    'dewpointDD': str,

    # --- Relative Humidity and its QC suite ---
    'relHumidity': float,
    'relHumidityDD': str,

    # --- Wind Speed and its full QC suite ---
    'windSpeed': float,
    'windSpeedDD': str,

    # --- Wind Direction and its full QC suite ---
    'windDir': float,
    'windDirDD': str,

    # --- Precipitation and its full QC suite ---
    'precipAccum': float,
    'precipAccumDD': str,

    # --- Solar Radiation ---
    'solarRadiation': float,
}

# --- Provider-Subprovider Tables (PST) ---
PST = {
    'code1PST': str, # method of precip reporting
    'code2PST': str, # method of solrad reporting
    'code3PST': str, # PST code1/code2 usage definition
    'code4PST': str, # rawPrecip variable definition -- not used
    'namePST': str, # PST Provider or Subprovider name
    'typePST': str, # PST type
}

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
            ds_ = xr.open_dataset(fp, engine='scipy', cache=False)
    except OverflowError as oe:
        print(f"OverflowError decoding time in file: {f}. Skipping. Error: {oe}")
        return None
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


def read_madis_hourly(data_directory, year_mo_str, output_directory, bounds=(-125., 24., -66., 53.),
                      select=None):
    """"""
    file_pattern = os.path.join(data_directory, f"*{year_mo_str}*.gz")
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        print(f"No files found for date: {year_mo_str}")
        return

    start = time.perf_counter()

    all_dfs, first, pst_lookup, providers = [], True, None, None

    for filename in file_list:

        ds = open_nc(filename)

        if ds is None:
            continue

        pstdct = {k: [int(i) for i in ds[k].values if np.isfinite(i)] for k in PST if 'code' in k}
        [pstdct.update({k: [i.strip() for i in ds[k].values.astype(str) if i != ''] for k in PST if 'code' not in k})]
        pst_map_df = pd.DataFrame(pstdct)
        pst_map_df = pst_map_df[pst_map_df['namePST'] != '']

        data_vars = {}
        params_to_extract = {k: v for k, v in DESIRED_PARAMS.items() if k not in PST}

        for var_name, var_type in params_to_extract.items():
            if var_name in ds:
                if var_name == 'observationTime':
                    data_vars[var_name] = pd.to_datetime(ds[var_name].values, errors='coerce')
                elif var_type == str:
                    data_vars[var_name] = [val.decode().strip() for val in ds[var_name].values]
                else:
                    data_vars[var_name] = ds[var_name].values.astype(var_type)
            else:
                if 'recNum' in ds.dims:
                    default_val = '' if var_type == str else np.nan
                    data_vars[var_name] = [default_val] * ds.dims['recNum']

        df = pd.DataFrame(data_vars)

        if df.empty:
            continue

        df = pd.merge(df, pst_map_df, left_on='dataProvider', right_on='namePST', how='left')

        df.dropna(how='all', inplace=True)

        if bounds is not None:
            df = df.loc[
                (df['latitude'] < bounds[3]) & (df['latitude'] >= bounds[1]) &
                (df['longitude'] < bounds[2]) & (df['longitude'] >= bounds[0])
                ]

        if df.empty:
            continue

        all_dfs.append(df)

    if not all_dfs:
        print("No valid data found in any files.")
        return

    master_df = pd.concat(all_dfs, ignore_index=True)

    if select is not None:
        master_df = master_df[master_df['stationId'].isin(select)]

    for station_id, station_df in master_df.groupby('stationId'):

        try:
            station_df = station_df.set_index('observationTime')

            if station_df.empty:
                continue

            out_dir = os.path.join(output_directory, station_id)
            os.makedirs(out_dir, exist_ok=True)

            out_file = os.path.join(out_dir, f'{station_id}_{year_mo_str}.parquet')

            station_df.reset_index(inplace=True)
            station_df.rename(columns={'observationTime': 'datetime'}, inplace=True)

            station_df['stationId'] = station_id

            final_cols = [c for c in DESIRED_PARAMS.keys() if c in station_df.columns]
            final_cols.insert(0, 'stationId')
            final_cols.insert(0, 'datetime')
            final_cols = list(dict.fromkeys(final_cols))

            station_df[final_cols].to_parquet(out_file)

        except Exception as err:
            print(f'error on {station_id}, {year_mo_str}: {err}', flush=True)
            continue

    end = time.perf_counter()
    current_datetime = datetime.now()
    dtstr = current_datetime.strftime("%Y-%m-%d %H:%M")
    print(f"{year_mo_str}: {len(file_list)} files took {end - start:0.4f} seconds {dtstr}", flush=True)


def process_time_chunk(args):
    time_tuple, meso_dir, out_dir, bnds, sel = args
    start_time, end_time = time_tuple
    read_madis_hourly(meso_dir, start_time[:6], out_dir, bounds=bnds, select=sel)


if __name__ == "__main__":

    mesonet_dir = '/data/ssd2/madis'
    out_dir_ = os.path.join(mesonet_dir, 'extracts_qaqc')

    tracker_ = os.path.join(mesonet_dir, 'stations.json')
    netcdf_src = os.path.join(mesonet_dir, 'netCDF')

    bnds = None

    times = generate_monthly_time_tuples(2017, 2025, check_dir=None)

    args_ = [(t, netcdf_src, out_dir_, bnds, None) for t in times]
    print(f'{len(args_)} months to process')

    debug = True

    if debug:
        for a in args_:
            process_time_chunk(a)
            print(f'success on {a[0]}')

    num_processes = 8
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_time_chunk, args_)

# ========================= EOF ====================================================================
