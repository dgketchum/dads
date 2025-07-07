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

    # --- Temperature and its full QC suite ---
    'temperature': float,
    'temperatureDD': str,
    # 'temperatureQCA': int,
    # 'temperatureQCR': int,
    # 'temperatureQCD': 'object',
    # 'temperatureICA': int,
    # 'temperatureICR': int,

    # --- Dewpoint and its full QC suite ---
    'dewpoint': float,
    'dewpointDD': str,
    # 'dewpointQCA': int,
    # 'dewpointQCR': int,
    # 'dewpointQCD': 'object',
    # 'dewpointICA': int,
    # 'dewpointICR': int,

    # --- Relative Humidity and its QC suite ---
    'relHumidity': float,
    'relHumidityDD': str,
    # 'relHumidityQCA': int,
    # 'relHumidityQCR': int,
    # 'relHumidityQCD': 'object',

    # --- Wind Speed and its full QC suite ---
    'windSpeed': float,
    'windSpeedDD': str,
    # 'windSpeedQCA': int,
    # 'windSpeedQCR': int,
    # 'windSpeedQCD': 'object',
    # 'windSpeedICA': int,
    # 'windSpeedICR': int,

    # --- Wind Direction and its full QC suite ---
    'windDir': float,
    'windDirDD': str,
    # 'windDirQCA': int,
    # 'windDirQCR': int,
    # 'windDirQCD': 'object',
    # 'windDirICA': int,
    # 'windDirICR': int,

    # --- Precipitation and its full QC suite ---
    'precipAccum': float,
    'precipAccumDD': str,
    # 'precipAccumQCA': int,
    # 'precipAccumQCR': int,
    # 'precipAccumQCD': 'object',
    # 'precipAccumICA': int,
    # 'precipAccumICR': int,

    # --- Solar Radiation (lacks standard QC flags in this file type) ---
    'solarRadiation': float,

}

# --- Provider-Subprovider Table (PST) ---
PST = {
    'code1PST': str,
    'code2PST': str,
    'code3PST': str,
    'code4PST': str,
    'namePST': str,
    'typePST': str,
}


def copy_file(source_file, dest_file):
    if os.path.exists(dest_file):
        return
    try:
        shutil.copy(source_file, dest_file)
        if dest_file.endswith('01_0000.gz'):
            print(dest_file)
    except Exception as e:
        print(e, os.path.basename(source_file))


def transfer_list(data_directory, dst, yrmo_str=None, workers=2):
    files_ = sorted(os.listdir(data_directory))
    yrmo = [str(f[:6]) for f in files_]

    if yrmo_str:
        file_list = [os.path.join(data_directory, f) for f, ym in zip(files_, yrmo) if ym in yrmo_str]
        dst_list = [os.path.join(dst, f) for f, ym in zip(files_, yrmo) if ym in yrmo_str]
    else:
        file_list = [os.path.join(data_directory, f) for f in files_]
        dst_list = [os.path.join(dst, f) for f in files_]

    print(f"{len(file_list)} files to transfer.")

    with multiprocessing.Pool(processes=workers) as pool:
        tqdm(pool.starmap(copy_file, zip(file_list, dst_list)), total=len(file_list),
             desc="Transferring files", unit="file")


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
                      select=None, pst_dir=None):
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

        if first:
            providers = ds['namePST'].values.astype(str)
            pst_lookup = {k: list([i.item() for i in ds[k].values.astype(v)]) for k, v in PST.items()}
            first = False
        else:
            ds_prov = ds['namePST'].values.astype(str)
            assert all([x == y for x, y in zip(providers, ds_prov)])

        data_vars = {}

        for var_name, var_type in DESIRED_PARAMS.items():
            if var_name in ds:
                data_vars[var_name] = ds[var_name].values.astype(var_type)
            else:
                if 'recNum' in ds.dims:
                    data_vars[var_name] = [np.nan] * ds.dims['recNum']

        df = pd.DataFrame(data_vars)

        if df.empty:
            continue

        df['observationTime'] = pd.to_datetime(df['observationTime'], errors='coerce')

        df.dropna(how='all', inplace=True)

        if bounds is not None:
            df = df.loc[
                (df['latitude'] < bounds[3]) & (df['latitude'] >= bounds[1]) &
                (df['longitude'] < bounds[2]) & (df['longitude'] >= bounds[0])
                ]

        if df.empty:
            continue

        all_dfs.append(df)

    if pst_dir:
        pst_file = os.path.join(pst_dir, f'pst_{year_mo_str}.json')
        with open(pst_file, 'w') as fp:
            json.dump(pst_lookup, fp, indent=4)

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

            out_file = os.path.join(out_dir, f'{station_id}_{year_mo_str}.parquet.gz')

            station_df.reset_index(inplace=True)
            station_df.rename(columns={'observationTime': 'datetime'}, inplace=True)

            station_df['stationId'] = station_id

            final_cols = ['datetime', 'stationId'] + list(DESIRED_PARAMS.keys())[1:]
            final_cols = [c for c in final_cols if c in station_df.columns]

            station_df[final_cols].to_parquet(out_file, compression='gzip')

        except Exception as err:
            print(f'error on {station_id}, {year_mo_str}: {err}')
            continue

    end = time.perf_counter()
    current_datetime = datetime.now()
    dtstr = current_datetime.strftime("%Y-%m-%d %H:%M")
    print(f"{year_mo_str}: {len(file_list)} files took {end - start:0.4f} seconds {dtstr}", flush=True)


def process_time_chunk(args):
    time_tuple, meso_dir, out_dir, bnds, sel, pst_d = args
    start_time, end_time = time_tuple
    read_madis_hourly(meso_dir, start_time[:6], out_dir, bounds=bnds, select=sel, pst_dir=pst_d)


if __name__ == "__main__":

    mesonet_dir = '/data/ssd2/madis'
    out_dir_ = os.path.join(mesonet_dir, 'extracts_qaqc')
    pst_ = os.path.join(mesonet_dir, 'pst')

    tracker_ = os.path.join(mesonet_dir, 'stations.json')
    netcdf_src = os.path.join(mesonet_dir, 'netCDF')

    bnds = None

    missing_yrmos = None

    log = os.path.join(mesonet_dir, 'write_madis_04JUL2025.txt')
    if os.path.isfile(log):
        with open(log, 'r') as fp:
            lines = fp.readlines()
            lines = [l for l in lines if 'files took']

        complete_yrmos = [l[:6] for l in lines]
    else:
        complete_yrmos = None

    print(f'{len(complete_yrmos)} completed months')
    bnds = None

    times = generate_monthly_time_tuples(2017, 2025, check_dir=None)

    if complete_yrmos:
        times = [t for t in times if t[0][:6] not in complete_yrmos]
        missing_yrmos = [s[0][:6] for s in times]

    args_ = [(t, netcdf_src, out_dir_, bnds, None, pst_) for t in times]
    print(f'{len(args_)} months to process')

    debug = False

    if debug:
        for a in args_:
            process_time_chunk(a)
            print(f'success on {a[0]}')

    num_processes = 5
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_time_chunk, args_)

# ========================= EOF ====================================================================
