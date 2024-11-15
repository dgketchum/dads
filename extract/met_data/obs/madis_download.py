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

credentials_file = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'dads', 'extract', 'met_data',
                                'madis_credentials.json')

with open(credentials_file, 'r') as fp:
    creds = json.load(fp)
USR = creds['usr']
PSWD = creds['pswd']

print('Requesting data for User: {}'.format(USR))


def generate_monthly_time_tuples(start_year, end_year):
    time_tuples = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            start_day = 1
            end_day = (
                (datetime(year, month + 1, 1) - timedelta(days=1)).day
                if month < 12
                else 31
            )

            start_time_str = f"{year}{month:02}{start_day:02} 00"
            end_time_str = f"{year}{month:02}{end_day:02} 23"

            time_tuples.append((start_time_str, end_time_str))

    return time_tuples


def download(start_time, end_time, madis_data_dir, username, password, downloader='wget'):
    """"""
    dataset_path = 'LDAD/mesonet/netCDF'
    local_dir = os.path.join(madis_data_dir, dataset_path)
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    start_dt = datetime.strptime(start_time, "%Y%m%d %H")
    end_dt = datetime.strptime(end_time, "%Y%m%d %H")

    current_dt = start_dt
    download_count = 0

    while current_dt <= end_dt:
        start = time.perf_counter()
        date_str = current_dt.strftime("%Y/%m/%d")
        remote_dir = f"/archive/{date_str}/{dataset_path}"

        hr_files = ['{}_{}.gz'.format(current_dt.strftime("%Y%m%d"),
                                      str(hr).rjust(2, '0').ljust(4, '0')) for hr in range(0, 24)]
        targets = [os.path.join(local_dir, hrfile) for hrfile in hr_files]

        if not all([os.path.exists(f) for f in targets]):
            try:
                if downloader == 'wget':
                    wget_cmd = [
                        "wget", "--user", username, "--password", password,
                        "--no-check-certificate", "--no-directories", "--recursive", "--level=1",
                        "--accept", "*.gz", "-q", "--timeout=600", f"{BASE_URL}{remote_dir}/"]
                    subprocess.run(wget_cmd, check=True, cwd=local_dir)
                    download_count += 1
                elif downloader == 'aria2c':
                    input_file = "aria2c_input.txt"
                    with open(input_file, "w") as f:
                        for file_name in hr_files:
                            strfile = "{}\n".format(''.join([BASE_URL, remote_dir, '/', file_name]))
                            f.write(strfile)
                    aria2c_cmd = [
                        "aria2c", "-x", "16", "-s", "16", "--http-user", username, "--http-passwd", password,
                        "--allow-overwrite=true", "--dir", local_dir, "--input-file", input_file,
                    ]
                    subprocess.run(aria2c_cmd, check=True)
                else:
                    raise NotImplementedError('Choose wget or aria2c to download')

            except subprocess.CalledProcessError as e:
                print(f"{BASE_URL}{remote_dir}/")
                print(f"Failed download to {local_dir} for day {current_dt.strftime('%Y%m%d')} {e}")

            if download_count % 10 == 0:
                print('sleep wait')
                time.sleep(0.5)

            end = time.perf_counter()
            print(f"Download Integrated Mesonet {current_dt} took {end - start:0.2f} seconds")

        else:
            print(f'{current_dt} exists, skipping')

        current_dt += timedelta(days=1)


def process_time_chunk(time_tuple):
    start_time, end_time = time_tuple
    dt = pd.date_range(start_time, end_time, freq='h')
    dt = [d.strftime("%Y%m%d_%H00") for d in dt]
    hr_files = ['{}.gz'.format(d) for d in dt]
    target_dir = os.path.join(madis_data_dir_, 'LDAD', 'mesonet', 'netCDF')
    targets = [os.path.join(target_dir, hrfile) for hrfile in hr_files]
    if not all([os.path.exists(f) for f in targets]):
        download(start_time, end_time, madis_data_dir_, USR, PSWD, downloader='aria2c')
    else:
        print('{} data exists in {}'.format(time_tuple, target_dir))


if __name__ == "__main__":

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    usr, pswd = 'usr', 'pswd'
    madis_data_dir_ = os.path.join(d, 'climate', 'madis')

    stn_meta = os.path.join(d, 'climate', 'madis', 'public_stn_list.csv')
    mesonet_dir = os.path.join(madis_data_dir_, 'LDAD', 'mesonet')

    # the FTP we're currently using has from 2001-07-01
    times = generate_monthly_time_tuples(2024, 2024)
    times = [t for t in times if int(t[0][:6]) >= 202409]
    times = [t for t in times if int(t[0][:6]) <= 202409]
    # random.shuffle(times)

    # num_processes = 1
    num_processes = 20

    mesonet_dir = os.path.join(madis_data_dir_, 'LDAD', 'mesonet')
    # debug
    # for t in times:
    #     process_time_chunk(t)

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_time_chunk, times)

# ========================= EOF ====================================================================
