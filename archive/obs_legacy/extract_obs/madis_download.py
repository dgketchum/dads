import json
import multiprocessing
import os
import subprocess
import time
from datetime import datetime, timedelta

import pandas as pd

BASE_URL = "https://madis-data.ncep.noaa.gov/madisResearch/data"

credentials_file = os.path.join(
    os.path.expanduser("~"),
    "code",
    "dads",
    "extract",
    "met_data",
    "obs",
    "madis_credentials.json",
)

with open(credentials_file, "r") as fp:
    creds = json.load(fp)
USR = creds["usr"]
PSWD = creds["pswd"]

print("Requesting data for User: {}".format(USR))


def generate_monthly_time_tuples(start_year, end_date=None):
    if end_date is None:
        end_date = datetime.now()
    time_tuples = []

    for year in range(start_year, end_date.year + 1):
        for month in range(1, 13):
            if year == end_date.year and month > end_date.month:
                break
            start_day = 1
            if year == end_date.year and month == end_date.month:
                end_day = end_date.day
            elif month < 12:
                end_day = (datetime(year, month + 1, 1) - timedelta(days=1)).day
            else:
                end_day = 31

            start_time_str = f"{year}{month:02}{start_day:02} 00"
            end_time_str = f"{year}{month:02}{end_day:02} 23"

            time_tuples.append((start_time_str, end_time_str))

    return time_tuples


def download(
    start_time, end_time, madis_data_dir, username, password, downloader="wget"
):
    """"""
    start_dt = datetime.strptime(start_time, "%Y%m%d %H")
    end_dt = datetime.strptime(end_time, "%Y%m%d %H")

    current_dt = start_dt
    download_count = 0

    while current_dt <= end_dt:
        start = time.perf_counter()
        date_str = current_dt.strftime("%Y/%m/%d")
        remote_dir = f"/archive/{date_str}/LDAD/mesonet/netCDF"

        hr_files = [
            "{}_{}.gz".format(
                current_dt.strftime("%Y%m%d"), str(hr).rjust(2, "0").ljust(4, "0")
            )
            for hr in range(0, 24)
        ]
        targets = [os.path.join(madis_data_dir, hrfile) for hrfile in hr_files]

        if not all([os.path.exists(f) for f in targets]):
            try:
                if downloader == "wget":
                    wget_cmd = [
                        "wget",
                        "--user",
                        username,
                        "--password",
                        password,
                        "--no-check-certificate",
                        "--no-directories",
                        "--recursive",
                        "--level=1",
                        "--accept",
                        "*.gz",
                        "-q",
                        "--timeout=600",
                        f"{BASE_URL}{remote_dir}/",
                    ]
                    subprocess.run(wget_cmd, check=True, cwd=madis_data_dir)
                    download_count += 1
                elif downloader == "aria2c":
                    input_file = "aria2c_input.txt"
                    with open(input_file, "w") as f:
                        for file_name in hr_files:
                            strfile = "{}\n".format(
                                "".join([BASE_URL, remote_dir, "/", file_name])
                            )
                            f.write(strfile)
                    aria2c_cmd = [
                        "aria2c",
                        "-x",
                        "16",
                        "-s",
                        "16",
                        "--http-user",
                        username,
                        "--http-passwd",
                        password,
                        "--allow-overwrite=true",
                        "--dir",
                        madis_data_dir,
                        "--input-file",
                        input_file,
                    ]
                    subprocess.run(aria2c_cmd, check=True)
                else:
                    raise NotImplementedError("Choose wget or aria2c to download")

            except subprocess.CalledProcessError as e:
                print(f"{BASE_URL}{remote_dir}/")
                print(
                    f"Failed download to {madis_data_dir} for day {current_dt.strftime('%Y%m%d')} {e}"
                )

            if download_count % 10 == 0:
                print("sleep wait")
                time.sleep(0.5)

            end = time.perf_counter()
            print(
                f"Download Integrated Mesonet {current_dt} took {end - start:0.2f} seconds"
            )

        else:
            print(f"{current_dt} exists, skipping")

        current_dt += timedelta(days=1)


def process_time_chunk(time_tuple):
    start_time, end_time, madis_data_dir_ = time_tuple
    dt = pd.date_range(start_time, end_time, freq="h")
    dt = [d.strftime("%Y%m%d_%H00") for d in dt]
    hr_files = ["{}.gz".format(d) for d in dt]
    targets = [os.path.join(madis_data_dir_, hrfile) for hrfile in hr_files]
    if not all([os.path.exists(f) for f in targets]):
        download(start_time, end_time, madis_data_dir_, USR, PSWD, downloader="aria2c")
    else:
        print("{} data exists in {}".format(time_tuple, madis_data_dir_))


def download_file_list(txt_file, output_dir):

    with open(txt_file, "r") as fp:
        lines = fp.read().splitlines()

    for filename in lines:
        target_path = os.path.join(output_dir, filename)
        if os.path.exists(target_path):
            return
        date_str = filename.split("_")[0]
        year, month, day = date_str[0:4], date_str[4:6], date_str[6:8]
        remote_url = (
            f"{BASE_URL}/archive/{year}/{month}/{day}/LDAD/mesonet/netCDF/{filename}"
        )
        subprocess.run(
            [
                "wget",
                "--user",
                USR,
                "--password",
                PSWD,
                "--no-check-certificate",
                "-q",
                "--timeout=600",
                "-O",
                target_path,
                remote_url,
            ],
            check=False,
            capture_output=True,
        )


if __name__ == "__main__":
    madis_data_dir = "/nas/climate/madis/LDAD/mesonet/netCDF"

    # the FTP we're currently using has from 2001-07-01
    times = generate_monthly_time_tuples(2001)

    args = [(t[0], t[1], madis_data_dir) for t in times]

    with multiprocessing.Pool(processes=10) as pool:
        pool.map(process_time_chunk, args)
