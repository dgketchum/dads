import toml
import datetime
import gzip
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

BASE_URL = "https://madis-data.ncep.noaa.gov/madisPublic/data"

DATASET_PATHS = {
    "METAR": "point/metar/netcdf",
    "SAO": "point/sao/netcdf",
    "MARITIME": "point/maritime/netcdf",
    "MODERNIZED_COOP": "LDAD/coop/netCDF",
    "URBANET": "LDAD/urbanet/netCDF",
    "INTEGRATED_MESONET": "LDAD/mesonet/netCDF",
    "HYDROLOGICAL_SURFACE": "LDAD/hydro/netCDF",
    "MULTI_AGENCY_PROFILER": "LDAD/profiler/netCDF",
    "SNOW": "LDAD/snow/netCDF",
    "WISDOM": "LDAD/WISDOM/netCDF",
    "SATELLITE_WIND_3_HOUR": "point/HDW/netcdf",
    "SATELLITE_WIND_1_HOUR": "point/HDW1h/netcdf",
    "SATELLITE_SOUNDING": "point/POES/netcdf",
    "SATELLITE_RADIANCE": "point/satrad/netcdf",
    "RADIOMETER": "point/radiometer/netcdf",
    "RADIOSONDE": "point/raob/netcdf",
    "AUTOMATED_AIRCRAFT_REPORTS": "point/acars/netcdf",
    "AUTOMATED_AIRCRAFT_PROFILES_AT_AIRPORTS": "point/acarsProfiles/netcdf",
    "NOAA_PROFILER_NETWORK": "point/profiler/netcdf",
    "CRN": "LDAD/crn/netCDF",
    "HCN": "LDAD/hcn/netCDF",
    "NEPP": "LDAD/nepp/netCDF",
    "HFMETAR": "LDAD/hfmetar/netCDF",
    "RWIS": "LDAD/rwis/netCDF",
}



def parse_config(filename):
    """Parse TOML configuration file to get datasets and time range."""

    with open(filename, "r") as f:
        config = toml.load(f)

    # Time range
    start_time = config["time"]["start"]
    end_time = config["time"]["end"]

    # Selected datasets (filtering for True values)
    datasets = [ds for ds, selected in config["datasets"].items() if selected]

    # FTP server location
    ftp_location = config["ftp"]["server_location"]

    return datasets, start_time, end_time, ftp_location


def download_and_extract(dataset, start_time, end_time, madis_data_dir, username, password, ftp_location):
    """Download and extract data for a given dataset."""

    # Construct dataset path (relative to MADIS_DATA)
    dataset_path = DATASET_PATHS.get(dataset.upper().replace(" ", "_"))
    if not dataset_path:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Ensure local directory exists
    local_dir = os.path.join(madis_data_dir, dataset_path)
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    # Convert start/end times to datetime objects
    start_dt = datetime.strptime(start_time, "%Y%m%d %H")
    end_dt = datetime.strptime(end_time, "%Y%m%d %H")

    # Iterate over days in the specified time range
    current_dt = start_dt
    download_count = 0
    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y/%m/%d")
        remote_dir = f"/archive/{date_str}/{dataset_path}"
        print(f"{BASE_URL}{remote_dir}/")
        print(f"Downloading data to {local_dir} for day {current_dt.strftime('%Y%m%d')}")

        try:
            wget_cmd = [
                "wget", "--user", username, "--password", password,
                "--no-check-certificate", "--no-directories", "--recursive", "--level=1",
                "--accept", "*.gz", "--timeout=600", f"{BASE_URL}{remote_dir}/"
            ]
            subprocess.run(wget_cmd, check=True, cwd=local_dir)
            download_count += 1
        except subprocess.CalledProcessError as e:
            print(f"Error downloading data: {e}")

        if download_count % 10 == 0:
            print('sleep wait')
            time.sleep(60)

        current_dt += timedelta(days=1)


if __name__ == "__main__":
    username, password = 'usr', 'pswd'

    madis_data_dir = '/home/dgketchum/data/IrrigationGIS/climate/madis'
    conf = '/home/dgketchum/PycharmProjects/dads/extract/met_data/madis_config.toml'

    # Parse configuration file to get parameters
    datasets, start_time, end_time, ftpLoc = parse_config(conf)

    # Iterate over datasets and download/extract
    for dataset in datasets:
        download_and_extract(dataset, start_time, end_time, madis_data_dir, username, password, ftpLoc)

# ========================= EOF ====================================================================
