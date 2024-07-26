import glob
import gzip
import os
import subprocess
import multiprocessing
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=xr.SerializationWarning)

BASE_URL = "https://madis-data.ncep.noaa.gov/madisPublic/data"

DATASET_PATHS = {
    "INTEGRATED_MESONET": "LDAD/mesonet/netCDF",
    "METAR": "point/metar/netcdf",
    "SAO": "point/sao/netcdf",
    "MARITIME": "point/maritime/netcdf",
    "MODERNIZED_COOP": "LDAD/coop/netCDF",
    "URBANET": "LDAD/urbanet/netCDF",
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

SUBHOUR_RESAMPLE_MAP = {'relHumidity': 'mean',
                        'precipAccum': 'sum',
                        'solarRadiation': 'mean',
                        'temperature': 'mean',
                        'windSpeed': 'mean',
                        'longitude': 'mean',
                        'latitude': 'mean'}


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


def download_and_extract(dataset, start_time, end_time, madis_data_dir, username, password):
    """Download and extract data for a given dataset."""

    dataset_path = DATASET_PATHS.get(dataset.upper().replace(" ", "_"))
    if not dataset_path:
        raise ValueError(f"Unknown dataset: {dataset}")

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
        print(f"{BASE_URL}{remote_dir}/")
        print(f"Downloading data to {local_dir} for day {current_dt.strftime('%Y%m%d')}")

        try:
            wget_cmd = [
                "wget", "--user", username, "--password", password,
                "--no-check-certificate", "--no-directories", "--recursive", "--level=1",
                "--accept", "*.gz", "-q", "--timeout=600", f"{BASE_URL}{remote_dir}/"
            ]
            subprocess.run(wget_cmd, check=True, cwd=local_dir)
            download_count += 1
        except subprocess.CalledProcessError as e:
            print(f"Error downloading data: {e}")

        if download_count % 10 == 0:
            print('sleep wait')
            time.sleep(60)

        current_dt += timedelta(days=1)
        end = time.perf_counter()
        print(f"Download for dataset '{dataset}' took {end - start:0.4f} seconds")


def read_madis_hourly(data_directory, date, output_directory, shapefile=None):
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

        try:
            with gzip.open(filename) as fp:
                ds = xr.open_dataset(fp)
                valid_data = ds[required_vars]
                df = valid_data.to_dataframe()
                df['stationId'] = df['stationId'].astype(str)
                df = df[(df['latitude'] < 49.1) & (df['latitude'] >= 25.0)]
                df = df[(df['longitude'] < -65) & (df['longitude'] >= -125.0)]
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

                print('Wrote {} records from {}'.format(len(df), os.path.basename(filename)))

        except EOFError as e:
            print('{}: {}'.format(os.path.basename(filename), e))
            continue

    for k, v in data.items():
        d = os.path.join(output_directory, k)
        if not os.path.exists(d):
            os.makedirs(d)
        f = os.path.join(d, '{}_{}.csv'.format(k, dt[:4]))
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


def process_time_chunk(time_tuple):
    start_time, end_time = time_tuple
    download_and_extract(dataset, start_time, end_time, madis_data_dir, username, password)

    mesonet_dir = os.path.join(madis_data_dir, 'LDAD', 'mesonet', 'netCDF')
    out_dir = os.path.join(madis_data_dir, 'LDAD', 'mesonet', 'csv')
    outshp = os.path.join(madis_data_dir, 'LDAD', 'mesonet', 'integrated_mesonet_{}.shp'.format(start_time))
    read_madis_hourly(mesonet_dir, start_time[:6], out_dir, shapefile=outshp)


if __name__ == "__main__":

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    username, password = 'usr', 'pswd'
    madis_data_dir = os.path.join(d, 'climate', 'madis')

    # the FTP we're currently using is from 2001-07-01
    times = generate_monthly_time_tuples(2001, 2023)
    times = [t for t in times if int(t[0][:6]) >= 200107]
    num_processes = 3

    for dataset in DATASET_PATHS.keys():
        print(f"Processing dataset: {dataset} with {num_processes} processes")

        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(process_time_chunk, times)
# ========================= EOF ====================================================================
