import glob
import gzip
import os
import tempfile
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
import warnings
import json
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=xr.SerializationWarning)

SUBHOUR_RESAMPLE_MAP = {'relHumidity': 'mean',
                        'precipAccum': 'sum',
                        'solarRadiation': 'mean',
                        'temperature': 'mean',
                        'windSpeed': 'mean'}



def read_madis_hourly(data_directory, date, output_directory):
    file_pattern = os.path.join(data_directory, f"*{date}*.gz")
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        print(f"No files found for date: {date}")
        return
    required_vars = ['stationId', 'relHumidity', 'precipAccum', 'solarRadiation', 'temperature',
                     'windSpeed', 'latitude', 'longitude']

    first = True
    data, sites = {}, pd.DataFrame().to_dict()

    for filename in file_list:

        dt = os.path.basename(filename).split('.')[0].replace('_', '')
        data[dt] = {}

        with gzip.open(filename) as fp:
            ds = xr.open_dataset(fp)
            valid_data = ds[required_vars]
            df = valid_data.to_dataframe()
            df['stationId'] = df['stationId'].astype(str)
            df = df[(df['latitude'] < 49.1) & (df['latitude'] >= 25.0)]
            df = df[(df['longitude'] < -65) & (df['longitude'] >= -125.0)]
            if first:
                sites = df[['stationId', 'latitude', 'longitude']]
            df.drop(columns=['latitude', 'longitude'], inplace=True)
            df.dropna(how='any', inplace=True)
            df.set_index('stationId', inplace=True, drop=True)
            df = df.groupby(df.index).agg(SUBHOUR_RESAMPLE_MAP)
            df['v'] = df.apply(lambda row: [float(row[v]) for v in required_vars[1:6]], axis=1)
            df.drop(columns=required_vars[1:6], inplace=True)
            data[dt] = df.to_dict(orient='index')
            shp = os.path.join(output_directory, 'madis_coords.shp'.format(date))
            write_locations(sites, shp)
            first = False

    js = os.path.join(output_directory, '{}.json'.format(date))
    with open(js, 'w') as f:
        json.dump(data, f)


def write_locations(loc, shp):
    gdf = gpd.GeoDataFrame(loc, geometry=gpd.points_from_xy(loc.longitude, loc.latitude), crs='EPSG:4326')
    gdf.to_file(shp)


if __name__ == "__main__":
    mesonet_dir = '/home/dgketchum/data/IrrigationGIS/climate/madis/LDAD/mesonet/netCDF'
    out_dir = '/home/dgketchum/data/IrrigationGIS/climate/madis/LDAD/mesonet/csv'
    read_madis_hourly(mesonet_dir, '20191001', out_dir)

# ========================= EOF ====================================================================
