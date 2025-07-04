import os
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import concurrent.futures
from tqdm import tqdm


def process_station_directory(station_dir):
    station_id = os.path.basename(station_dir)

    parquet_files = glob.glob(os.path.join(station_dir, '*.parquet'))
    if not parquet_files:
        return None

    df_list = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(df_list)

    if 'datetime' in df.columns:
        df.index = pd.to_datetime(df['datetime'])
        df = df.drop(columns=['datetime'])

    if df.empty:
        return None

    required_meta_cols = ['latitude', 'longitude', 'elevation']
    if not all(col in df.columns for col in required_meta_cols):
        return None

    lat = df['latitude'].iloc[0]
    lon = df['longitude'].iloc[0]
    elev = df['elevation'].iloc[0]

    start_date = df.index.min()
    end_date = df.index.max()

    prcp_is_zero = False
    if 'precipAccum' in df.columns and df['precipAccum'].notna().any():
        prcp_is_zero = (df['precipAccum'][df['precipAccum'].notna()] == 0).all()

    cols_to_count = ['precipAccum', 'solarRadiation', 'temperature',
                     'dewpoint', 'relHumidity', 'windSpeed', 'windDir']

    summary = {
        'station_id': station_id,
        'latitude': lat,
        'longitude': lon,
        'elevation': elev,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'prcp_zero': prcp_is_zero
    }

    short_names = {
        'precipAccum': 'prcp_ct',
        'solarRadiation': 'srad_ct',
        'temperature': 'temp_ct',
        'dewpoint': 'dewpt_ct',
        'relHumidity': 'rh_ct',
        'windSpeed': 'wspd_ct',
        'windDir': 'wdir_ct'
    }

    for col in cols_to_count:
        count_col_name = short_names[col]
        if col in df.columns:
            summary[count_col_name] = df[col].notna().sum()
        else:
            summary[count_col_name] = 0

    return summary


def create_station_summary_shapefile(input_dir, output_shp, n_workers=None):
    station_dirs = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if
                    os.path.isdir(os.path.join(input_dir, d))]

    if not station_dirs:
        return

    station_summaries = []

    if n_workers == 1:
        ct = 0
        for d in station_dirs:
            result = process_station_directory(d)
            if result:
                station_summaries.append(result)
                ct += 1
            if ct > 10:
                break
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_dir = {executor.submit(process_station_directory, d): d for d in station_dirs}

            for future in tqdm(concurrent.futures.as_completed(future_to_dir), total=len(station_dirs),
                               desc="Processing Stations"):
                result = future.result()
                if result:
                    station_summaries.append(result)

    if not station_summaries:
        return

    summary_df = pd.DataFrame(station_summaries)

    initial_count = len(summary_df)
    summary_df.dropna(subset=['latitude', 'longitude'], inplace=True)
    summary_df = summary_df[np.isfinite(summary_df['latitude']) & np.isfinite(summary_df['longitude'])]

    if len(summary_df) < initial_count:
        print(f"Dropped {initial_count - len(summary_df)} stations with invalid coordinates.")

    if summary_df.empty:
        print("No valid stations remaining after cleaning coordinates. Exiting.")
        return

    geometry = [Point(xy) for xy in zip(summary_df['longitude'], summary_df['latitude'])]
    geo_df = gpd.GeoDataFrame(summary_df, geometry=geometry, crs="EPSG:4326")

    output_dir = os.path.dirname(output_shp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    geo_df.to_file(output_shp, driver='ESRI Shapefile')


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS')

    daily_data_dir = '/data/ssd2/madis/extracts'

    output_shapefile_path = os.path.join(d, 'dads', 'met', 'stations', 'station_summary.shp')

    num_workers = 30

    create_station_summary_shapefile(input_dir=daily_data_dir,
                                     output_shp=output_shapefile_path,
                                     n_workers=num_workers)

# ========================= EOF ====================================================================
