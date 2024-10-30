import os
import glob
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.utils import shuffle

from utils.elevation import elevation_from_coordinate
from process.obs.write_madis import METADATA, open_nc


def process_weather_stations(stations_csv, shapefiles_dir, flag_file, mgrs, out_csv):
    sdf = pd.read_csv(stations_csv, index_col='fid')
    gdf = gpd.GeoDataFrame(sdf, geometry=gpd.points_from_xy(sdf.longitude, sdf.latitude))

    sdf['privileged'] = 0

    mgrs_gdf = gpd.read_file(mgrs)

    shapefile_list = [f for f in os.listdir(shapefiles_dir) if f.endswith('.shp')]
    flagged_stations, ct = {}, 0

    for i, shp in enumerate(shapefile_list, start=1):

        print('{} of {}; {}'.format(i, len(shapefile_list), os.path.basename(shp)))

        shapefile_path = os.path.join(shapefiles_dir, shp)
        new_stations_gdf = gpd.read_file(shapefile_path)

        for index, row in new_stations_gdf.iterrows():
            existing_station = sdf[sdf.index == row['index']]

            if not existing_station.empty:
                lat1, lon1 = (existing_station['latitude'].iloc[0], existing_station['longitude'].iloc[0])
                lat2, lon2 = (row['latitude'], row['longitude'])
                distance = haversine_distance(lat1, lon1, lat2, lon2) * 1000

                if distance > 250:
                    if row['index'] in flagged_stations:
                        flagged_stations[row['index']].append((shp[19:27], distance))
                    else:
                        flagged_stations[row['index']] = [(shp[19:27], distance)]
            else:
                try:
                    point = Point(row['longitude'], row['latitude'])
                    mgrs_tile = mgrs_gdf[mgrs_gdf.contains(point)]['MGRS'].values[0]

                    elev = elevation_from_coordinate(row['latitude'], row['longitude'])
                    new_record = {'index': row['index'],
                                  'name': 'None',
                                  'latitude': row['latitude'],
                                  'longitude': row['longitude'],
                                  'elevation': elev,
                                  'fid': row['index'],
                                  'orig_netid': row['index'],
                                  'privileged': 1,
                                  'MGRS_TILE': mgrs_tile}

                    sdf.loc[row['index']] = new_record

                    print('add station {}; {:.2f}, {:.2f} at {:.2f} m'.format(row['index'], row['latitude'],
                                                                              row['longitude'], elev))
                    ct += 1

                except Exception as e:
                    print(index, e)
                    continue

    with open(flag_file, 'w') as f:
        json.dump(flagged_stations, f, indent=4)

    sdf.to_csv(out_csv, index=False)
    geometry = [Point(xy) for xy in zip(sdf['longitude'], sdf['latitude'])]
    gdf = gpd.GeoDataFrame(sdf, geometry=geometry)
    gdf.to_file(out_csv.replace('.csv', '.shp'), crs='EPSG:4326', engine='fiona')
    print(out_csv, ct, 'added stations')


def haversine_distance(lat1_, lon1_, lat2_, lon2_):
    R = 6371
    dlat = np.radians(lat2_ - lat1_)
    dlon = np.radians(lon2_ - lon1_)
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + \
        np.cos(np.radians(lat1_)) * np.cos(np.radians(lat2_)) * \
        np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


def get_station_metadata(data_directory, csv_dir, station_tracker):
    station_dct, stations = {}, set()

    station_yearmonths = defaultdict(list)
    for station_dir in os.listdir(csv_dir):
        station_path = os.path.join(csv_dir, station_dir)
        if os.path.isdir(station_path):
            for csv_file in os.listdir(station_path):
                if csv_file.endswith(".csv"):
                    yearmonth = csv_file.split('_')[1].split('.')[0]
                    station_yearmonths[station_dir].append(yearmonth)

    yearmonth_counts = defaultdict(int)
    for yearmonths in station_yearmonths.values():
        for yearmonth in yearmonths:
            yearmonth_counts[yearmonth] += 1
    sorted_yearmonths = sorted(yearmonth_counts, key=yearmonth_counts.get, reverse=True)

    file_pattern = os.path.join(data_directory, f"*.gz")
    file_list = sorted(glob.glob(file_pattern))
    print(f'{len(file_list)} files')

    for yearmonth in sorted_yearmonths:
        yearmo_files = [f for f in file_list if yearmonth in f]
        yearmo_files = shuffle(yearmo_files)
        for j, filename in enumerate(yearmo_files):
            try:
                ds = open_nc(filename)
                if ds is None:
                    print(f"Skipping file {filename}: Could not open.")
                    continue

                p = ['stationId'] + METADATA
                valid_data = ds[p]
                df = valid_data.to_dataframe()
                df['stationId'] = df['stationId'].astype(str)
                df.index = df['stationId']
                df.dropna(how='all', inplace=True)
                new_stations = [s for s in df.index if s not in stations]

                if len(new_stations) > 0:
                    print(f'Adding {len(new_stations)} new stations from {filename}')
                    add_stn = df.loc[new_stations]
                    station_dct.update({
                        i: {'lat': r['latitude'],
                            'lon': r['longitude'],
                            'elev': r['elevation'],
                            'stype': r['stationType'].decode('utf-8')}
                        for i, r in add_stn.iterrows()
                    })
                    stations.update(new_stations)
                    with open(station_tracker, 'w') as fp:
                        json.dump(station_dct, fp, indent=4)

                if stations == set(station_yearmonths.keys()):
                    print("Metadata for all stations found.")
                    return

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    print("Finished processing all files.")


def write_stations_to_shapefile(station_tracker, shapefile_path):
    with open(station_tracker, 'r') as f:
        station_dct = json.load(f)
    print(len(station_dct))
    data = []
    for station_id, info in station_dct.items():

        geo_ = Point(info['lon'], info['lat'])
        stype = info['stype']
        entry = {
            'fid': station_id,
            'latitude': info['lat'],
            'longitude': info['lon'],
            'elev': info['elev'],
            'stype': stype,
            'geometry': geo_,
        }
        entry = validate_entry(station_id, entry)

        if entry:
            data.append(entry)

    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    print(gdf.shape[0], 'shapefile')
    gdf.to_file(shapefile_path)
    df = gdf[[c for c in gdf.columns if c != 'geometry']]
    df.to_csv(shapefile_path.replace('.shp', '.csv'))


def validate_entry(station_id, info):
    try:
        if not all(key in info for key in ('latitude', 'longitude', 'elev', 'stype')):
            raise ValueError("Missing required key in station information")

        lat = float(info['latitude'])
        if not -90 <= lat <= 90:
            raise ValueError("Invalid latitude value")

        lon = float(info['longitude'])
        if not -180 <= lon <= 180:
            raise ValueError("Invalid longitude value")

        elev = float(info['elev'])
        stype = str(info['stype'])
        geo_ = Point(lon, lat)

        entry = {
            'fid': station_id,
            'latitude': lat,
            'longitude': lon,
            'elev': elev,
            'stype': stype,
            'geometry': geo_,
        }

        return entry

    except (ValueError, TypeError) as e:
        print(f"Error validating station {station_id}: {e}")
        return None

def write_missing(madis, dads, missing):
    sdf = gpd.read_file(madis)
    sdf.index = sdf['fid']

    ddf = gpd.read_file(dads)
    ddf.index = ddf['fid']
    missing_idx = [i for i in sdf.index if i not in ddf.index]

    sdf = sdf.loc[missing_idx]
    sdf.index = [i for i in range(sdf.shape[0])]
    sdf.to_file(missing)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    mesonet_dir = os.path.join(d, 'climate', 'madis', 'LDAD', 'mesonet')
    netcdf_src = os.path.join(mesonet_dir, 'netCDF')

    tracker_ = os.path.join('/data/ssd1/madis', 'stations.json')
    out_dir_ = os.path.join('/data/ssd1/madis', 'inclusive_csv')

    shp = os.path.join('/data/ssd1/madis', 'madis_shapefile_29OCT2024.shp')
    # get_station_metadata(netcdf_src, out_dir_, tracker_)
    write_stations_to_shapefile(tracker_, shp)

    stations = os.path.join(d, 'dads', 'met', 'stations', 'dads_stations_elev_mgrs.csv')
    flagged = os.path.join(d, 'dads', 'met', 'stations', 'madis_research_flagged.json')
    madis_shapes = os.path.join(d, 'climate', 'madis', 'LDAD', 'mesonet', 'shapes')
    mgrs_ = os.path.join(d, 'boundaries', 'mgrs', 'MGRS_100km_world.shp')
    # process_weather_stations(stations, madis_shapes, flagged, mgrs_, stations_out)

    shp = os.path.join('/home/dgketchum/Downloads', 'madis_shapefile.shp')
    stations_out = os.path.join(d, 'dads', 'met', 'stations', 'dads_stations_res_elev_mgrs.shp')
    shp_out = os.path.join('/home/dgketchum/Downloads', 'madis_28OCT2024.shp')
    # write_missing(shp, stations_out, shp_out)

# ========================= EOF ====================================================================
