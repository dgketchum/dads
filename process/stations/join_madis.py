import os

import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from utils.elevation import elevation_from_coordinate


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


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    stations = os.path.join(d, 'dads', 'met', 'stations', 'dads_stations_elev_mgrs.csv')
    stations_out = os.path.join(d, 'dads', 'met', 'stations', 'dads_stations_res_elev_mgrs.csv')
    flagged = os.path.join(d, 'dads', 'met', 'stations', 'madis_research_flagged.json')
    madis_shapes = os.path.join(d, 'climate', 'madis', 'LDAD', 'mesonet', 'shapes')
    mgrs_ = os.path.join(d, 'boundaries', 'mgrs', 'MGRS_100km_world.shp')
    process_weather_stations(stations, madis_shapes, flagged, mgrs_, stations_out)

# ========================= EOF ====================================================================
