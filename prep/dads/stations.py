import os

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from models.dads import SIMILARITY_COLS

l = ['lat', 'lon', 'B10', 'nd', 'slope', 'aspect', 'elevation', 'tpi_1250', 'tpi_250', 'tpi_150', 'rsun']


def get_stations(stations, csv_dir, out_csv, bounds=None):
    stations = gpd.read_file(stations)
    stations.index = stations['fid']
    stations.sort_index(inplace=True)

    if bounds:
        w, s, e, n = bounds
        stations = stations[(stations['latitude'] < n) & (stations['latitude'] >= s)]
        stations = stations[(stations['longitude'] < e) & (stations['longitude'] >= w)]

    first, df, data = True, None, None
    for i, (f, row) in enumerate(stations.iterrows(), start=1):

        file_ = os.path.join(csv_dir, '{}.csv'.format(f))
        if not os.path.exists(file_):
            continue

        data = pd.read_csv(file_)
        data = data[SIMILARITY_COLS.keys()]
        data['fid'] = f
        data = data.groupby('fid').agg(SIMILARITY_COLS)
        data['train'] = row['train']
        if first:
            df = data.copy()
            first = False
        else:
            df = pd.concat([df, data.copy()], ignore_index=False)

        if i % 100 == 0:
            print(i)

    df.to_csv(out_csv)
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.to_file(out_csv.replace('.csv', '.shp'), crs='EPSG:4326', engine='fiona')
    gdf.drop(columns=['geometry'], inplace=True)
    print(out_csv)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    target_var = 'mean_temp'

    glob_ = 'dads_stations_res_elev_mgrs_split'

    fields = os.path.join(d, 'met', 'stations', '{}.shp'.format(glob_))
    landsat_ = os.path.join(d, 'rs', 'dads_stations', 'landsat', 'station_data')
    solrad = os.path.join(d, 'dem', 'rsun_tables')

    zoran = '/home/dgketchum/training'
    nvm = '/media/nvm/training'

    if os.path.exists(zoran):
        print('reading from zoran')
        training = zoran
    elif os.path.exists(nvm):
        print('reading from nvm drive')
        training = nvm
    else:
        print('reading from UM drive')
        training = os.path.join(d, 'training')

    # TODO remove this confusing dependence on getting RS and terrain data from lstm training data
    csv_dir_ = os.path.join(training, 'simple_lstm', target_var, 'compiled_csv')

    out_csv_ = os.path.join(training, 'dads', 'graph', 'stations.csv')

    get_stations(fields, csv_dir_, out_csv_, bounds=None)

# ========================= EOF ===============================================================================
