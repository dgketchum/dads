import os
from tqdm import tqdm

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from prep.columns_desc import GEO_FEATURES

GRAPH_FEATURES = ['lat', 'lon', 'B10', 'nd', 'slope', 'aspect',
                  'elevation', 'tpi_1250', 'tpi_250', 'tpi_150', 'rsun']


def get_stations(stations, csv_dir, out_csv, bounds=None):
    stations = gpd.read_file(stations)
    stations.index = stations['fid']
    stations.sort_index(inplace=True)

    if bounds:
        w, s, e, n = bounds
        stations = stations[(stations['latitude'] < n) & (stations['latitude'] >= s)]
        stations = stations[(stations['longitude'] < e) & (stations['longitude'] >= w)]

    first, df, data, dct = True, None, None, {}
    for f, row in tqdm(stations.iterrows(), total=len(stations)):
        file_ = os.path.join(csv_dir, '{}.parquet'.format(f))
        if not os.path.exists(file_):
            continue

        data = pd.read_parquet(file_)
        data = data[GEO_FEATURES]
        _len = data.shape[0]
        data = data.mean()
        data['records'] = _len
        data['train'] = np.random.choice([0, 1], p=[0.2, 0.8])
        dct[f] = data.to_dict()

    df = pd.DataFrame.from_dict(dct, orient='index')
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

    glob_ = 'dads_stations_10FEB2025'

    fields = os.path.join(d, 'met', 'stations', '{}.shp'.format(glob_))
    landsat_ = os.path.join(d, 'rs', 'dads_stations', 'landsat', 'station_data')
    solrad = os.path.join(d, 'dem', 'rsun_tables')

    zoran = '/data/ssd2/dads/training'
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

    csv_dir_ = os.path.join(training, 'parquet')
    out_csv_ = os.path.join(training, 'graph', 'stations.csv')

    get_stations(fields, csv_dir_, out_csv_, bounds=None)

# ========================= EOF ===============================================================================
