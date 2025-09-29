import os
from tqdm import tqdm

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from prep.columns_desc import GEO_FEATURES
from utils.station_parameters import station_par_map

GRAPH_FEATURES = ['lat', 'lon', 'B10', 'nd', 'slope', 'aspect',
                  'elevation', 'tpi_1250', 'tpi_250', 'tpi_150', 'rsun']


def get_stations(stations, csv_dir, out_csv, source='madis', bounds=None):

    kw = station_par_map(source)

    stations = pd.read_csv(stations, index_col=kw['index'])
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

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    target_var = 'tmax_obs'

    _source = 'madis'

    if _source == 'madis':
        glob_ = 'madis_02JULY2025_mgrs'
        fields = os.path.join(d, 'dads', 'met', 'stations', '{}.csv'.format(glob_))

    elif _source == 'ghcn':
        glob_ = 'ghcn_CANUSA_stations_mgrs'
        fields = os.path.join(d, 'climate', 'ghcn', 'stations', '{}.csv'.format(glob_))

    else:
        raise ValueError()

    training = '/data/ssd2/dads/training'

    csv_dir_ = os.path.join(training, 'parquet', target_var)
    out_csv_ = os.path.join(training, 'graph', 'stations.csv')

    get_stations(fields, csv_dir_, out_csv_, bounds=None, source=_source)

# ========================= EOF ===============================================================================
