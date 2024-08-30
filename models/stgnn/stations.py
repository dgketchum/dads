import os

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from models.stgnn import SIMILARITY_COLS

def get_stations(stations, csv_dir, out_csv, bounds=None, validation_frac=0.2, validation_stations=None):
    stations = pd.read_csv(stations)
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
        if first:
            df = data.copy()
            first = False
        else:
            df = pd.concat([df, data.copy()], ignore_index=False)

    if validation_stations:
        df['val'] = 0
        df.loc[validation_stations, 'val'] = 1
    else:
        df['rand'] = [np.random.rand() for _ in range(len(df))]
        df['val'] = 0
        df.loc[df['rand'] > (1 - validation_frac), 'val'] = 1
        df.drop(columns=['rand'], inplace=True)

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

    target_var = 'vpd'

    glob_ = 'dads_stations_elev_mgrs'

    fields = os.path.join(d, 'met', 'stations', '{}.csv'.format(glob_))
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

    param_dir = os.path.join(training, target_var)
    csv_dir_ = os.path.join(param_dir, 'compiled_csv')

    out_csv_ = os.path.join(param_dir, 'graph', 'stations.csv')

    bounds = (-116., 44., -110., 49.)

    get_stations(fields, csv_dir_, out_csv_, bounds=bounds, validation_frac=1.0)

# ========================= EOF ===============================================================================
