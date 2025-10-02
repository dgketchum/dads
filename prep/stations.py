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


def merge_shapefiles(shapefiles):
    frames = []
    for shp in shapefiles:
        shp_lower = os.path.basename(shp).lower()
        if 'ghcn' in shp_lower:
            stype = 'ghcn'
        elif 'madis' in shp_lower:
            stype = 'madis'
        else:
            stype = 'dads'

        mp = station_par_map(stype)
        gdf = gpd.read_file(shp)
        df = gdf.copy()

        if 'fid' not in df.columns and mp.get('index') in df.columns:
            df['fid'] = df[mp['index']].astype(str)
        if 'latitude' not in df.columns and mp.get('lat') in df.columns:
            df['latitude'] = df[mp['lat']]
        if 'longitude' not in df.columns and mp.get('lon') in df.columns:
            df['longitude'] = df[mp['lon']]
        if 'elevation' not in df.columns and mp.get('elev') in df.columns:
            df['elevation'] = df[mp['elev']]

        if 'lat' not in df.columns and 'latitude' in df.columns:
            df['lat'] = df['latitude']
        if 'lon' not in df.columns and 'longitude' in df.columns:
            df['lon'] = df['longitude']

        frames.append(df)

    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs if frames else None)
    return merged


def get_station_observation_metadata(parquet_root, obs_vars, stations_gdf, out_shp, top_percent=0.8):
    counts = {}
    for var in obs_vars:
        var_dir = os.path.join(parquet_root, var)
        if not os.path.isdir(var_dir):
            continue
        for f in os.listdir(var_dir):
            if not f.endswith('.parquet'):
                continue
            stn = os.path.splitext(f)[0]
            p = os.path.join(var_dir, f)
            try:
                n = pd.read_parquet(p).shape[0]
            except Exception:
                n = 0
            d = counts.get(stn, {})
            d[var + '_count'] = d.get(var + '_count', 0) + n
            counts[stn] = d

    idx = [str(i) for i in stations_gdf.get('fid', stations_gdf.index).astype(str)]
    df_counts = pd.DataFrame.from_dict(counts, orient='index')
    df_counts.index = df_counts.index.astype(str)
    df_counts = df_counts.reindex(idx)
    df_counts = df_counts.fillna(0).astype(int)
    df_counts['obs_count_total'] = df_counts.sum(axis=1)

    gdf = stations_gdf.copy()
    gdf['fid'] = gdf['fid'].astype(str)
    gdf = gdf.merge(df_counts, how='left', left_on='fid', right_index=True)
    gdf['obs_count_total'] = gdf['obs_count_total'].fillna(0).astype(int)

    gdf.to_file(out_shp, crs='EPSG:4326', engine='fiona')
    return gdf


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
