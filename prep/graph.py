import json
import os

import pandas as pd
import geopandas as gpd
import numpy as np
from fiona.crs import CRS
from shapely.geometry import LineString, Point

from prep.columns_desc import GEO_FEATURES
from sklearn.preprocessing import MinMaxScaler
from utils.station_parameters import station_par_map
from prep.stations import merge_shapefiles, get_station_observation_metadata


class Graph:
    def __init__(self, stations, output_dir, k_nearest=2, bounds=None, index_col='fid',
                 split_mode='observations', split_percent=0.8, random_state=None):
        self.fields = stations
        self.output_dir = output_dir
        self.k_nearest = k_nearest
        self.pth_stations = []
        self.bounds = bounds
        self.index_col = index_col
        self.split_mode = split_mode
        self.split_percent = split_percent
        self.random_state = random_state

        # assign train/val split immediately
        if isinstance(self.fields, gpd.GeoDataFrame):
            self.fields = self._assign_train_split(self.fields)
        else:
            gdf_ = gpd.read_file(self.fields)
            self.fields = self._assign_train_split(gdf_)

    def _assign_train_split(self, stations):
        gdf = stations.copy()
        n = gdf.shape[0]
        if self.split_mode == 'observations' and 'obs_count_total' in gdf.columns:
            k = int(max(1, round(n * self.split_percent)))
            order = gdf['obs_count_total'].rank(method='first', ascending=False)
            gdf['train'] = (order <= k).astype(int)
        else:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            k = int(max(1, round(n * self.split_percent)))
            idx = np.arange(n)
            np.random.shuffle(idx)
            train_idx = set(idx[:k])
            gdf['train'] = [1 if i in train_idx else 0 for i in range(n)]
        return gdf

    def generate_edge_index(self):
        if isinstance(self.fields, gpd.GeoDataFrame):
            stations = self.fields.copy()
        else:
            stations = gpd.read_file(self.fields)
        stations.index = list(range(stations.shape[0]))

        if self.bounds:
            w, s, e, n = self.bounds
            stations = stations[(stations['latitude'] < n) & (stations['latitude'] >= s)]
            stations = stations[(stations['longitude'] < e) & (stations['longitude'] >= w)]

        attrs_select = [c for c in list(GEO_FEATURES) + ['train', self.index_col] if c in stations.columns]  # allow partial
        attributes = stations[attrs_select].copy()
        attributes.index = attributes[self.index_col]

        df_copy = attributes.copy()
        for col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        attributes = df_copy.copy()

        scaler = MinMaxScaler()
        attributes = pd.DataFrame(scaler.fit_transform(attributes),
                                  columns=attributes.columns,
                                  index=attributes.index)
        val_dct, train_dct = {}, {}
        for i, r in attributes.iterrows():
            t = r['train']
            if t:
                train_dct[i] = r.to_list()[:-1]

            # put all in val dict, since training stations' nodes are sent to validation nodes for now
            val_dct[i] = r.to_list()[:-1]

        gdf = stations[[self.index_col, 'train', 'geometry']].copy()
        train_gdf = gdf[gdf['train'] == 1]
        val_gdf = gdf[gdf['train'] == 0]

        train_coords = np.array([[point.y, point.x] for point in train_gdf.geometry])
        val_coords = np.array([[point.y, point.x] for point in val_gdf.geometry])

        # bipartite validation edge set (i.e., from training to validation nodes)
        distance_val = []
        for i in range(len(train_coords)):
            distance_val.append([
                haversine_distance(train_coords[i][0], train_coords[i][1],
                                   val_coords[j][0], val_coords[j][1]) for j in range(len(val_coords))])
        distance_val = np.array(distance_val)
        k_ind_v = np.argsort(distance_val, axis=0)[:self.k_nearest, :]
        train_index_map = {i: j for i, j in enumerate(train_gdf.index)}
        k_ind_v = np.vectorize(train_index_map.get)(k_ind_v.ravel()).reshape(k_ind_v.shape)
        row_indices_v = np.repeat(np.arange(len(val_gdf)), self.k_nearest)
        spatial_edges_v = np.column_stack([row_indices_v, k_ind_v.T.ravel()])
        val_index_map = {i: j for i, j in enumerate(val_gdf.index)}
        spatial_edges_v[:, 0] = np.vectorize(val_index_map.get)(spatial_edges_v[:, 0])

        # undirected train-to-train edge set
        distance = []
        for i in range(len(train_coords)):
            distance.append([
                haversine_distance(train_coords[i][0], train_coords[i][1],
                                   train_coords[j][0], train_coords[j][1]) for j in range(len(train_coords))])
        distance = np.array(distance)
        k_indices = np.argsort(distance, axis=0)[:self.k_nearest, :]
        train_index_map = {i: j for i, j in enumerate(train_gdf.index)}
        k_indices = np.vectorize(train_index_map.get)(k_indices.ravel()).reshape(k_indices.shape)
        row_indices = np.repeat(np.arange(len(train_gdf)), self.k_nearest)
        spatial_edges_t = np.column_stack([row_indices, k_indices.T.ravel()])
        train_index_map = {i: j for i, j in enumerate(train_gdf.index)}
        spatial_edges_t[:, 0] = np.vectorize(train_index_map.get)(spatial_edges_t[:, 0])

        spatial_edges = np.concatenate([spatial_edges_v, spatial_edges_t])

        to_from = [1, 0]
        spatial_edges = spatial_edges[:, to_from]

        index_to_staid = {i: staid for i, staid in enumerate(gdf[self.index_col])}
        edge_lines, train = [], []
        to_, from_ = [], []

        for e, (i, j) in enumerate(spatial_edges):

            from_fid = gdf.iloc[i][self.index_col]
            to_fid = gdf.iloc[j][self.index_col]

            point1 = Point(gdf.iloc[i].geometry)
            point2 = Point(gdf.iloc[j].geometry)
            line = LineString([point1, point2])

            from_.append(from_fid)
            to_.append(to_fid)

            edge_lines.append(line)

            if gdf.iloc[i]['train'] and gdf.iloc[j]['train']:
                train.append(1)
            else:
                train.append(0)

        gdf_edges = gpd.GeoDataFrame({'geometry': edge_lines})
        gdf_edges['to'] = to_
        gdf_edges['from'] = from_
        gdf_edges['train'] = train

        # TODO: limit the reach into validation stations when larger receptive field is implemented
        train_edges = gdf_edges.groupby('to')['from'].agg(list).to_dict()
        with open(os.path.join(self.output_dir, 'train_edge_index.json'), 'w') as f:
            json.dump(train_edges, f)

        with open(os.path.join(self.output_dir, 'train_edge_attr.json'), 'w') as f:
            json.dump(train_dct, f)

        # TODO: limit the reach into validation stations when larger receptive field is implemented
        val_edges = gdf_edges.groupby('to')['from'].agg(list).to_dict()
        with open(os.path.join(self.output_dir, 'val_edge_index.json'), 'w') as f:
            json.dump(val_edges, f)

        with open(os.path.join(self.output_dir, 'val_edge_attr.json'), 'w') as f:
            json.dump(val_dct, f)

        gdf.crs = CRS.from_epsg(4326)
        gdf_edges.crs = CRS.from_epsg(4326)

        gdf_edges.to_file(os.path.join(self.output_dir, 'edges.shp'), epsg='EPSG:4326', engine='fiona')

        return spatial_edges, gdf, index_to_staid

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + \
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * \
        np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    training = '/data/ssd2/dads/training'

    parquet_root = os.path.join(training, 'parquet')
    obs_vars = [v for v in os.listdir(parquet_root) if os.path.isdir(os.path.join(parquet_root, v))]

    madis_glob = 'madis_02JULY2025_mgrs'
    ghcn_glob = 'ghcn_CANUSA_stations_mgrs'
    madis_shp = os.path.join(d, 'dads', 'met', 'stations', f'{madis_glob}.shp')
    ghcn_shp = os.path.join(d, 'climate', 'ghcn', 'stations', f'{ghcn_glob}.shp')

    merged = merge_shapefiles([ghcn_shp, madis_shp])

    obs_meta_shp = os.path.join(training, 'graph', 'station_observations.shp')
    top_pct = 0.8
    stations_obs = get_station_observation_metadata(parquet_root, obs_vars, merged, obs_meta_shp, top_percent=top_pct)

    for target_var in obs_vars:
        output_dir_ = os.path.join(training, 'graph', target_var)
        os.makedirs(output_dir_, exist_ok=True)
        node_prep = Graph(stations_obs, output_dir_, k_nearest=10, index_col='fid',
                          split_mode='observations', split_percent=0.8, random_state=42)
        node_prep.generate_edge_index()

# ========================= EOF ====================================================================
