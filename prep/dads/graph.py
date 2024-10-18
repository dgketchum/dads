import json
import os

import pandas as pd
import geopandas as gpd
import numpy as np
from fiona.crs import CRS
from shapely.geometry import LineString, Point

from models.dads import SIMILARITY_COLS
from sklearn.preprocessing import MinMaxScaler


class Graph:
    def __init__(self, stations, output_dir, k_nearest=2, bounds=None):
        self.fields = stations
        self.output_dir = output_dir
        self.k_nearest = k_nearest
        self.pth_stations = []
        self.bounds = bounds

    def generate_edge_index(self):
        stations = gpd.read_file(self.fields)
        stations.index = list(range(stations.shape[0]))

        if self.bounds:
            w, s, e, n = self.bounds
            stations = stations[(stations['latitude'] < n) & (stations['latitude'] >= s)]
            stations = stations[(stations['longitude'] < e) & (stations['longitude'] >= w)]

        attrs_select = list(SIMILARITY_COLS.keys()) + ['train']
        attributes = stations[attrs_select].copy()
        attributes.index = stations['fid']
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

        gdf = stations[['fid', 'train', 'geometry']].copy()
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

        index_to_staid = {i: staid for i, staid in enumerate(gdf['fid'])}
        edge_lines, train = [], []
        to_, from_ = [], []

        for e, (i, j) in enumerate(spatial_edges):

            from_fid = gdf.iloc[i]['fid']
            to_fid = gdf.iloc[j]['fid']

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

    d = '/media/research/IrrigationGIS/dads'
    clim = '/media/research/IrrigationGIS/climate'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'
        clim = '/home/dgketchum/data/IrrigationGIS/climate'

    stations_ = '/media/nvm/training/dads/graph/stations.shp'
    output_dir_ = '/media/nvm/training/dads/graph'

    node_prep = Graph(stations_, output_dir_, k_nearest=10)
    node_prep.generate_edge_index()

# ========================= EOF ====================================================================
