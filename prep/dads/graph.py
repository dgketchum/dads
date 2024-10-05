import json
import os
import pandas as pd

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import torch
import networkx as nx
from fiona.crs import CRS


class Graph:
    def __init__(self, stations, output_dir, k_nearest=2, bounds=None):
        self.fields = stations
        self.valid_target_stations = None
        self.output_dir = output_dir
        self.k_nearest = k_nearest
        self.pth_stations = []
        self.bounds = bounds

    def generate_edge_index(self):
        gdf = gpd.read_file(self.fields)
        gdf.index = list(range(gdf.shape[0]))

        if self.bounds:
            w, s, e, n = self.bounds
            gdf = gdf[(gdf['latitude'] < n) & (gdf['latitude'] >= s)]
            gdf = gdf[(gdf['longitude'] < e) & (gdf['longitude'] >= w)]

        if self.valid_target_stations:
            with open(self.valid_target_stations, 'r') as f:
                select = json.load(f)
                gdf = gdf.loc[select['stations']]

        gdf = gdf[['fid', 'train', 'geometry']]
        train_gdf = gdf[gdf['train'] == 1]
        val_gdf = gdf[gdf['train'] == 0]

        train_coords = np.array([[point.y, point.x] for point in train_gdf.geometry])
        val_coords = np.array([[point.y, point.x] for point in val_gdf.geometry])

        distances = []
        for i in range(len(train_coords)):
            distances.append([
                haversine_distance(train_coords[i][0], train_coords[i][1],
                                   val_coords[j][0], val_coords[j][1]) for j in range(len(val_coords))])
        distances = np.array(distances)

        k_indices = np.argsort(distances, axis=0)[:self.k_nearest, :]
        train_index_map = {i: j for i, j in enumerate(train_gdf.index)}
        k_indices = np.vectorize(train_index_map.get)(k_indices.ravel()).reshape(k_indices.shape)

        row_indices = np.repeat(np.arange(len(val_gdf)), self.k_nearest)
        spatial_edges = np.column_stack([row_indices, k_indices.T.ravel()])

        val_index_map = {i: j for i, j in enumerate(val_gdf.index)}
        spatial_edges[:, 0] = np.vectorize(val_index_map.get)(spatial_edges[:, 0])

        to_from = [1, 0]
        spatial_edges = spatial_edges[:, to_from]

        index_to_staid = {i: staid for i, staid in enumerate(gdf['fid'])}
        edge_lines = []
        to_, from_ = [], []

        for i, j in spatial_edges:
            point1 = Point(gdf.iloc[i].geometry)
            point2 = Point(gdf.iloc[j].geometry)
            line = LineString([point1, point2])

            from_.append(gdf.iloc[i]['fid'])
            to_.append(gdf.iloc[j]['fid'])
            edge_lines.append(line)

        gdf_edges = gpd.GeoDataFrame({'geometry': edge_lines})
        gdf_edges['to'] = to_
        gdf_edges['from'] = from_

        with open(os.path.join(self.output_dir, 'edge_indx_map.json'), 'w') as f:
            json.dump(index_to_staid, f)

        np.savetxt(os.path.join(self.output_dir, 'edge_indx.np'), spatial_edges)

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

    node_prep = Graph(stations_, output_dir_, k_nearest=5)
    node_prep.generate_edge_index()
    # node_prep.preprocess_data()

# ========================= EOF ====================================================================
