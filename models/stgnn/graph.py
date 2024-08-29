import json
import os
import torch
import pandas as pd

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import networkx as nx

from tsl.data.datamodule.splitters import Splitter, disjoint_months
from tsl.data.synch_mode import HORIZON
from tsl.datasets.prototypes import DatetimeDataset
from tsl.datasets.prototypes.mixin import MissingValuesMixin
from tsl.utils import download_url, extract_zip

from models.stgnn import SIMILARITY_COLS


from tsl.datasets import AirQuality

dataset = AirQuality(root='./data')

print(dataset)


class Meteorology(DatetimeDataset, MissingValuesMixin):
    similarity_options = {'distance'}

    def __init__(self,
                 target_parameter,
                 root=None,
                 freq='D',
                 ):

        self.root = root

        df, mask, eval_mask, dist = self.load()
        super().__init__(target=df,
                         mask=mask,
                         freq=freq,
                         similarity_score='distance',
                         temporal_aggregation='mean',
                         spatial_aggregation='mean',
                         default_splitting_method='val',
                         name=target_parameter)

        self.set_eval_mask(eval_mask)

    def generate_edge_index(self):
        gdf = gpd.read_file(self.fields)
        gdf.index = gdf[self.index_col]
        if self.bounds:
            w, s, e, n = self.bounds
            gdf = gdf[(gdf['latitude'] < n) & (gdf['latitude'] >= s)]
            gdf = gdf[(gdf['longitude'] < e) & (gdf['longitude'] >= w)]

        if self.valid_target_stations:
            with open(self.valid_target_stations, 'r') as f:
                select = json.load(f)
                gdf = gdf.loc[select['stations']]
        coordinates = np.array([[point.x, point.y] for point in gdf.geometry])
        distances = []
        for i in range(len(coordinates)):
            distances.append([
                haversine_distance(coordinates[i][0], coordinates[i][1],
                                   coordinates[j][0], coordinates[j][1])
                for j in range(len(coordinates))
            ])
        distances = np.array(distances)

        k_nearest_indices = np.argsort(distances, axis=1)[:, 1:self.k_nearest + 1]
        row_indices = np.repeat(np.arange(len(gdf)), self.k_nearest)
        spatial_edges = np.column_stack([row_indices, k_nearest_indices.ravel()])

        try:
            features = gdf[['ELEV', 'longitude', 'latitude']].values
        except KeyError:
            features = gdf[['STATION_EL', 'STATION_LO', 'STATION_LA']].values
        scaler = MinMaxScaler()
        normalized_features = np.column_stack([
            scaler.fit_transform(features[:, i].reshape(-1, 1)) for i in range(features.shape[1])
        ])
        similarity_matrix = cosine_similarity(normalized_features)
        k_nearest_feature_indices = np.argsort(-similarity_matrix, axis=1)[:, 1:self.k_nearest + 1]
        row_indices = np.repeat(np.arange(len(gdf)), self.k_nearest)
        feature_edges = np.column_stack([row_indices, k_nearest_feature_indices.ravel()])
        all_edges = np.unique(np.vstack([spatial_edges, feature_edges]), axis=0)
        combined_edges = all_edges[all_edges[:, 0] != all_edges[:, 1]]
        G = nx.Graph()
        G.add_edges_from(combined_edges)
        if not nx.is_connected(G):
            for component in list(nx.connected_components(G)):
                main_component = list(nx.connected_components(G))[0]
                if component != main_component:
                    node_from_main = list(main_component)[0]
                    node_from_component = list(component)[0]
                    G.add_edge(node_from_main, node_from_component)
        combined_edges = np.array(G.edges)
        index_to_staid = {i: staid for i, staid in enumerate(gdf[self.index_col])}
        edge_lines = []
        to_, from_ = [], []
        for i, j in combined_edges:
            point1 = Point(gdf.iloc[i].geometry)
            point2 = Point(gdf.iloc[j].geometry)
            line = LineString([point1, point2])
            from_.append(gdf.iloc[i][self.index_col])
            to_.append(gdf.iloc[j][self.index_col])
            edge_lines.append(line)
        gdf_edges = gpd.GeoDataFrame({'geometry': edge_lines})
        gdf_edges['to'] = to_
        gdf_edges['from'] = from_

        with open(os.path.join(self.output_dir, 'edge_indx_map.json'), 'w') as f:
            json.dump(index_to_staid, f)

        np.savetxt(os.path.join(self.output_dir, 'edge_indx.np'), combined_edges)

        gdf_edges.to_file(os.path.join(self.output_dir, 'edges.shp'), epsg='EPSG:4326', engine='fiona')

        return combined_edges, gdf, index_to_staid


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

    # stations_ = os.path.join(d, 'met', 'stations', 'openet_gridwxcomp_input.shp')
    stations_ = os.path.join(clim, 'madis', 'mesonet_sites.shp')
    # output_dir_ = os.path.join(d, 'graphs', 'gwx')
    output_dir_ = os.path.join(d, 'graphs', 'mesonet')
    meteorology_dir_ = os.path.join(d, 'met', 'tables', 'obs_grid')

    node_prep = Graph(stations_, meteorology_dir_, output_dir_, k_nearest=5, index_col='index',
                      bounds=(-116., 46., -111., 49.))
    node_prep.generate_edge_index()
    # node_prep.preprocess_data()

# ========================= EOF ====================================================================
