import json
import os

import geopandas as gpd
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, Point
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import networkx as nx


def generate_edge_index(fields, select_js, outfile, k_nearest=2):
    with open(select_js, 'r') as f:
        select = json.load(f)

    gdf = gpd.read_file(fields)
    gdf.index = gdf['STAID']
    gdf = gdf.loc[select['stations']]

    coordinates = np.array([[point.x, point.y] for point in gdf.geometry])
    distances = cdist(coordinates, coordinates, metric='euclidean')

    k_nearest_indices = np.argsort(distances, axis=1)[:, 1:k_nearest + 1]
    row_indices = np.repeat(np.arange(len(gdf)), k_nearest)
    spatial_edges = np.column_stack([row_indices, k_nearest_indices.ravel()])

    features = gdf[['ELEV', 'LON', 'LAT']].values
    scaler = MinMaxScaler()
    normalized_features = np.column_stack([
        scaler.fit_transform(features[:, i].reshape(-1, 1)) for i in range(features.shape[1])
    ])
    similarity_matrix = cosine_similarity(normalized_features)

    k_nearest_feature_indices = np.argsort(-similarity_matrix, axis=1)[:, 1:k_nearest + 1]
    row_indices = np.repeat(np.arange(len(gdf)), k_nearest)
    feature_edges = np.column_stack([row_indices, k_nearest_feature_indices.ravel()])

    all_edges = np.unique(np.vstack([spatial_edges, feature_edges]), axis=0)
    combined_edges = all_edges[all_edges[:, 0] != all_edges[:, 1]]

    # Ensure all nodes are connected using NetworkX
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

    index_to_staid = {i: staid for i, staid in enumerate(gdf['STAID'])}
    with open(outfile.replace('.np', '.json'), 'w') as f:
        json.dump(index_to_staid, f)

    edge_lines = []
    to_, from_ = [], []
    for i, j in combined_edges:
        point1 = Point(gdf.iloc[i].geometry)
        point2 = Point(gdf.iloc[j].geometry)
        line = LineString([point1, point2])
        from_.append(gdf.iloc[i]['STAID'])
        to_.append(gdf.iloc[j]['STAID'])
        edge_lines.append(line)
    gdf_edges = gpd.GeoDataFrame({'geometry': edge_lines})
    gdf_edges['to'] = to_
    gdf_edges['from'] = from_
    gdf_edges.to_file(outfile.replace('.np', '.shp'), epsg='EPSG:4326', engine='fiona')


if __name__ == '__main__':
    if __name__ == '__main__':
        d = '/media/research/IrrigationGIS/dads'
        clim = '/media/research/IrrigationGIS/climate'
        if not os.path.exists(d):
            d = '/home/dgketchum/data/IrrigationGIS/dads'
            clim = '/home/dgketchum/data/IrrigationGIS/climate'

        fields = os.path.join(clim, 'stations', 'ghcn_MT_stations.shp')
        # fields = os.path.join(d, 'met', 'stations', 'ghcn_MT_stations.shp')
        np_out = os.path.join(d, 'met', 'obs', 'ghcn', 'edges.np')
        targets_ = os.path.join(d, 'met', 'obs', 'ghcn', 'metadata.json')

        generate_edge_index(fields, targets_, np_out)
# ========================= EOF ====================================================================
