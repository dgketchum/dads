import json
import os
from datetime import datetime

import pandas as pd
import geopandas as gpd
import numpy as np
from fiona.crs import CRS
from shapely.geometry import LineString, Point

from prep.columns_desc import GEO_FEATURES
from sklearn.preprocessing import MinMaxScaler
from prep.stations import merge_shapefiles, get_station_observation_metadata
from sklearn.neighbors import BallTree


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

        # Build feature matrix from numeric GEO features only; keep 'train' and id out of scaling
        feat_cols = [c for c in list(GEO_FEATURES) if c in stations.columns]
        feats = stations[feat_cols].copy()
        feats.index = stations[self.index_col]
        # Coerce non-numeric to NaN, handle inf, and impute with column medians
        for col in feats.columns:
            feats[col] = pd.to_numeric(feats[col], errors='coerce')
        # Debug: summarize anomalies before imputation/scaling
        arr = feats.to_numpy()
        nonfinite_mask = ~np.isfinite(arr)
        if nonfinite_mask.any():
            counts = nonfinite_mask.sum(axis=0)
            problem_cols = {c: int(n) for c, n in zip(feats.columns, counts) if n > 0}
            print('Graph prep: non-finite values detected before scaling (per column counts):', problem_cols)
        large_mask = np.abs(arr) > 1e9
        if large_mask.any():
            counts = large_mask.sum(axis=0)
            large_cols = {c: int(n) for c, n in zip(feats.columns, counts) if n > 0}
            # show maxima for those columns
            max_abs = {c: float(np.nanmax(np.abs(arr[:, i]))) for i, c in enumerate(feats.columns) if large_cols.get(c, 0) > 0}
            print('Graph prep: unusually large magnitudes before scaling (>|1e9|):', large_cols)
            print('Graph prep: column max |value| before scaling:', max_abs)
        feats.replace([np.inf, -np.inf], np.nan, inplace=True)
        if not feats.empty:
            medians = feats.median(axis=0, numeric_only=True)
            feats = feats.fillna(medians)
            # Debug: confirm issues resolved
            arr2 = feats.to_numpy()
            if ~np.isfinite(arr2).any():
                pass
            else:
                counts2 = (~np.isfinite(arr2)).sum(axis=0)
                remaining = {c: int(n) for c, n in zip(feats.columns, counts2) if n > 0}
                print('Graph prep: remaining non-finite after imputation:', remaining)

        scaler = MinMaxScaler()
        feats_scaled = pd.DataFrame(scaler.fit_transform(feats),
                                    columns=feats.columns,
                                    index=feats.index)

        train_flags = stations.set_index(self.index_col)['train'] if 'train' in stations.columns else pd.Series(0, index=feats_scaled.index)
        val_dct, train_dct = {}, {}
        for i, r in feats_scaled.iterrows():
            t = int(train_flags.loc[i]) if i in train_flags.index else 0
            if t:
                train_dct[i] = r.to_list()
            # Include all in validation attributes (train nodes also available to validation graph)
            val_dct[i] = r.to_list()

        gdf = stations[[self.index_col, 'train', 'geometry']].copy()
        train_gdf = gdf[gdf['train'] == 1]
        val_gdf = gdf[gdf['train'] == 0]

        # Build BallTree on training nodes (lat, lon in radians)
        spatial_edges_parts = []
        k = int(self.k_nearest) if self.k_nearest and self.k_nearest > 0 else 1

        train_coords = np.array([[pt.y, pt.x] for pt in train_gdf.geometry], dtype=np.float64)
        train_rad = np.radians(train_coords)
        tree = BallTree(train_rad, metric='haversine')

        # Validation nodes receive edges from k nearest training nodes
        val_coords = np.array([[pt.y, pt.x] for pt in val_gdf.geometry], dtype=np.float64)
        val_rad = np.radians(val_coords)
        k_v = min(k, len(train_gdf))
        _, ind_v = tree.query(val_rad, k=k_v)

        # Map local indices to global gdf positions
        train_index_map = {i: j for i, j in enumerate(train_gdf.index)}
        mapped_neighbors = np.vectorize(train_index_map.get)(ind_v)
        row_indices_v = np.repeat(np.arange(len(val_gdf)), k_v)
        spatial_edges_v = np.column_stack([row_indices_v, mapped_neighbors.ravel()])

        # Map val local rows to global
        val_index_map = {i: j for i, j in enumerate(val_gdf.index)}
        spatial_edges_v[:, 0] = np.vectorize(val_index_map.get)(spatial_edges_v[:, 0])
        spatial_edges_parts.append(spatial_edges_v)

        # Training nodes receive edges from k nearest training nodes (excluding self)
        k_t = min(k + 1, len(train_gdf))
        _, ind_t = tree.query(train_rad, k=k_t)
        # drop self (first neighbor is itself)
        ind_t = ind_t[:, 1:]
        k_eff = ind_t.shape[1]
        train_index_map = {i: j for i, j in enumerate(train_gdf.index)}
        mapped_neighbors_t = np.vectorize(train_index_map.get)(ind_t)
        row_indices_t = np.repeat(np.arange(len(train_gdf)), k_eff)
        spatial_edges_t = np.column_stack([row_indices_t, mapped_neighbors_t.ravel()])
        # Map train local rows to global
        spatial_edges_t[:, 0] = np.vectorize(train_index_map.get)(spatial_edges_t[:, 0])
        spatial_edges_parts.append(spatial_edges_t)

        spatial_edges = np.concatenate(spatial_edges_parts) if spatial_edges_parts else np.empty((0, 2), dtype=int)

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

    d = '/home/dgketchum/data/IrrigationGIS'

    training = '/data/ssd2/dads/training'

    parquet_root = os.path.join(training, 'parquet')
    obs_vars = ['tmax_obs']
    # obs_vars = [v for v in os.listdir(parquet_root) if os.path.isdir(os.path.join(parquet_root, v))]

    madis_glob = 'madis_02JULY2025_mgrs'
    ghcn_glob = 'ghcn_CANUSA_stations_mgrs'
    madis_shp = os.path.join(d, 'dads', 'met', 'stations', f'{madis_glob}.shp')
    ghcn_shp = os.path.join(d, 'climate', 'ghcn', 'stations', f'{ghcn_glob}.shp')

    # Build or load merged stations shapefile; timestamp to day
    merged_dir = os.path.join(d, 'dads', 'met', 'stations', 'merged')

    ts = datetime.now().strftime('%Y%m%d')
    merged_name = f'merged_{ts}.shp'
    merged_path = os.path.join(merged_dir, merged_name)
    overwrite_merged = True
    if (not overwrite_merged) and os.path.exists(merged_path):
        print(f'Loading existing merged stations: {merged_path}')
        merged = gpd.read_file(merged_path)
    else:
        print(f'Building merged stations: {merged_path}')
        merged = merge_shapefiles([ghcn_shp, madis_shp], save=True, out_dir=merged_dir, filename=merged_name)

    obs_meta_shp = os.path.join(training, 'graph', 'station_observations.shp')
    overwrite_obs_meta = True
    if (not overwrite_obs_meta) and os.path.exists(obs_meta_shp):
        print(f'Skipping observation metadata build; loading existing: {obs_meta_shp}')
        stations_obs = gpd.read_file(obs_meta_shp)
    else:
        stations_obs = get_station_observation_metadata(parquet_root, obs_vars, merged, obs_meta_shp)

    for target_var in obs_vars:
        output_dir_ = os.path.join(training, 'graph', target_var)
        os.makedirs(output_dir_, exist_ok=True)
        node_prep = Graph(stations_obs, output_dir_, k_nearest=10, index_col='fid',
                          split_mode='observations', split_percent=0.8, random_state=42)
        node_prep.generate_edge_index()

# ========================= EOF ====================================================================
