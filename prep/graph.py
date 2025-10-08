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
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def _read_geo_from_parquet(args):
    staid, parquet_dir = args
    p = os.path.join(parquet_dir, f'{staid}.parquet')
    if not os.path.exists(p):
        return None
    try:
        dfp = pd.read_parquet(p)
    except Exception:
        return None
    cols = [c for c in GEO_FEATURES if c in dfp.columns]
    if not cols:
        return None
    m = dfp[cols].mean(numeric_only=True)
    return (staid, m)


class Graph:
    def __init__(self, stations, output_dir, k_nearest=2, bounds=None, index_col='fid',
                 split_mode='observations', split_percent=0.8, random_state=None,
                 parquet_dir=None, use_parquet_features=False, num_workers=1,
                 neighbor_mode='spatial', spatial_weight=0.2):
        self.fields = stations
        self.output_dir = output_dir
        self.k_nearest = k_nearest
        self.pth_stations = []
        self.bounds = bounds
        self.index_col = index_col
        self.split_mode = split_mode
        self.split_percent = split_percent
        self.random_state = random_state
        self.parquet_dir = parquet_dir
        self.use_parquet_features = use_parquet_features
        self.num_workers = int(num_workers) if num_workers is not None else 1
        self.neighbor_mode = neighbor_mode
        self.spatial_weight = float(spatial_weight)

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
        print('[Graph] Preparing stations')
        stations = self._prepare_stations()

        print('[Graph] Building features (use_parquet_features={})'.format(self.use_parquet_features))
        feats = self._compute_raw_features(stations)

        print('[Graph] Cleaning and scaling features')
        feats_scaled = self._clean_and_scale_features(feats)

        print('[Graph] Building attribute dictionaries')
        train_flags = stations.set_index(self.index_col)['train'] if 'train' in stations.columns else pd.Series(0, index=feats_scaled.index)
        train_dct, val_dct = self._build_attr_dicts(feats_scaled, train_flags)

        print('[Graph] Splitting train/val GeoDataFrames')
        gdf, train_gdf, val_gdf = self._split_train_val(stations)

        print('[Graph] Building neighbor edges (mode={}, k={}, spatial_weight={})'.format(self.neighbor_mode, self.k_nearest, self.spatial_weight))
        spatial_edges = self._build_edges(train_gdf, val_gdf, feats_scaled)

        print('[Graph] Converting edges to GeoDataFrame ({} edges)'.format(len(spatial_edges)))
        gdf_edges = self._edges_to_gdf(spatial_edges, gdf)

        print('[Graph] Writing outputs to {}'.format(self.output_dir))
        self._write_outputs(gdf_edges, train_dct, val_dct)

        index_to_staid = {i: staid for i, staid in enumerate(gdf[self.index_col])}
        return spatial_edges, gdf, index_to_staid

    def _prepare_stations(self):
        if isinstance(self.fields, gpd.GeoDataFrame):
            stations = self.fields.copy()
        else:
            stations = gpd.read_file(self.fields)
        stations.index = list(range(stations.shape[0]))
        if self.bounds:
            w, s, e, n = self.bounds
            stations = stations[(stations['latitude'] < n) & (stations['latitude'] >= s)]
            stations = stations[(stations['longitude'] < e) & (stations['longitude'] >= w)]
        if self.parquet_dir:
            try:
                files = [f for f in os.listdir(self.parquet_dir) if f.endswith('.parquet')]
                have_files = {os.path.splitext(f)[0] for f in files}
                stations = stations[stations[self.index_col].astype(str).isin(have_files)]
            except Exception:
                pass  # likely error: ensure parquet_dir exists
        return stations

    def _compute_raw_features(self, stations):
        if self.use_parquet_features and self.parquet_dir:
            sta_list = stations[self.index_col].astype(str).tolist()
            rows, idxs = [], []
            if self.num_workers is None or self.num_workers <= 1:
                it = (_read_geo_from_parquet((staid, self.parquet_dir)) for staid in sta_list)
                it = tqdm(it, total=len(sta_list), desc='parquet GEO_FEATURES')
                for r in it:
                    if r is None:
                        continue
                    staid, m = r
                    rows.append(m)
                    idxs.append(staid)
            else:
                tasks = [(staid, self.parquet_dir) for staid in sta_list]
                with ProcessPoolExecutor(max_workers=int(self.num_workers)) as ex:
                    for r in tqdm(ex.map(_read_geo_from_parquet, tasks), total=len(tasks), desc='parquet GEO_FEATURES'):
                        if r is None:
                            continue
                        staid, m = r
                        rows.append(m)
                        idxs.append(staid)
            if rows:
                feats = pd.DataFrame(rows, index=idxs)
            else:
                feats = stations[[c for c in list(GEO_FEATURES) if c in stations.columns]].copy()
                feats.index = stations[self.index_col]
        else:
            feat_cols = [c for c in list(GEO_FEATURES) if c in stations.columns]
            feats = stations[feat_cols].copy()
            feats.index = stations[self.index_col]
        return feats

    def _clean_and_scale_features(self, feats):
        for col in feats.columns:
            feats[col] = pd.to_numeric(feats[col], errors='coerce')
        arr = feats.to_numpy()
        nonfinite_mask = ~np.isfinite(arr)
        if nonfinite_mask.any():
            counts = nonfinite_mask.sum(axis=0)
            problem_cols = {c: int(n) for c, n in zip(feats.columns, counts) if n > 0}
            print('Graph prep: non-finite before scaling:', problem_cols)
        large_mask = np.abs(arr) > 1e9
        if large_mask.any():
            counts = large_mask.sum(axis=0)
            large_cols = {c: int(n) for c, n in zip(feats.columns, counts) if n > 0}
            max_abs = {c: float(np.nanmax(np.abs(arr[:, i]))) for i, c in enumerate(feats.columns) if large_cols.get(c, 0) > 0}
            print('Graph prep: large magnitudes (>|1e9|):', large_cols)
            print('Graph prep: column max |value|:', max_abs)
        feats.replace([np.inf, -np.inf], np.nan, inplace=True)
        if not feats.empty:
            medians = feats.median(axis=0, numeric_only=True)
            feats = feats.fillna(medians)
            arr2 = feats.to_numpy()
            if np.isfinite(arr2).all():
                pass
            else:
                counts2 = (~np.isfinite(arr2)).sum(axis=0)
                remaining = {c: int(n) for c, n in zip(feats.columns, counts2) if n > 0}
                print('Graph prep: remaining non-finite after imputation:', remaining)
        scaler = MinMaxScaler()
        feats_scaled = pd.DataFrame(scaler.fit_transform(feats), columns=feats.columns, index=feats.index)
        return feats_scaled

    def _build_attr_dicts(self, feats_scaled, train_flags):
        train_dct, val_dct = {}, {}
        for i, r in feats_scaled.iterrows():
            t = int(train_flags.loc[i]) if i in train_flags.index else 0
            if t:
                train_dct[i] = r.to_list()
            val_dct[i] = r.to_list()
        return train_dct, val_dct

    def _split_train_val(self, stations):
        gdf = stations[[self.index_col, 'train', 'geometry']].copy()
        train_gdf = gdf[gdf['train'] == 1]
        val_gdf = gdf[gdf['train'] == 0]
        print('[Graph] Train nodes: {}, Val nodes: {}'.format(len(train_gdf), len(val_gdf)))
        return gdf, train_gdf, val_gdf

    def _build_edges(self, train_gdf, val_gdf, feats_scaled):
        parts = []
        k = int(self.k_nearest) if self.k_nearest and self.k_nearest > 0 else 1
        train_coords = np.array([[pt.y, pt.x] for pt in train_gdf.geometry], dtype=np.float64)
        train_rad = np.radians(train_coords)
        val_coords = np.array([[pt.y, pt.x] for pt in val_gdf.geometry], dtype=np.float64)
        val_rad = np.radians(val_coords)
        if self.neighbor_mode == 'spatial':
            tree = BallTree(train_rad, metric='haversine')
            k_v = min(k, len(train_gdf))
            _, ind_v = tree.query(val_rad, k=k_v)
            mapped_neighbors = train_gdf.index.values[ind_v]
            row_indices_v = np.repeat(val_gdf.index.values, k_v)
            parts.append(np.column_stack([row_indices_v, mapped_neighbors.ravel()]))
            k_t = min(k + 1, len(train_gdf))
            _, ind_t = tree.query(train_rad, k=k_t)
            ind_t = ind_t[:, 1:]
            k_eff = ind_t.shape[1]
            mapped_neighbors_t = train_gdf.index.values[ind_t]
            row_indices_t = np.repeat(train_gdf.index.values, k_eff)
            parts.append(np.column_stack([row_indices_t, mapped_neighbors_t.ravel()]))
        else:
            feats_scaled.index = feats_scaled.index.astype(str)
            tr_ids = train_gdf[self.index_col].astype(str).tolist()
            va_ids = val_gdf[self.index_col].astype(str).tolist()
            train_feat = feats_scaled.loc[tr_ids].to_numpy(dtype=np.float64) if len(tr_ids) else np.empty((0, 0))
            val_feat = feats_scaled.loc[va_ids].to_numpy(dtype=np.float64) if len(va_ids) and train_feat.size else np.empty((0, train_feat.shape[1] if train_feat.size else 0))
            if train_feat.size:
                tree_feat = BallTree(train_feat, metric='euclidean')
                cand_mult = 8
                if len(val_feat):
                    kv = min(max(k * cand_mult, k), len(train_gdf))
                    d_feat_v, ind_feat_v = tree_feat.query(val_feat, k=kv)
                    edges_v = []
                    for i in range(len(val_feat)):
                        cand_local = ind_feat_v[i]
                        fd = d_feat_v[i]
                        if cand_local.size == 0:
                            continue
                        lat1 = val_rad[i, 0]
                        lon1 = val_rad[i, 1]
                        lat2 = train_rad[cand_local, 0]
                        lon2 = train_rad[cand_local, 1]
                        dlat = lat2 - lat1
                        dlon = lon2 - lon1
                        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
                        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                        sd = 6371.0 * c
                        fmin, fmax = fd.min(), fd.max()
                        smin, smax = sd.min(), sd.max()
                        fn = (fd - fmin) / (fmax - fmin + 1e-12) if fmax > fmin else fd * 0
                        sn = (sd - smin) / (smax - smin + 1e-12) if smax > smin else sd * 0
                        comb = self.spatial_weight * sn + (1.0 - self.spatial_weight) * fn
                        order = np.argsort(comb)[:k]
                        nbr_global = train_gdf.index.values[cand_local[order]]
                        to_global = val_gdf.index.values[i]
                        edges_v.append(np.column_stack([np.full_like(nbr_global, to_global), nbr_global]))
                    if edges_v:
                        parts.append(np.vstack(edges_v))
                if len(train_feat):
                    kt = min(max(k * cand_mult, k + 1), len(train_gdf))
                    d_feat_t, ind_feat_t = tree_feat.query(train_feat, k=kt)
                    edges_t = []
                    for i in range(len(train_feat)):
                        cand_local = ind_feat_t[i]
                        fd = d_feat_t[i]
                        if cand_local.size == 0:
                            continue
                        if cand_local[0] == i:
                            cand_local = cand_local[1:]
                            fd = fd[1:]
                        if cand_local.size == 0:
                            continue
                        lat1 = train_rad[i, 0]
                        lon1 = train_rad[i, 1]
                        lat2 = train_rad[cand_local, 0]
                        lon2 = train_rad[cand_local, 1]
                        dlat = lat2 - lat1
                        dlon = lon2 - lon1
                        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
                        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                        sd = 6371.0 * c
                        fmin, fmax = fd.min(), fd.max()
                        smin, smax = sd.min(), sd.max()
                        fn = (fd - fmin) / (fmax - fmin + 1e-12) if fmax > fmin else fd * 0
                        sn = (sd - smin) / (smax - smin + 1e-12) if smax > smin else sd * 0
                        comb = self.spatial_weight * sn + (1.0 - self.spatial_weight) * fn
                        order = np.argsort(comb)[:k]
                        nbr_global = train_gdf.index.values[cand_local[order]]
                        to_global = train_gdf.index.values[i]
                        edges_t.append(np.column_stack([np.full_like(nbr_global, to_global), nbr_global]))
                    if edges_t:
                        parts.append(np.vstack(edges_t))
        spatial_edges = np.concatenate(parts) if parts else np.empty((0, 2), dtype=int)
        spatial_edges = spatial_edges[:, [1, 0]]
        return spatial_edges

    def _edges_to_gdf(self, spatial_edges, gdf):
        edge_lines, train = [], []
        to_, from_ = [], []
        it = tqdm(enumerate(spatial_edges), total=len(spatial_edges), desc='build edge geometries')
        for e, (i, j) in it:
            from_fid = gdf.loc[i, self.index_col]
            to_fid = gdf.loc[j, self.index_col]
            point1 = Point(gdf.loc[i, 'geometry'])
            point2 = Point(gdf.loc[j, 'geometry'])
            line = LineString([point1, point2])
            from_.append(from_fid)
            to_.append(to_fid)
            edge_lines.append(line)
            if gdf.loc[i, 'train'] and gdf.loc[j, 'train']:
                train.append(1)
            else:
                train.append(0)
        gdf_edges = gpd.GeoDataFrame({'geometry': edge_lines})
        gdf_edges['to'] = to_
        gdf_edges['from'] = from_
        gdf_edges['train'] = train
        return gdf_edges

    def _write_outputs(self, gdf_edges, train_dct, val_dct):
        train_edges = gdf_edges.groupby('to')['from'].agg(list).to_dict()
        with open(os.path.join(self.output_dir, 'train_edge_index.json'), 'w') as f:
            json.dump(train_edges, f)
        with open(os.path.join(self.output_dir, 'train_edge_attr.json'), 'w') as f:
            json.dump(train_dct, f)
        val_edges = gdf_edges.groupby('to')['from'].agg(list).to_dict()
        with open(os.path.join(self.output_dir, 'val_edge_index.json'), 'w') as f:
            json.dump(val_edges, f)
        with open(os.path.join(self.output_dir, 'val_edge_attr.json'), 'w') as f:
            json.dump(val_dct, f)
        gdf_edges.crs = CRS.from_epsg(4326)
        gdf_edges.to_file(os.path.join(self.output_dir, 'edges.shp'), epsg='EPSG:4326', engine='fiona')


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
    overwrite_merged = False
    if (not overwrite_merged) and os.path.exists(merged_path):
        print(f'Loading existing merged stations: {merged_path}')
        merged = gpd.read_file(merged_path)
    else:
        print(f'Building merged stations: {merged_path}')
        merged = merge_shapefiles([ghcn_shp, madis_shp], save=True, out_dir=merged_dir, filename=merged_name)

    obs_meta_shp = os.path.join(training, 'graph', 'station_observations.shp')
    overwrite_obs_meta = False
    if (not overwrite_obs_meta) and os.path.exists(obs_meta_shp):
        print(f'Skipping observation metadata build; loading existing: {obs_meta_shp}')
        stations_obs = gpd.read_file(obs_meta_shp)
    else:
        stations_obs = get_station_observation_metadata(parquet_root, obs_vars, merged, obs_meta_shp)

    for target_var in obs_vars:
        output_dir_ = os.path.join(training, 'graph', target_var)
        os.makedirs(output_dir_, exist_ok=True)
        sequence_parq = os.path.join(parquet_root, target_var)
        node_prep = Graph(stations_obs, output_dir_, k_nearest=10, index_col='fid',
                          parquet_dir=sequence_parq, use_parquet_features=True,
                          num_workers=16, neighbor_mode='mixed', spatial_weight=0.2,
                          split_percent=0.8, random_state=42)
        node_prep.generate_edge_index()

# ========================= EOF ====================================================================
