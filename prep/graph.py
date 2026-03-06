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
from prep.build_variable_scaler import build_variable_scaler


def _read_geo_from_parquet(args):
    staid, parquet_dir, cols_req = args
    p = os.path.join(parquet_dir, f"{staid}.parquet")
    if not os.path.exists(p):
        return None
    try:
        if cols_req is not None:
            dfp = pd.read_parquet(p, columns=[c for c in cols_req])
            cols = [c for c in cols_req if c in dfp.columns]
        else:
            dfp = pd.read_parquet(p)
            cols = [c for c in GEO_FEATURES if c in dfp.columns]
    except Exception:
        return None
    if not cols:
        return None
    m = dfp[cols].mean(numeric_only=True)
    return (staid, m)


class Graph:
    """Builds station neighbor graphs and static edge/node attributes for DADS.

    Outputs per target variable directory
    - train_edge_index.json / val_edge_index.json: mapping to->list[from] neighbor ids.
    - train_edge_attr.json / val_edge_attr.json: per-station scaled attribute vectors
      used to form (target - neighbor) edge attributes at runtime.
    - train_edge_bearing.json / val_edge_bearing.json: per-edge bearing to neighbor (deg).
    - train_edge_distance.json / val_edge_distance.json: per-edge geodesic distance (km).
    - node_index.json, train_ids.json, val_ids.json: indexing metadata.

    Static geometry (bearing, distance) is computed once here for scalability; day-specific
    data such as target exogenous features and neighbor contexts are assembled at runtime.
    Additionally, a variable-specific MinMax scaler is constructed using train_ids only
    (graph split) and recorded in graph_meta.json to avoid information leakage.
    """

    def __init__(
        self,
        stations,
        output_dir,
        k_nearest=10,
        bounds=None,
        index_col="fid",
        split_percent=0.8,
        random_state=None,
        parquet_dir=None,
        scaler_json=None,
        use_parquet_features=False,
        num_workers=1,
        neighbor_pool_factor=5,
        rebuild_scaler=False,
        record_holders: int = 3,
        features=None,
        n_diverse: int = 3,
    ):

        self.rebuild_scaler = rebuild_scaler
        self.fields = stations
        self.output_dir = output_dir
        self.k_nearest = k_nearest
        self.pth_stations = []
        self.bounds = bounds
        self.index_col = index_col
        self.split_percent = split_percent
        self.random_state = random_state
        self.parquet_dir = parquet_dir
        self.scaler_path = scaler_json
        self.use_parquet_features = use_parquet_features
        self.num_workers = int(num_workers) if num_workers is not None else 1
        self.neighbor_pool_factor = int(neighbor_pool_factor)
        self.record_holders = int(record_holders)
        self.n_diverse = int(n_diverse)
        # feature selection (optional)
        if features is None:
            self.features = list(GEO_FEATURES)
        else:
            assert all(f in GEO_FEATURES for f in features), (
                "unknown feature(s) provided"
            )  # likely error if not subset
            self.features = list(features)

        # assign train/val split immediately
        if isinstance(self.fields, gpd.GeoDataFrame):
            self.fields = self._assign_train_split(self.fields)
        else:
            gdf_ = gpd.read_file(self.fields)
            self.fields = self._assign_train_split(gdf_)

        # load variable-specific scaler using train-only stations (dependent on split)
        self.scaler = None
        self.scaler_feature_names = None
        if self.parquet_dir:
            parquet_root = os.path.dirname(self.parquet_dir)
            var_name = os.path.basename(self.parquet_dir).replace("_obs", "")
            try:
                tr_ids = (
                    self.fields[self.fields["train"] == 1][self.index_col]
                    .astype(str)
                    .tolist()
                )
            except Exception:
                tr_ids = None  # likely error if split not present
            # require explicit scaler_json when using parquet_dir (single-writer policy)
            assert self.scaler_path is not None, (
                "scaler_json required when parquet_dir is provided"
            )
            # honor explicit scaler_json: load if present; else build once under its directory
            if os.path.exists(self.scaler_path) and not self.rebuild_scaler:
                with open(self.scaler_path, "r") as f:
                    params = json.load(f)
                self.scaler_feature_names = params.get("feature_names")
                s = type("S", (), {})()
                s.bias = np.array(params["bias"]).reshape(1, -1)
                s.scale = np.array(params["scale"]).reshape(1, -1)
                self.scaler = s
            else:
                scaler_dir = os.path.dirname(self.scaler_path)
                built_path = build_variable_scaler(
                    parquet_root, var_name, scaler_dir=scaler_dir, station_ids=tr_ids
                )
                # if requested filename mismatches built_path, we record built_path  # likely error if mismatch
                self.scaler_path = built_path
                with open(self.scaler_path, "r") as f:
                    params = json.load(f)
                self.scaler_feature_names = params.get("feature_names")
                s = type("S", (), {})()
                s.bias = np.array(params["bias"]).reshape(1, -1)
                s.scale = np.array(params["scale"]).reshape(1, -1)
                self.scaler = s
        self.attr_columns = None

    def _assign_train_split(self, stations):
        gdf = stations.copy()
        n = gdf.shape[0]
        rng = np.random.default_rng(self.random_state)  # deterministic split
        k = int(max(1, round(n * self.split_percent)))
        idx = rng.permutation(n)
        train_idx = set(idx[:k])
        gdf["train"] = [1 if i in train_idx else 0 for i in range(n)]
        return gdf

    def generate_edge_index(self):
        print("[Graph] Preparing stations")
        stations = self._prepare_stations()

        print(
            "[Graph] Building features (use_parquet_features={})".format(
                self.use_parquet_features
            )
        )
        feats = self._compute_raw_features(stations)

        print("[Graph] Cleaning and scaling features")
        feats_scaled = self._clean_and_scale_features(feats)
        self.attr_columns = list(feats_scaled.columns)

        print("[Graph] Building attribute dictionaries")
        train_flags = (
            stations.set_index(self.index_col)["train"]
            if "train" in stations.columns
            else pd.Series(0, index=feats_scaled.index)
        )
        train_dct, val_dct = self._build_attr_dicts(feats_scaled, train_flags)

        print("[Graph] Splitting train/val GeoDataFrames")
        gdf, train_gdf, val_gdf = self._split_train_val(stations)

        print("[Graph] Building neighbor edges (k={})".format(self.k_nearest))
        spatial_edges = self._build_edges(train_gdf, val_gdf, feats_scaled)

        print(
            "[Graph] Converting edges to GeoDataFrame ({} edges)".format(
                len(spatial_edges)
            )
        )
        gdf_edges = self._edges_to_gdf(spatial_edges, gdf)

        print("[Graph] Writing outputs to {}".format(self.output_dir))
        self._write_outputs(gdf_edges, train_dct, val_dct, gdf)

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
            stations = stations[
                (stations["latitude"] < n) & (stations["latitude"] >= s)
            ]
            stations = stations[
                (stations["longitude"] < e) & (stations["longitude"] >= w)
            ]
        if self.parquet_dir:
            try:
                files = [
                    f for f in os.listdir(self.parquet_dir) if f.endswith(".parquet")
                ]
                have_files = {os.path.splitext(f)[0] for f in files}
                stations = stations[
                    stations[self.index_col].astype(str).isin(have_files)
                ]
            except Exception:
                print(f"{self.parquet_dir} not found or empty")
                raise
        return stations

    def _compute_raw_features(self, stations):
        selected = (
            list(self.features) if self.features is not None else list(GEO_FEATURES)
        )
        if self.use_parquet_features and self.parquet_dir:
            sta_list = stations[self.index_col].astype(str).tolist()
            rows, idxs = [], []
            if self.num_workers is None or self.num_workers <= 1:
                it = (
                    _read_geo_from_parquet((staid, self.parquet_dir, selected))
                    for staid in sta_list
                )
                it = tqdm(it, total=len(sta_list), desc="parquet GEO_FEATURES")
                for r in it:
                    if r is None:
                        continue
                    staid, m = r
                    m = m[[c for c in selected if c in m.index]]
                    rows.append(m)
                    idxs.append(staid)
            else:
                tasks = [(staid, self.parquet_dir, selected) for staid in sta_list]
                with ProcessPoolExecutor(max_workers=int(self.num_workers)) as ex:
                    for r in tqdm(
                        ex.map(_read_geo_from_parquet, tasks),
                        total=len(tasks),
                        desc="parquet GEO_FEATURES",
                    ):
                        if r is None:
                            continue
                        staid, m = r
                        m = m[[c for c in selected if c in m.index]]
                        rows.append(m)
                        idxs.append(staid)
            if rows:
                feats = pd.DataFrame(rows, index=idxs)
            else:
                feats = stations[[c for c in selected if c in stations.columns]].copy()
                feats.index = stations[self.index_col]
        else:
            feat_cols = [c for c in selected if c in stations.columns]
            feats = stations[feat_cols].copy()
            feats.index = stations[self.index_col]
        return feats

    def _clean_and_scale_features(self, feats):
        for col in feats.columns:
            feats[col] = pd.to_numeric(feats[col], errors="coerce")
        arr = feats.to_numpy()
        nonfinite_mask = ~np.isfinite(arr)
        if nonfinite_mask.any():
            counts = nonfinite_mask.sum(axis=0)
            problem_cols = {c: int(n) for c, n in zip(feats.columns, counts) if n > 0}
            print("Graph prep: non-finite before scaling:", problem_cols)
        large_mask = np.abs(arr) > 1e9
        if large_mask.any():
            counts = large_mask.sum(axis=0)
            large_cols = {c: int(n) for c, n in zip(feats.columns, counts) if n > 0}
            max_abs = {
                c: float(np.nanmax(np.abs(arr[:, i])))
                for i, c in enumerate(feats.columns)
                if large_cols.get(c, 0) > 0
            }
            print("Graph prep: large magnitudes (>|1e9|):", large_cols)
            print("Graph prep: column max |value|:", max_abs)
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
                print("Graph prep: remaining non-finite after imputation:", remaining)
        if self.scaler is not None and self.scaler_feature_names is not None:
            cols = list(feats.columns)
            # ensure selected columns exist in scaler feature set
            assert all(c in self.scaler_feature_names for c in cols), (
                "scaler feature_names mismatch"
            )
            idxs = [self.scaler_feature_names.index(c) for c in cols]
            bias = np.asarray(self.scaler.bias).reshape(-1)[idxs]
            scale = np.asarray(self.scaler.scale).reshape(-1)[idxs]
            x = feats.to_numpy(dtype=np.float32)
            x = (x - bias) / scale + 5e-8
            feats_scaled = pd.DataFrame(x, columns=cols, index=feats.index)
        else:
            scaler = MinMaxScaler()
            feats_scaled = pd.DataFrame(
                scaler.fit_transform(feats), columns=feats.columns, index=feats.index
            )
        return feats_scaled

    def _build_attr_dicts(self, feats_scaled, train_flags):
        train_dct, val_dct = {}, {}
        for i, r in feats_scaled.iterrows():
            v = train_flags.loc[i] if i in train_flags.index else 0
            t = (
                int(v.max()) if isinstance(v, pd.Series) else int(v)
            )  # duplicate index -> Series
            if t:
                train_dct[i] = r.to_list()
            val_dct[i] = r.to_list()
        return train_dct, val_dct

    def _split_train_val(self, stations):
        cols = [self.index_col, "train", "geometry"]
        if "obs_count_total" in stations.columns:
            cols.append("obs_count_total")
        gdf = stations[cols].copy()
        train_gdf = gdf[gdf["train"] == 1]
        val_gdf = gdf[gdf["train"] == 0]
        print(
            "[Graph] Train nodes: {}, Val nodes: {}".format(
                len(train_gdf), len(val_gdf)
            )
        )
        return gdf, train_gdf, val_gdf

    def _build_edges(self, train_gdf, val_gdf, feats_scaled):
        # Unified: spatial candidate pool, then select mostly similar with a small diverse set
        k = int(self.k_nearest) if self.k_nearest and self.k_nearest > 0 else 1
        k = max(1, k)
        k_diverse = min(max(0, self.n_diverse), k)
        k_sim = max(0, k - k_diverse)
        parts = []
        # features by station id
        feats_scaled.index = feats_scaled.index.astype(str)
        tr_ids = train_gdf[self.index_col].astype(str).tolist()
        va_ids = val_gdf[self.index_col].astype(str).tolist()
        train_feat = (
            feats_scaled.loc[tr_ids].to_numpy(dtype=np.float64)
            if len(tr_ids)
            else np.empty((0, 0))
        )
        val_feat = (
            feats_scaled.loc[va_ids].to_numpy(dtype=np.float64)
            if len(va_ids) and train_feat.size
            else np.empty((0, train_feat.shape[1] if train_feat.size else 0))
        )
        # spatial coords and tree
        train_coords = np.array(
            [[pt.y, pt.x] for pt in train_gdf.geometry], dtype=np.float64
        )
        train_rad = np.radians(train_coords)
        val_coords = np.array(
            [[pt.y, pt.x] for pt in val_gdf.geometry], dtype=np.float64
        )
        val_rad = np.radians(val_coords)
        if train_feat.size:
            tree_sp = BallTree(train_rad, metric="haversine")
            cand_mult = max(2, int(self.neighbor_pool_factor))

            # val -> train edges
            if len(val_feat):
                kv = min(max(k * cand_mult, k), len(train_gdf))
                _, ind_sp_v = tree_sp.query(val_rad, k=kv)
                edges_v = []
                n_val = min(len(val_feat), ind_sp_v.shape[0])
                for i in range(n_val):
                    cand_local = ind_sp_v[i]
                    if cand_local.size == 0:
                        continue
                    v = val_feat[i]
                    fd = np.linalg.norm(train_feat[cand_local] - v, axis=1)
                    sim_order = np.argsort(fd)
                    dissim_order = np.argsort(-fd)
                    chosen = []
                    # take k_sim most similar
                    for idx in sim_order:
                        if len(chosen) >= k_sim:
                            break
                        chosen.append(cand_local[idx])
                    # add up to k_diverse most dissimilar (diversity)
                    for idx in dissim_order:
                        if len(chosen) >= k:
                            break
                        j = cand_local[idx]
                        if j not in chosen:
                            chosen.append(j)
                    # fill remaining with similar order to reach k
                    if len(chosen) < k:
                        for idx in sim_order:
                            if len(chosen) >= k:
                                break
                            j = cand_local[idx]
                            if j not in chosen:
                                chosen.append(j)
                    nbr_global = train_gdf.index.values[np.array(chosen[:k])]
                    to_global = val_gdf.index.values[i]
                    edges_v.append(
                        np.column_stack(
                            [np.full_like(nbr_global, to_global), nbr_global]
                        )
                    )
                if edges_v:
                    parts.append(np.vstack(edges_v))

            # train -> train edges (exclude self)
            if len(train_feat):
                kt = min(max(k * cand_mult, k + 1), len(train_gdf))
                kq = min(kt + 1, len(train_gdf))
                _, ind_sp_t = tree_sp.query(train_rad, k=kq)
                edges_t = []
                n_tr = min(len(train_feat), ind_sp_t.shape[0])
                for i in range(n_tr):
                    cand_local = ind_sp_t[i]
                    if cand_local.size == 0:
                        continue
                    if cand_local[0] == i:
                        cand_local = cand_local[1:]
                    if cand_local.size == 0:
                        continue
                    v = train_feat[i]
                    fd = np.linalg.norm(train_feat[cand_local] - v, axis=1)
                    sim_order = np.argsort(fd)
                    dissim_order = np.argsort(-fd)
                    chosen = []
                    # take k_sim most similar
                    for idx in sim_order:
                        if len(chosen) >= k_sim:
                            break
                        chosen.append(cand_local[idx])
                    # add up to k_diverse most dissimilar
                    for idx in dissim_order:
                        if len(chosen) >= k:
                            break
                        j = cand_local[idx]
                        if j not in chosen:
                            chosen.append(j)
                    # fill remaining with similar order to reach k
                    if len(chosen) < k:
                        for idx in sim_order:
                            if len(chosen) >= k:
                                break
                            j = cand_local[idx]
                            if j not in chosen:
                                chosen.append(j)
                    nbr_global = train_gdf.index.values[np.array(chosen[:k])]
                    to_global = train_gdf.index.values[i]
                    edges_t.append(
                        np.column_stack(
                            [np.full_like(nbr_global, to_global), nbr_global]
                        )
                    )
                if edges_t:
                    parts.append(np.vstack(edges_t))
        spatial_edges = np.concatenate(parts) if parts else np.empty((0, 2), dtype=int)
        spatial_edges = spatial_edges[:, [1, 0]]
        return spatial_edges

    def _edges_to_gdf(self, spatial_edges, gdf):
        edge_lines, train = [], []
        to_, from_ = [], []
        bearing = []
        bearing_out = []
        distance_km = []
        it = tqdm(
            enumerate(spatial_edges),
            total=len(spatial_edges),
            desc="build edge geometries",
        )
        for e, (i, j) in it:
            from_fid = gdf.loc[i, self.index_col]
            to_fid = gdf.loc[j, self.index_col]
            point1 = Point(gdf.loc[i, "geometry"])
            point2 = Point(gdf.loc[j, "geometry"])
            line = LineString([point1, point2])
            from_.append(from_fid)
            to_.append(to_fid)
            edge_lines.append(line)
            lat1 = np.radians(point1.y)
            lon1 = np.radians(point1.x)
            lat2 = np.radians(point2.y)
            lon2 = np.radians(point2.x)
            dlon = lon2 - lon1
            x = np.sin(dlon) * np.cos(lat2)
            y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
            brng = np.degrees(np.arctan2(x, y))
            brng = (brng + 360.0) % 360.0
            bearing.append(float(brng))
            bearing_out.append(float((brng + 180.0) % 360.0))
            dlat = lat2 - lat1
            a = (
                np.sin(dlat / 2.0) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
            )
            c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
            distance_km.append(float(6371.0088 * c))
            if gdf.loc[i, "train"] and gdf.loc[j, "train"]:
                train.append(1)
            else:
                train.append(0)
        gdf_edges = gpd.GeoDataFrame({"geometry": edge_lines})
        gdf_edges["to"] = to_
        gdf_edges["from"] = from_
        gdf_edges["train"] = train
        gdf_edges["bearing"] = bearing
        gdf_edges["bearing_to_neighbor"] = bearing_out
        gdf_edges["distance_km"] = distance_km
        return gdf_edges

    def _write_outputs(self, gdf_edges, train_dct, val_dct, gdf=None):
        # Scale distance (km) to [0,1] using train-only edges to avoid leakage
        # Compute scaling on edges where both nodes are in train split
        train_mask = gdf_edges["train"] == 1
        if train_mask.any():
            d_train = gdf_edges.loc[train_mask, "distance_km"].to_numpy()
        else:
            d_train = gdf_edges["distance_km"].to_numpy()
        d_min = float(np.min(d_train)) if d_train.size else 0.0
        d_max = float(np.max(d_train)) if d_train.size else 1.0
        d_scale = d_max - d_min
        if d_scale == 0.0:
            d_scale = 1.0  # likely error if all distances identical across edges
        gdf_edges["distance_scaled"] = (
            gdf_edges["distance_km"] - d_min
        ) / d_scale + 5e-8

        train_edges = gdf_edges.groupby("to")["from"].agg(list).to_dict()
        train_bearing = (
            gdf_edges.groupby("to")["bearing_to_neighbor"].agg(list).to_dict()
        )
        train_distance = gdf_edges.groupby("to")["distance_scaled"].agg(list).to_dict()
        # Append global record holders (longest records) to the back as fallbacks
        if (
            gdf is not None
            and "obs_count_total" in gdf.columns
            and self.record_holders > 0
        ):
            train_ids = gdf[gdf["train"] == 1][self.index_col]
            counts = (
                gdf.set_index(self.index_col)["obs_count_total"]
                .reindex(train_ids)
                .fillna(0)
            )
            top_fids = counts.sort_values(ascending=False).index.tolist()[
                : self.record_holders
            ]
            gdf_by_fid = gdf.set_index(self.index_col)
            for k, nbrs in train_edges.items():
                base = list(nbrs)
                blist = list(train_bearing.get(k, []))
                dlist = list(train_distance.get(k, []))
                for fid in top_fids:
                    if fid != k and fid not in base:
                        base.append(fid)
                        if (k in gdf_by_fid.index) and (fid in gdf_by_fid.index):
                            p_to = Point(gdf_by_fid.loc[k, "geometry"])
                            p_nb = Point(gdf_by_fid.loc[fid, "geometry"])
                            lat_to = np.radians(p_to.y)
                            lon_to = np.radians(p_to.x)
                            lat_nb = np.radians(p_nb.y)
                            lon_nb = np.radians(p_nb.x)
                            dlon = lon_nb - lon_to
                            # bearing
                            x = np.sin(dlon) * np.cos(lat_nb)
                            y = np.cos(lat_to) * np.sin(lat_nb) - np.sin(
                                lat_to
                            ) * np.cos(lat_nb) * np.cos(dlon)
                            b = np.degrees(np.arctan2(x, y))
                            b = (b + 360.0) % 360.0
                            blist.append(float(b))
                            # distance (km), then scaled using train-only d_min/d_scale
                            dlat = lat_nb - lat_to
                            a = (
                                np.sin(dlat / 2.0) ** 2
                                + np.cos(lat_to)
                                * np.cos(lat_nb)
                                * np.sin(dlon / 2.0) ** 2
                            )
                            c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
                            dist_km = 6371.0088 * c
                            d_scaled = (dist_km - d_min) / d_scale + 5e-8
                            dlist.append(float(d_scaled))
                train_edges[k] = base
                if blist:
                    train_bearing[k] = blist
                if dlist:
                    train_distance[k] = dlist
        with open(os.path.join(self.output_dir, "train_edge_index.json"), "w") as f:
            json.dump(train_edges, f)
        with open(os.path.join(self.output_dir, "train_edge_bearing.json"), "w") as f:
            json.dump(train_bearing, f)
        with open(os.path.join(self.output_dir, "train_edge_distance.json"), "w") as f:
            json.dump(train_distance, f)
        with open(os.path.join(self.output_dir, "train_edge_attr.json"), "w") as f:
            json.dump(train_dct, f)
        val_edges = gdf_edges.groupby("to")["from"].agg(list).to_dict()
        val_bearing = gdf_edges.groupby("to")["bearing_to_neighbor"].agg(list).to_dict()
        val_distance = gdf_edges.groupby("to")["distance_scaled"].agg(list).to_dict()
        if (
            gdf is not None
            and "obs_count_total" in gdf.columns
            and self.record_holders > 0
        ):
            train_ids = gdf[gdf["train"] == 1][self.index_col]
            counts = (
                gdf.set_index(self.index_col)["obs_count_total"]
                .reindex(train_ids)
                .fillna(0)
            )
            top_fids = counts.sort_values(ascending=False).index.tolist()[
                : self.record_holders
            ]
            gdf_by_fid = gdf.set_index(self.index_col)
            for k, nbrs in val_edges.items():
                base = list(nbrs)
                blist = list(val_bearing.get(k, []))
                dlist = list(val_distance.get(k, []))
                for fid in top_fids:
                    if fid != k and fid not in base:
                        base.append(fid)
                        if (k in gdf_by_fid.index) and (fid in gdf_by_fid.index):
                            p_to = Point(gdf_by_fid.loc[k, "geometry"])
                            p_nb = Point(gdf_by_fid.loc[fid, "geometry"])
                            lat_to = np.radians(p_to.y)
                            lon_to = np.radians(p_to.x)
                            lat_nb = np.radians(p_nb.y)
                            lon_nb = np.radians(p_nb.x)
                            dlon = lon_nb - lon_to
                            # bearing
                            x = np.sin(dlon) * np.cos(lat_nb)
                            y = np.cos(lat_to) * np.sin(lat_nb) - np.sin(
                                lat_to
                            ) * np.cos(lat_nb) * np.cos(dlon)
                            b = np.degrees(np.arctan2(x, y))
                            b = (b + 360.0) % 360.0
                            blist.append(float(b))
                            # distance (km), then scaled using train-only d_min/d_scale
                            dlat = lat_nb - lat_to
                            a = (
                                np.sin(dlat / 2.0) ** 2
                                + np.cos(lat_to)
                                * np.cos(lat_nb)
                                * np.sin(dlon / 2.0) ** 2
                            )
                            c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
                            dist_km = 6371.0088 * c
                            d_scaled = (dist_km - d_min) / d_scale + 5e-8
                            dlist.append(float(d_scaled))
                val_edges[k] = base
                if blist:
                    val_bearing[k] = blist
                if dlist:
                    val_distance[k] = dlist
        with open(os.path.join(self.output_dir, "val_edge_index.json"), "w") as f:
            json.dump(val_edges, f)
        with open(os.path.join(self.output_dir, "val_edge_bearing.json"), "w") as f:
            json.dump(val_bearing, f)
        with open(os.path.join(self.output_dir, "val_edge_distance.json"), "w") as f:
            json.dump(val_distance, f)
        with open(os.path.join(self.output_dir, "val_edge_attr.json"), "w") as f:
            json.dump(val_dct, f)
        gdf_edges.crs = CRS.from_epsg(4326)
        gdf_edges.to_file(
            os.path.join(self.output_dir, "edges.shp"), epsg="EPSG:4326", engine="fiona"
        )

        # persist node index and graph metadata for reproducibility/interoperability
        if gdf is not None:
            node_idx = [str(s) for s in gdf[self.index_col].tolist()]
            with open(os.path.join(self.output_dir, "node_index.json"), "w") as f:
                json.dump(node_idx, f)
            # explicit split id lists
            tr_ids = [str(s) for s in gdf[gdf["train"] == 1][self.index_col].tolist()]
            va_ids = [str(s) for s in gdf[gdf["train"] == 0][self.index_col].tolist()]
            with open(os.path.join(self.output_dir, "train_ids.json"), "w") as f:
                json.dump(tr_ids, f)
            with open(os.path.join(self.output_dir, "val_ids.json"), "w") as f:
                json.dump(va_ids, f)
        meta = {
            "features_order": self.attr_columns
            if self.attr_columns is not None
            else None,
            "scaler_path": self.scaler_path,
            "scaler_feature_names": self.scaler_feature_names,
            "k_nearest": self.k_nearest,
            "neighbor_pool_factor": self.neighbor_pool_factor,
            "record_holders": self.record_holders,
            "split_percent": self.split_percent,
            "random_state": self.random_state,
            "index_col": self.index_col,
            "parquet_dir": self.parquet_dir,
            "use_parquet_features": self.use_parquet_features,
            "distance_scaler": {"bias": d_min, "scale": d_scale},
        }
        with open(os.path.join(self.output_dir, "graph_meta.json"), "w") as f:
            json.dump(meta, f)
        # Also persist dedicated extras scaler JSON for downstream consumers
        extras = {"distance": {"bias": d_min, "scale": d_scale}}
        with open(os.path.join(self.output_dir, "edge_extras_scaler.json"), "w") as f:
            json.dump(extras, f)


if __name__ == "__main__":
    d = "/nas"

    training = "/data/ssd2/dads/training"

    parquet_root = os.path.join(training, "parquet")
    obs_vars = ["tmax_obs"]
    # obs_vars = [v for v in os.listdir(parquet_root) if os.path.isdir(os.path.join(parquet_root, v))]

    madis_glob = "madis_17MAY2025_mgrs"
    ghcn_glob = "ghcn_CANUSA_stations_mgrs"
    madis_shp = os.path.join(d, "dads", "met", "stations", f"{madis_glob}.shp")
    ghcn_shp = os.path.join(d, "climate", "ghcn", "stations", f"{ghcn_glob}.shp")

    # Build or load merged stations shapefile; timestamp to day
    merged_dir = os.path.join(d, "dads", "met", "stations", "merged")

    ts = datetime.now().strftime("%Y%m%d")
    merged_name = f"merged_{ts}.shp"
    merged_path = os.path.join(merged_dir, merged_name)
    overwrite_merged = False
    if (not overwrite_merged) and os.path.exists(merged_path):
        print(f"Loading existing merged stations: {merged_path}")
        merged = gpd.read_file(merged_path)
    else:
        print(f"Building merged stations: {merged_path}")
        merged = merge_shapefiles(
            [ghcn_shp, madis_shp], save=True, out_dir=merged_dir, filename=merged_name
        )

    obs_meta_shp = os.path.join(training, "graph", "station_observations.shp")
    overwrite_obs_meta = False
    if (not overwrite_obs_meta) and os.path.exists(obs_meta_shp):
        print(f"Skipping observation metadata build; loading existing: {obs_meta_shp}")
        stations_obs = gpd.read_file(obs_meta_shp)
    else:
        stations_obs = get_station_observation_metadata(
            parquet_root, obs_vars, merged, obs_meta_shp
        )

    select_feats = [
        # 'lat',
        # 'lon',
        "rsun",
        "aspect",
        "elevation",
        "slope",
        # 'B10',
        # 'B2',
        # 'B3',
        # 'B4',
        # 'B5',
        # 'B6',
        # 'B7',
    ]

    for target_var in obs_vars:
        output_dir_ = os.path.join(training, "graph", target_var)
        os.makedirs(output_dir_, exist_ok=True)
        sequence_parq = os.path.join(parquet_root, target_var)
        scaler_json_ = os.path.join(
            training, "scalers", f"{target_var.replace('_obs', '')}.json"
        )

        node_prep = Graph(
            stations_obs,
            output_dir_,
            k_nearest=10,
            index_col="fid",
            parquet_dir=sequence_parq,
            scaler_json=scaler_json_,
            use_parquet_features=True,
            num_workers=16,
            bounds=None,
            rebuild_scaler=True,
            split_percent=0.8,
            random_state=42,
            features=select_feats,
        )
        node_prep.generate_edge_index()

# ========================= EOF ====================================================================
