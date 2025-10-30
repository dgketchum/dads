import json
import os
from typing import Dict, List, Tuple, Optional, Set

import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, LineString
import geopandas as gpd
from fiona.crs import CRS


def _load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def _write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f)


def _stations_in_bounds_from_parquet(
    parquet_dir: str,
    station_ids: List[str],
    bounds: Tuple[float, float, float, float],
) -> Set[str]:
    """Return station ids whose lat/lon fall within bounds.

    bounds: (west, south, east, north)
    """
    w, s, e, n = bounds
    keep: Set[str] = set()
    for sid in station_ids:
        fp = os.path.join(parquet_dir, f"{sid}.parquet")
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_parquet(fp, columns=['lat', 'lon'])
            if 'lat' not in df.columns or 'lon' not in df.columns or df.empty:
                continue
            # first valid row is fine for static lat/lon
            lat = float(df['lat'].iloc[0])
            lon = float(df['lon'].iloc[0])
            if (s <= lat <= n) and (w <= lon <= e):
                keep.add(str(sid))
        except Exception:
            # skip on read errors
            continue
    return keep


def _filter_neighbors(
    edges: Dict[str, List[str]],
    bearings: Dict[str, List[float]],
    distances: Dict[str, List[float]],
    in_bounds: Set[str],
    ensure_min_k: int = 1,
    include_border: bool = True,
) -> Tuple[Dict[str, List[str]], Dict[str, List[float]], Dict[str, List[float]], Set[str]]:
    """Filter neighbor dicts to keep only in-bounds neighbors, with optional border inclusion.

    Returns: (edges_out, bearings_out, distances_out, used_nodes)
    used_nodes: union of all 'to' kept and all neighbors used (may include border nodes)
    """
    edges_out: Dict[str, List[str]] = {}
    bearings_out: Dict[str, List[float]] = {}
    distances_out: Dict[str, List[float]] = {}
    used_nodes: Set[str] = set()

    for to_id, nbrs in edges.items():
        # Only build subgraph entries for targets in bounds
        if to_id not in in_bounds:
            continue
        bs = bearings.get(to_id, [])
        ds = distances.get(to_id, [])
        # align lengths defensively
        L = min(len(nbrs), len(bs), len(ds))
        nbrs = nbrs[:L]
        bs = bs[:L]
        ds = ds[:L]

        keep_idx = [i for i, nb in enumerate(nbrs) if nb in in_bounds]
        if keep_idx:
            flt_nbrs = [nbrs[i] for i in keep_idx]
            flt_bs = [bs[i] for i in keep_idx]
            flt_ds = [ds[i] for i in keep_idx]
        else:
            flt_nbrs, flt_bs, flt_ds = [], [], []

        # Ensure connectivity: if no in-bounds neighbors and requested, bring a few border neighbors
        if include_border and len(flt_nbrs) < ensure_min_k:
            want = ensure_min_k - len(flt_nbrs)
            # take from original order (deterministic)
            add_pairs = []
            for i in range(L):
                if nbrs[i] in in_bounds:
                    continue
                add_pairs.append((nbrs[i], bs[i], ds[i]))
                if len(add_pairs) >= want:
                    break
            if add_pairs:
                for nb, b, d in add_pairs:
                    flt_nbrs.append(nb)
                    flt_bs.append(b)
                    flt_ds.append(d)

        if flt_nbrs:
            edges_out[to_id] = flt_nbrs
            bearings_out[to_id] = flt_bs
            distances_out[to_id] = flt_ds
            used_nodes.add(to_id)
            used_nodes.update(flt_nbrs)

    return edges_out, bearings_out, distances_out, used_nodes


    


def _read_latlon(parquet_dir: str, station_ids: List[str]) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for sid in station_ids:
        fp = os.path.join(parquet_dir, f"{sid}.parquet")
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_parquet(fp, columns=['lat', 'lon'])
            if df.empty:
                continue
            lat = float(df['lat'].iloc[0])
            lon = float(df['lon'].iloc[0])
            out[str(sid)] = (lat, lon)
        except Exception:
            continue
    return out


def _haversine_bearing(lat1, lon1, lat2, lon2) -> Tuple[float, float]:
    rlat1 = np.radians(lat1)
    rlon1 = np.radians(lon1)
    rlat2 = np.radians(lat2)
    rlon2 = np.radians(lon2)
    dlon = rlon2 - rlon1
    dlat = rlat2 - rlat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    dist_km = 6371.0088 * c
    x = np.sin(dlon) * np.cos(rlat2)
    y = np.cos(rlat1) * np.sin(rlat2) - np.sin(rlat1) * np.cos(rlat2) * np.cos(dlon)
    brng = np.degrees(np.arctan2(x, y))
    brng = (brng + 360.0) % 360.0
    return float(dist_km), float(brng)


def _looks_like_polygon(spec) -> bool:
    try:
        return (
            hasattr(spec, '__len__') and len(spec) >= 3 and
            hasattr(spec[0], '__len__') and len(spec[0]) == 2
        )
    except Exception:
        return False


def _stations_in_polygon_from_parquet(
    parquet_dir: str,
    station_ids: List[str],
    poly_latlon: List[Tuple[float, float]],
) -> Set[str]:
    # poly_latlon given as [(lat, lon), ...]; shapely expects (x=lon, y=lat)
    ring = [(float(lon), float(lat)) for (lat, lon) in poly_latlon]
    poly = Polygon(ring)
    keep: Set[str] = set()
    for sid, (lat, lon) in _read_latlon(parquet_dir, station_ids).items():
        pt = Point(float(lon), float(lat))
        if poly.covers(pt):
            keep.add(str(sid))
    return keep


def write_tiered_val_subgraph(
    source_graph_dir: str,
    parquet_dir: str,
    output_dir: str,
    outer_bounds: Tuple[float, float, float, float],
    val_bounds: Tuple[float, float, float, float],
    k_val: int = 10,
    ensure_min_k_train: int = 1,
    val_target_ids: Optional[List[str]] = None,
) -> None:
    # load source artifacts
    t_idx = _load_json(os.path.join(source_graph_dir, 'train_edge_index.json'))
    t_bear = _load_json(os.path.join(source_graph_dir, 'train_edge_bearing.json'))
    t_dist = _load_json(os.path.join(source_graph_dir, 'train_edge_distance.json'))
    t_attr = _load_json(os.path.join(source_graph_dir, 'train_edge_attr.json'))
    v_attr = _load_json(os.path.join(source_graph_dir, 'val_edge_attr.json'))

    meta_path = os.path.join(source_graph_dir, 'graph_meta.json')
    meta = _load_json(meta_path) if os.path.exists(meta_path) else {}
    ds = meta.get('distance_scaler', {'bias': 0.0, 'scale': 1.0})
    d_bias = float(ds.get('bias', 0.0))
    d_scale = float(ds.get('scale', 1.0)) or 1.0  # likely error if zero scale

    # membership by bounds
    attr_nodes = set(t_attr.keys()) | set(v_attr.keys())
    in_outer = _stations_in_bounds_from_parquet(parquet_dir, list(attr_nodes), outer_bounds)
    if _looks_like_polygon(val_bounds):
        in_val = _stations_in_polygon_from_parquet(parquet_dir, list(in_outer), val_bounds)
    else:
        in_val = _stations_in_bounds_from_parquet(parquet_dir, list(in_outer), val_bounds)
    train_set = in_outer - in_val
    val_set = in_val

    # filter train edges strictly to train_set (no border fill)
    tr_idx_out, tr_bear_out, tr_dist_out, used_tr = _filter_neighbors(
        t_idx, t_bear, t_dist, in_bounds=train_set, ensure_min_k=ensure_min_k_train, include_border=False
    )
    tr_attr_out = {k: (t_attr[k] if k in t_attr else v_attr[k]) for k in used_tr if (k in t_attr or k in v_attr)}

    # determine validation target/context sets
    if val_target_ids is not None and len(val_target_ids) > 0:
        val_targets = set(str(s) for s in val_target_ids) & set(val_set)
    else:
        ids_sorted = sorted(list(val_set))
        val_targets = set(ids_sorted[::4])  # likely error if too sparse
    val_context = set(val_set) - set(val_targets)

    # build fresh val->val edges: targets draw neighbors only from val_context
    need_latlon = list(val_context | set(val_targets))
    ll = _read_latlon(parquet_dir, need_latlon)

    va_idx_out: Dict[str, List[str]] = {}
    va_bear_out: Dict[str, List[float]] = {}
    va_dist_out: Dict[str, List[float]] = {}
    used_va: Set[str] = set()

    ctx_ids = [sid for sid in val_context if sid in ll]
    if ctx_ids:
        ctx_lat = np.array([ll[s][0] for s in ctx_ids], dtype=float)
        ctx_lon = np.array([ll[s][1] for s in ctx_ids], dtype=float)
    else:
        ctx_lat = np.zeros((0,), dtype=float)
        ctx_lon = np.zeros((0,), dtype=float)

    for to_id in sorted(list(val_targets)):
        if to_id not in ll:
            continue
        if ctx_lat.size == 0:
            continue
        tlat, tlon = ll[to_id]
        # distances to all context nodes
        dists = []
        for i, sid in enumerate(ctx_ids):
            dist_km, br = _haversine_bearing(tlat, tlon, ctx_lat[i], ctx_lon[i])
            dists.append((dist_km, br, sid))
        if not dists:
            continue
        dists.sort(key=lambda x: x[0])
        k = min(int(k_val), len(dists))
        sel = dists[:k]
        nbrs = [sid for (_, _, sid) in sel]
        bears = [float(br) for (_, br, _) in sel]
        d_scaled = [float((dk - d_bias) / d_scale + 5e-8) for (dk, _, _) in sel]
        if not nbrs:
            continue
        va_idx_out[to_id] = nbrs
        va_bear_out[to_id] = bears
        va_dist_out[to_id] = d_scaled
        used_va.add(to_id)
        used_va.update(nbrs)

    # assemble attribute dicts and id lists
    attr_map = dict(t_attr)
    attr_map.update(v_attr)
    va_attr_out = {k: attr_map[k] for k in used_va if k in attr_map}

    node_index = sorted(list(used_tr | used_va))
    train_ids = sorted(list(set(tr_idx_out.keys())))
    val_ids = sorted(list(set(va_idx_out.keys())))
    val_target_ids_out = sorted(list(set(va_idx_out.keys())))

    os.makedirs(output_dir, exist_ok=True)
    _write_json(os.path.join(output_dir, 'train_edge_index.json'), tr_idx_out)
    _write_json(os.path.join(output_dir, 'train_edge_bearing.json'), tr_bear_out)
    _write_json(os.path.join(output_dir, 'train_edge_distance.json'), tr_dist_out)
    _write_json(os.path.join(output_dir, 'train_edge_attr.json'), tr_attr_out)

    _write_json(os.path.join(output_dir, 'val_edge_index.json'), va_idx_out)
    _write_json(os.path.join(output_dir, 'val_edge_bearing.json'), va_bear_out)
    _write_json(os.path.join(output_dir, 'val_edge_distance.json'), va_dist_out)
    _write_json(os.path.join(output_dir, 'val_edge_attr.json'), va_attr_out)

    _write_json(os.path.join(output_dir, 'node_index.json'), node_index)
    _write_json(os.path.join(output_dir, 'train_ids.json'), train_ids)
    _write_json(os.path.join(output_dir, 'val_ids.json'), val_ids)
    _write_json(os.path.join(output_dir, 'val_target_ids.json'), val_target_ids_out)

    # build shapefile with edge geometries and attributes
    ll_all = _read_latlon(parquet_dir, list(used_tr | used_va))
    edge_lines = []
    to_, from_ = []
    train_flag = []
    bearing = []
    bearing_out = []
    distance_km = []

    # train edges (train=1)
    for to_id, nbrs in tr_idx_out.items():
        if to_id not in ll_all:
            continue
        lat_to, lon_to = ll_all[to_id]
        p_to = Point(float(lon_to), float(lat_to))
        for nb in nbrs:
            if nb not in ll_all:
                continue
            lat_nb, lon_nb = ll_all[nb]
            p_nb = Point(float(lon_nb), float(lat_nb))
            line = LineString([p_nb, p_to])
            dk, br = _haversine_bearing(lat_nb, lon_nb, lat_to, lon_to)
            edge_lines.append(line)
            to_.append(str(to_id))
            from_.append(str(nb))
            train_flag.append(1)
            bearing.append(float(br))
            bearing_out.append(float((br + 180.0) % 360.0))
            distance_km.append(float(dk))

    # val edges (train=0)
    for to_id, nbrs in va_idx_out.items():
        if to_id not in ll_all:
            continue
        lat_to, lon_to = ll_all[to_id]
        p_to = Point(float(lon_to), float(lat_to))
        for nb in nbrs:
            if nb not in ll_all:
                continue
            lat_nb, lon_nb = ll_all[nb]
            p_nb = Point(float(lon_nb), float(lat_nb))
            line = LineString([p_nb, p_to])
            dk, br = _haversine_bearing(lat_nb, lon_nb, lat_to, lon_to)
            edge_lines.append(line)
            to_.append(str(to_id))
            from_.append(str(nb))
            train_flag.append(0)
            bearing.append(float(br))
            bearing_out.append(float((br + 180.0) % 360.0))
            distance_km.append(float(dk))

    if edge_lines:
        gdf_edges = gpd.GeoDataFrame({'geometry': edge_lines})
        gdf_edges['to'] = to_
        gdf_edges['from'] = from_
        gdf_edges['train'] = train_flag
        gdf_edges['bearing'] = bearing
        gdf_edges['bearing_to_neighbor'] = bearing_out
        gdf_edges['distance_km'] = distance_km
        gdf_edges.crs = CRS.from_epsg(4326)
        gdf_edges.to_file(os.path.join(output_dir, 'edges.shp'), epsg='EPSG:4326', engine='fiona')

    meta_out = dict(meta) if isinstance(meta, dict) else {}
    three = {
        'outer_bounds': {'west': outer_bounds[0], 'south': outer_bounds[1], 'east': outer_bounds[2], 'north': outer_bounds[3]},
        'k_val': int(k_val),
        'ensure_min_k_train': int(ensure_min_k_train),
        'val_targets': val_target_ids_out,
        'source_graph_dir': os.path.abspath(source_graph_dir),
        'parquet_dir': os.path.abspath(parquet_dir),
    }
    if _looks_like_polygon(val_bounds):
        three['val_polygon'] = [(float(lat), float(lon)) for (lat, lon) in val_bounds]
    else:
        three['val_bounds'] = {'west': val_bounds[0], 'south': val_bounds[1], 'east': val_bounds[2], 'north': val_bounds[3]}
    meta_out['three_tier_subgraph'] = three
    _write_json(os.path.join(output_dir, 'graph_meta.json'), meta_out)


if __name__ == '__main__':
    target_var = 'tmax'
    training = '/data/ssd2/dads/training'
    source_graph_dir_ = os.path.join(training, 'graph', f'{target_var}_obs')
    parquet_dir_ = os.path.join(training, 'parquet', f'{target_var}_obs')
    output_dir_ = os.path.join(training, 'subgraph', f'{target_var}_obs')

    # training domain (CA and NV)
    west, south, east, north = (-124.5, 32.0, -112.0, 42.0)

    # validation hold out
    val_poly_ = [
        (38.0, -123.0),
        (36.5, -122.0),
        (38.0, -116.0),
        (40.8, -120.0),
    ]

    k_val_ = 10
    ensure_min_k_train_ = 1

    sub_graph_dir_ = os.path.join(training, 'subgraph', f'{target_var}_obs')
    validation_target_shp = os.path.join(sub_graph_dir_, 'validation_targets_30OCT2025.shp')
    val_target_gdf = gpd.read_file(validation_target_shp)
    val_target_ids_ = val_target_gdf['fid'].to_list()

    write_tiered_val_subgraph(
        source_graph_dir=source_graph_dir_,
        parquet_dir=parquet_dir_,
        output_dir=output_dir_,
        outer_bounds=(west, south, east, north),
        val_bounds=val_poly_,
        k_val=k_val_,
        ensure_min_k_train=ensure_min_k_train_,
        val_target_ids=val_target_ids_,
    )

# ========================= EOF ====================================================================

