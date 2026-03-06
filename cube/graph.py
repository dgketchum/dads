"""
Builder for graph.zarr — dense neighbor graph with edge/node attributes.

Replaces per-variable JSON graph artifacts (prep/graph.py) with a single zarr
store over all stations from stations.zarr.  Neighbor selection uses
elevation-aware spatial distance in EPSG:5070, matching the metric that
cell_station_index.zarr will use at inference time.

Output arrays
-------------
1-D coords (S,):
    station_id  — VLenUTF8, same order as stations.zarr
    is_train    — bool, deterministic train/val split
2-D arrays (S, K):
    neighbor_idx  — int32 indices into station dimension
    distance_km   — float32 Haversine distance
    bearing_sin   — float32 sin(bearing)
    bearing_cos   — float32 cos(bearing)
2-D arrays (S, F):
    node_attr     — float32 per-station scaled features
3-D arrays (S, K, F):
    edge_attr     — float32 (target − neighbor) feature deltas
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import zarr
from sklearn.neighbors import BallTree

from cube.config import get_compression

logger = logging.getLogger(__name__)

TARGET_VARIABLES = [
    "tmax_obs",
    "tmin_obs",
    "ea_obs",
    "prcp_obs",
    "rsds_obs",
    "wind_obs",
]

DEFAULT_GEO_FEATURES = ["rsun", "aspect", "elevation", "slope"]


# ---------------------------------------------------------------------------
# Worker helpers (run in child processes)
# ---------------------------------------------------------------------------


def _read_one_features(
    args: Tuple[str, str, List[str]],
) -> Optional[Tuple[str, np.ndarray]]:
    """Read GEO_FEATURES columns from one parquet, return time-mean vector.

    Parameters
    ----------
    args : tuple
        (parquet_path, station_id, feature_columns)
    """
    path, fid, columns = args
    try:
        df = pd.read_parquet(path, columns=columns)
    except Exception:
        return None
    present = [c for c in columns if c in df.columns]
    if not present:
        return None
    means = df[present].mean(numeric_only=True)
    result = np.array([means.get(c, np.nan) for c in columns], dtype=np.float64)
    return fid, result


# ---------------------------------------------------------------------------
# Parquet discovery
# ---------------------------------------------------------------------------


def _discover_parquet_paths(parquet_root: Path) -> Dict[str, Path]:
    """Scan target dirs and return {fid: first_hit_parquet_path}.

    Follows the same pattern as cube.stations.discover_stations: scan each
    TARGET_VARIABLES subdirectory and keep the first parquet found per fid.
    """
    fid_to_path: Dict[str, Path] = {}
    for target in TARGET_VARIABLES:
        target_dir = parquet_root / target
        if not target_dir.is_dir():
            continue
        for p in target_dir.glob("*.parquet"):
            fid = p.stem
            if fid not in fid_to_path:
                fid_to_path[fid] = p
    return fid_to_path


# ---------------------------------------------------------------------------
# Haversine helpers (vectorised)
# ---------------------------------------------------------------------------


def _haversine_distance_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Vectorised Haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = (np.radians(a) for a in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _haversine_bearing(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Vectorised initial bearing (radians) from point 1 → point 2."""
    lat1, lon1, lat2, lon2 = (np.radians(a) for a in (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(x, y)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_graph_zarr(
    stations_zarr: str,
    parquet_root: str,
    output_path: str,
    k: int = 10,
    elev_scale: float = 10.0,
    split_percent: float = 0.8,
    random_state: int = 42,
    geo_features: Optional[List[str]] = None,
    num_workers: int = 12,
    overwrite: bool = False,
) -> Path:
    """Build the graph.zarr store.

    Parameters
    ----------
    stations_zarr : str
        Path to the stations.zarr store (source of station metadata).
    parquet_root : str
        Root with per-target subdirectories containing training parquets.
    output_path : str
        Output zarr store path.
    k : int
        Number of nearest neighbours per station.
    elev_scale : float
        Multiplier on elevation (metres) for 3-D distance.
        elev_scale=10 → 200 m elevation diff ≈ 2 km horizontal.
    split_percent : float
        Fraction of stations assigned to training split.
    random_state : int
        Seed for deterministic train/val split.
    geo_features : list[str] or None
        Feature columns to read from parquets. Defaults to DEFAULT_GEO_FEATURES.
    num_workers : int
        Workers for parallel parquet reads.
    overwrite : bool
        If False, raise if store already exists.

    Returns
    -------
    Path to the created zarr store.
    """
    stations_zarr = Path(stations_zarr)
    parquet_root = Path(parquet_root)
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Store already exists: {output_path}. Use overwrite=True to rebuild."
        )

    if geo_features is None:
        geo_features = list(DEFAULT_GEO_FEATURES)

    n_features = len(geo_features)

    # ------------------------------------------------------------------
    # Step 1: Load station metadata from stations.zarr
    # ------------------------------------------------------------------
    logger.info("Step 1: loading station metadata from %s", stations_zarr)
    sz = zarr.open(str(stations_zarr), mode="r")
    station_ids = sz["station_id"][:]
    lats = sz["lat"][:].astype(np.float64)
    lons = sz["lon"][:].astype(np.float64)
    elevations = sz["elevation"][:].astype(np.float64)
    eastings = sz["easting"][:].astype(np.float64)
    northings = sz["northing"][:].astype(np.float64)
    n_stations = len(station_ids)
    logger.info("  %d stations loaded", n_stations)

    # ------------------------------------------------------------------
    # Step 2: Read edge features from training parquets
    # ------------------------------------------------------------------
    logger.info("Step 2: reading features from parquets (%d workers)", num_workers)
    fid_to_path = _discover_parquet_paths(parquet_root)
    logger.info("  %d parquets discovered", len(fid_to_path))

    features = np.full((n_stations, n_features), np.nan, dtype=np.float64)
    fid_to_idx = {fid: i for i, fid in enumerate(station_ids)}

    work_items = []
    for fid, path in fid_to_path.items():
        if fid in fid_to_idx:
            work_items.append((str(path), fid, geo_features))

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_read_one_features, item): item for item in work_items}
        for fut in as_completed(futures):
            result = fut.result()
            if result is None:
                continue
            fid, vec = result
            features[fid_to_idx[fid]] = vec

    n_found = np.sum(~np.all(np.isnan(features), axis=1))
    logger.info("  features loaded for %d / %d stations", n_found, n_stations)

    # ------------------------------------------------------------------
    # Step 3: Train/val split + scale features
    # ------------------------------------------------------------------
    logger.info("Step 3: splitting and scaling features")
    rng = np.random.RandomState(random_state)
    is_train = np.zeros(n_stations, dtype=bool)
    n_train = int(round(split_percent * n_stations))
    train_indices = rng.choice(n_stations, size=n_train, replace=False)
    is_train[train_indices] = True

    # Replace inf with NaN, then fill NaN with train-only column medians
    features[~np.isfinite(features)] = np.nan
    train_features = features[is_train]
    col_medians = np.nanmedian(train_features, axis=0)
    for f in range(n_features):
        if np.isnan(col_medians[f]):
            col_medians[f] = 0.0
    nan_mask = np.isnan(features)
    for f in range(n_features):
        features[nan_mask[:, f], f] = col_medians[f]

    # MinMax scaling fit on train only
    train_features_clean = features[is_train]
    scaler_min = train_features_clean.min(axis=0)
    scaler_scale = train_features_clean.max(axis=0) - scaler_min
    # Avoid division by zero for constant features
    scaler_scale[scaler_scale == 0] = 1.0

    node_attr = ((features - scaler_min) / scaler_scale).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 4: Build neighbor graph via elevation-aware distance
    # ------------------------------------------------------------------
    logger.info("Step 4: building BallTree (elev_scale=%.1f)", elev_scale)
    nan_elev = np.isnan(elevations)
    if nan_elev.any():
        median_elev = np.nanmedian(elevations)
        elevations[nan_elev] = median_elev
        logger.info(
            "  filled %d NaN elevations with median %.1f", nan_elev.sum(), median_elev
        )
    points_3d = np.column_stack([eastings, northings, elev_scale * elevations])
    tree = BallTree(points_3d, metric="euclidean")

    # Query extra candidates so we can skip self and co-located stations
    n_query = k + 20
    dist_3d, ind = tree.query(points_3d, k=min(n_query, n_stations))
    neighbor_idx = np.empty((n_stations, k), dtype=np.int32)
    n_colocated = 0
    for s in range(n_stations):
        chosen = []
        for j in range(ind.shape[1]):
            nb = ind[s, j]
            if nb == s:
                continue
            # Skip co-located: same lat/lon
            if lats[nb] == lats[s] and lons[nb] == lons[s]:
                n_colocated += 1
                continue
            chosen.append(nb)
            if len(chosen) == k:
                break
        # If not enough (very unlikely), pad with last valid or nearest
        while len(chosen) < k:
            chosen.append(chosen[-1] if chosen else 0)
        neighbor_idx[s] = chosen

    if n_colocated:
        logger.info("  skipped %d co-located neighbor edges", n_colocated)
    logger.info("  neighbour graph built")

    # ------------------------------------------------------------------
    # Step 5: Compute edge geometry (Haversine)
    # ------------------------------------------------------------------
    logger.info("Step 5: computing edge geometry")
    # Expand lat/lon for targets and neighbors: shapes (S, K)
    lat_target = lats[:, np.newaxis].repeat(k, axis=1)
    lon_target = lons[:, np.newaxis].repeat(k, axis=1)
    lat_neighbor = lats[neighbor_idx]
    lon_neighbor = lons[neighbor_idx]

    distance_km = _haversine_distance_km(
        lat_target, lon_target, lat_neighbor, lon_neighbor
    ).astype(np.float32)

    bearing = _haversine_bearing(lat_target, lon_target, lat_neighbor, lon_neighbor)
    bearing_sin = np.sin(bearing).astype(np.float32)
    bearing_cos = np.cos(bearing).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 6: Compute edge attributes (feature deltas)
    # ------------------------------------------------------------------
    logger.info("Step 6: computing edge attributes")
    # node_attr[neighbor_idx] has shape (S, K, F) via fancy indexing
    edge_attr = (node_attr[:, np.newaxis, :] - node_attr[neighbor_idx]).astype(
        np.float32
    )

    # ------------------------------------------------------------------
    # Step 7: Write zarr store
    # ------------------------------------------------------------------
    logger.info("Step 7: writing zarr store to %s", output_path)
    compressor = get_compression("float")
    store = zarr.open(str(output_path), mode="w")

    # station_id
    store.create_dataset(
        "station_id",
        shape=station_ids.shape,
        data=station_ids,
        dtype=str,
        chunks=(min(1000, n_stations),),
    )

    # is_train
    store.create_dataset(
        "is_train",
        shape=is_train.shape,
        data=is_train,
        dtype=bool,
        chunks=(n_stations,),
    )

    # neighbor_idx
    store.create_dataset(
        "neighbor_idx",
        shape=neighbor_idx.shape,
        data=neighbor_idx,
        dtype="int32",
        chunks=(n_stations, k),
        compressors=get_compression("int"),
    )

    # distance_km
    store.create_dataset(
        "distance_km",
        shape=distance_km.shape,
        data=distance_km,
        dtype="float32",
        chunks=(n_stations, k),
        compressors=compressor,
    )

    # bearing_sin
    store.create_dataset(
        "bearing_sin",
        shape=bearing_sin.shape,
        data=bearing_sin,
        dtype="float32",
        chunks=(n_stations, k),
        compressors=compressor,
    )

    # bearing_cos
    store.create_dataset(
        "bearing_cos",
        shape=bearing_cos.shape,
        data=bearing_cos,
        dtype="float32",
        chunks=(n_stations, k),
        compressors=compressor,
    )

    # node_attr
    store.create_dataset(
        "node_attr",
        shape=node_attr.shape,
        data=node_attr,
        dtype="float32",
        chunks=(n_stations, n_features),
        compressors=compressor,
    )

    # edge_attr
    store.create_dataset(
        "edge_attr",
        shape=edge_attr.shape,
        data=edge_attr,
        dtype="float32",
        chunks=(n_stations, k, n_features),
        compressors=compressor,
    )

    # Root attrs
    store.attrs["k"] = k
    store.attrs["elev_scale"] = elev_scale
    store.attrs["split_percent"] = split_percent
    store.attrs["random_state"] = random_state
    store.attrs["feature_names"] = geo_features
    store.attrs["scaler_min"] = scaler_min.tolist()
    store.attrs["scaler_scale"] = scaler_scale.tolist()

    logger.info("Done — graph.zarr at %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_graph_zarr(store_path: str) -> Dict[str, bool]:
    """Run basic integrity checks on a graph.zarr store.

    Returns dict mapping check names to pass/fail booleans.
    """
    store_path = Path(store_path)
    checks: Dict[str, bool] = {}

    checks["exists"] = store_path.exists()
    if not checks["exists"]:
        return checks

    store = zarr.open(str(store_path), mode="r")

    required = [
        "station_id",
        "is_train",
        "neighbor_idx",
        "distance_km",
        "bearing_sin",
        "bearing_cos",
        "node_attr",
        "edge_attr",
    ]
    for name in required:
        checks[f"{name}_exists"] = name in store

    if not all(checks.get(f"{n}_exists", False) for n in required):
        return checks

    s = store["station_id"].shape[0]
    k = store["neighbor_idx"].shape[1] if store["neighbor_idx"].ndim == 2 else 0
    f = store["node_attr"].shape[1] if store["node_attr"].ndim == 2 else 0

    checks["station_id_shape"] = store["station_id"].shape == (s,)
    checks["is_train_shape"] = store["is_train"].shape == (s,)
    checks["neighbor_idx_shape"] = store["neighbor_idx"].shape == (s, k)
    checks["distance_km_shape"] = store["distance_km"].shape == (s, k)
    checks["bearing_sin_shape"] = store["bearing_sin"].shape == (s, k)
    checks["bearing_cos_shape"] = store["bearing_cos"].shape == (s, k)
    checks["node_attr_shape"] = store["node_attr"].shape == (s, f)
    checks["edge_attr_shape"] = store["edge_attr"].shape == (s, k, f)

    # dtype checks
    checks["neighbor_idx_dtype"] = store["neighbor_idx"].dtype == np.int32
    checks["distance_km_dtype"] = store["distance_km"].dtype == np.float32
    checks["is_train_dtype"] = store["is_train"].dtype == bool

    # Value range checks
    neighbor_idx = store["neighbor_idx"][:]
    checks["neighbor_idx_range"] = bool(
        np.all(neighbor_idx >= 0) and np.all(neighbor_idx < s)
    )

    distance_km = store["distance_km"][:]
    checks["distance_positive"] = bool(np.all(distance_km > 0))

    bearing_sin = store["bearing_sin"][:]
    bearing_cos = store["bearing_cos"][:]
    checks["bearing_sin_range"] = bool(
        np.all(bearing_sin >= -1.0) and np.all(bearing_sin <= 1.0)
    )
    checks["bearing_cos_range"] = bool(
        np.all(bearing_cos >= -1.0) and np.all(bearing_cos <= 1.0)
    )

    # Required root attrs
    for attr in ["k", "elev_scale", "split_percent", "random_state", "feature_names"]:
        checks[f"attr_{attr}"] = attr in store.attrs

    return checks
