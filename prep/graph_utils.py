"""
Shared graph-building utilities for GNN pipelines.

Contains k-NN neighbor map construction, static edge attribute computation,
per-day edge tensor building, and edge normalization — used by both scalar
and wind bias-correction models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import BallTree

EDGE_STATIC_COLS = [
    "distance_norm",
    "bearing_sin",
    "bearing_cos",
    "delta_elevation",
    "delta_tpi",
]
EDGE_DYNAMIC_COLS = ["upwind_cos", "upwind_sin"]


def compute_edge_norm(
    static_edges: dict[str, dict[str, dict[str, float]]],
) -> dict[str, float]:
    """Compute distance and delta-elevation normalization stats from static edges.

    Returns {dist_mean, dist_std, delev_mean, delev_std}.
    """
    all_dists = [
        ea["distance_km"]
        for fid_attrs in static_edges.values()
        for ea in fid_attrs.values()
    ]
    all_delev = [
        ea["delta_elevation"]
        for fid_attrs in static_edges.values()
        for ea in fid_attrs.values()
    ]
    return {
        "dist_mean": float(np.mean(all_dists)) if all_dists else 1.0,
        "dist_std": float(max(np.std(all_dists), 1e-6)),
        "delev_mean": float(np.mean(all_delev)) if all_delev else 0.0,
        "delev_std": float(max(np.std(all_delev), 1e-6)),
    }


def build_knn_map(
    stations_csv: str,
    k: int = 16,
    max_radius_km: float = 150.0,
) -> dict[str, list[str]]:
    """Build a k-NN neighbor map from station inventory.

    Returns {fid: [neighbor_fids]}.
    """
    stations = pd.read_csv(stations_csv)
    id_col = "station_id" if "station_id" in stations.columns else "fid"
    fids = stations[id_col].astype(str).values
    coords = np.radians(stations[["latitude", "longitude"]].values)

    tree = BallTree(coords, metric="haversine")
    max_rad = max_radius_km / 6371.0  # convert km to radians

    kq = min(k + 1, len(fids))
    dists, indices = tree.query(coords, k=kq)

    knn_map: dict[str, list[str]] = {}
    for i, fid in enumerate(fids):
        nbrs = []
        for j_idx in range(kq):
            j = indices[i, j_idx]
            if j == i:
                continue
            if dists[i, j_idx] > max_rad:
                continue
            nbrs.append(str(fids[j]))
            if len(nbrs) >= k:
                break
        knn_map[str(fid)] = nbrs
    return knn_map


def build_static_edge_attrs(
    stations_csv: str,
    knn_map: dict[str, list[str]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute static edge attributes: distance, bearing, delta_elev, delta_tpi.

    Returns {fid_i: {fid_j: {attr: value}}}.
    """
    stations = pd.read_csv(stations_csv)
    id_col = "station_id" if "station_id" in stations.columns else "fid"
    stations[id_col] = stations[id_col].astype(str)
    sta = stations.set_index(id_col)

    edge_attrs: dict[str, dict[str, dict[str, float]]] = {}
    for fid_i, nbrs in knn_map.items():
        if fid_i not in sta.index:
            continue
        lat_i = np.radians(sta.loc[fid_i, "latitude"])
        lon_i = np.radians(sta.loc[fid_i, "longitude"])
        elev_i = sta.loc[fid_i, "elevation"] if "elevation" in sta.columns else 0.0

        edge_attrs[fid_i] = {}
        for fid_j in nbrs:
            if fid_j not in sta.index:
                continue
            lat_j = np.radians(sta.loc[fid_j, "latitude"])
            lon_j = np.radians(sta.loc[fid_j, "longitude"])
            elev_j = sta.loc[fid_j, "elevation"] if "elevation" in sta.columns else 0.0

            # Haversine distance
            dlat = lat_j - lat_i
            dlon = lon_j - lon_i
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat_i) * np.cos(lat_j) * np.sin(dlon / 2) ** 2
            )
            dist_km = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            # Bearing
            x = np.sin(dlon) * np.cos(lat_j)
            y = np.cos(lat_i) * np.sin(lat_j) - np.sin(lat_i) * np.cos(lat_j) * np.cos(
                dlon
            )
            brng_rad = np.arctan2(x, y)

            edge_attrs[fid_i][fid_j] = {
                "distance_km": float(dist_km),
                "bearing_sin": float(np.sin(brng_rad)),
                "bearing_cos": float(np.cos(brng_rad)),
                "delta_elevation": float(elev_j - elev_i),
            }
    return edge_attrs


def build_edges_for_day(
    fids: list[str],
    ugrd: np.ndarray | None,
    vgrd: np.ndarray | None,
    knn_map: dict[str, list[str]],
    static_edges: dict[str, dict[str, dict[str, float]]],
    edge_norm: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build edge_index and edge_attr for one day's active stations.

    Parameters
    ----------
    fids : list of station IDs present this day
    ugrd, vgrd : gridded u/v wind components per station (same order as fids)
    knn_map : {fid: [neighbor_fids]}
    static_edges : {fid_i: {fid_j: {attr: val}}}
    edge_norm : {dist_mean, dist_std, delev_mean, delev_std}

    Returns
    -------
    edge_index : (2, E) int64 tensor
    edge_attr : (E, 7) float32 tensor
    """
    fid_to_idx = {f: i for i, f in enumerate(fids)}
    src_list, dst_list = [], []
    edge_feats = []

    for i, fid_i in enumerate(fids):
        nbrs = knn_map.get(fid_i, [])
        u_i = float(ugrd[i]) if ugrd is not None else 0.0
        v_i = float(vgrd[i]) if vgrd is not None else 0.0
        theta_from = np.arctan2(-u_i, -v_i)

        for fid_j in nbrs:
            if fid_j not in fid_to_idx:
                continue
            j = fid_to_idx[fid_j]

            ea = static_edges.get(fid_i, {}).get(fid_j, None)
            if ea is None:
                continue

            dist_norm = (ea["distance_km"] - edge_norm["dist_mean"]) / edge_norm[
                "dist_std"
            ]
            brng_sin = ea["bearing_sin"]
            brng_cos = ea["bearing_cos"]
            d_elev = (ea["delta_elevation"] - edge_norm["delev_mean"]) / edge_norm[
                "delev_std"
            ]
            d_tpi = 0.0

            brng_rad = np.arctan2(brng_sin, brng_cos)
            upwind_cos = float(np.cos(theta_from - brng_rad))
            upwind_sin = float(np.sin(theta_from - brng_rad))

            src_list.append(j)
            dst_list.append(i)
            edge_feats.append(
                [dist_norm, brng_sin, brng_cos, d_elev, d_tpi, upwind_cos, upwind_sin]
            )

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 7), dtype=torch.float32)

    return edge_index, edge_attr
