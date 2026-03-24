"""
Heterogeneous HRRR data-assimilation dataset.

Each sample is a local graph centered on one supervised target location:

- one patch of grid nodes from a daily gridded background raster
- nearby station nodes carrying innovations against that background
- station -> grid edges for observation influence
- grid -> grid edges for spatial propagation

The target lives on the center grid node, not on the station nodes. That keeps
the supervision aligned with the analysis objective and avoids trivial
observation copying.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import rasterio
import torch
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


DEFAULT_TARGET_NAMES = [
    "delta_tmax",
    "delta_tmin",
    "delta_ea",
    "delta_rsds",
    "delta_w_par",
    "delta_w_perp",
]

DEFAULT_STATION_FEATURE_CANDIDATES = [
    "ugrd_hrrr",
    "vgrd_hrrr",
    "wind_hrrr",
    "tmp_hrrr",
    "dpt_hrrr",
    "pres_hrrr",
    "tcdc_hrrr",
    "ea_hrrr",
    "dswrf_hrrr",
    "hpbl_hrrr",
    "spfh_hrrr",
    "tmp_dpt_diff",
    "delta_tmax",
    "delta_tmin",
    "delta_ea",
    "delta_rsds",
    "delta_w_par",
    "delta_w_perp",
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "tpi_4",
    "tpi_10",
    "flow_upslope",
    "flow_cross",
    "wind_aligned_sx",
    "doy_sin",
    "doy_cos",
    "latitude",
    "longitude",
]


def _doy_features(day: pd.Timestamp) -> tuple[float, float]:
    doy = int(day.dayofyear)
    return (
        float(np.sin(2.0 * np.pi * doy / 365.25)),
        float(np.cos(2.0 * np.pi * doy / 365.25)),
    )


def _haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    return 6371.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _bearing_sin_cos(
    lat_from: np.ndarray,
    lon_from: np.ndarray,
    lat_to: np.ndarray,
    lon_to: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lat1 = np.radians(lat_from)
    lon1 = np.radians(lon_from)
    lat2 = np.radians(lat_to)
    lon2 = np.radians(lon_to)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.arctan2(x, y)
    return np.sin(brng), np.cos(brng)


def _apply_norm(
    arr: np.ndarray,
    cols: list[str],
    stats: dict[str, dict[str, float]],
) -> np.ndarray:
    """Z-score normalise columns of a float32 array in-place (copy returned)."""
    out = arr.astype("float32", copy=True)
    for i, col in enumerate(cols):
        if col in stats:
            out[:, i] = (out[:, i] - stats[col]["mean"]) / stats[col]["std"]
    return out


def _discover_band_names(path: str, fallback_prefix: str) -> list[str]:
    with rasterio.open(path) as src:
        desc = list(src.descriptions) if src.descriptions else []
        if desc and any(d for d in desc):
            return [d if d else f"{fallback_prefix}_{i}" for i, d in enumerate(desc)]
        return [f"{fallback_prefix}_{i}" for i in range(src.count)]


class _RasterCache:
    """Tiny LRU cache for daily rasters and static rasters."""

    def __init__(self, max_items: int = 4):
        self.max_items = max_items
        self._cache: OrderedDict[str, dict] = OrderedDict()

    def get(self, path: str) -> dict:
        if path in self._cache:
            item = self._cache.pop(path)
            self._cache[path] = item
            return item

        with rasterio.open(path) as src:
            item = {
                "data": src.read().astype("float32"),
                "transform": src.transform,
                "crs": src.crs,
                "descriptions": list(src.descriptions)
                if src.descriptions
                else [None] * src.count,
            }

        self._cache[path] = item
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return item


class HRRRHeteroTileDataset(Dataset):
    """Build local hetero graphs centered on supervised station locations."""

    def __init__(
        self,
        table_path: str,
        background_dir: str,
        background_pattern: str = "HRRR_1km_{date}.tif",
        terrain_tif: str | None = None,
        target_names: list[str] | None = None,
        station_feature_cols: list[str] | None = None,
        background_feature_names: list[str] | None = None,
        train_days: set | None = None,
        target_include_fids: set[str] | None = None,
        target_exclude_fids: set[str] | None = None,
        neighbor_exclude_fids: set[str] | None = None,
        grid_radius_cells: int = 2,
        station_radius_km: float = 150.0,
        max_neighbor_stations: int = 16,
        cache_size: int = 4,
    ):
        super().__init__()
        self.background_dir = background_dir
        self.background_pattern = background_pattern
        self.terrain_tif = terrain_tif
        self.target_names = target_names or list(DEFAULT_TARGET_NAMES)
        self.grid_radius_cells = int(grid_radius_cells)
        self.station_radius_km = float(station_radius_km)
        self.max_neighbor_stations = int(max_neighbor_stations)
        self.raster_cache = _RasterCache(max_items=cache_size)

        df = pd.read_parquet(table_path)
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        df["fid"] = df["fid"].astype(str)
        df["day"] = pd.to_datetime(df["day"]).dt.normalize()

        if train_days is not None:
            day_set = {pd.Timestamp(d).normalize() for d in train_days}
            df = df[df["day"].isin(day_set)]
        if target_include_fids is not None:
            df = df[df["fid"].isin(target_include_fids)]
        if target_exclude_fids is not None:
            df = df[~df["fid"].isin(target_exclude_fids)]

        sample_mask = df[self.target_names].notna().any(axis=1)
        df = df[sample_mask].copy()

        # Keep only days with available background rasters.
        bg_exists = df["day"].map(lambda d: os.path.exists(self._background_path(d)))
        df = df[bg_exists].copy()
        if df.empty:
            raise ValueError("No supervised samples remain after raster/day filtering.")

        # Neighbor pool is day-filtered, but excludes withheld stations if requested.
        neighbor_df = pd.read_parquet(table_path)
        if isinstance(neighbor_df.index, pd.MultiIndex):
            neighbor_df = neighbor_df.reset_index()
        neighbor_df["fid"] = neighbor_df["fid"].astype(str)
        neighbor_df["day"] = pd.to_datetime(neighbor_df["day"]).dt.normalize()
        if train_days is not None:
            day_set = {pd.Timestamp(d).normalize() for d in train_days}
            neighbor_df = neighbor_df[neighbor_df["day"].isin(day_set)]
        if neighbor_exclude_fids is not None:
            neighbor_df = neighbor_df[~neighbor_df["fid"].isin(neighbor_exclude_fids)]

        self.samples = df.sort_values("day").reset_index(drop=True)
        self._neighbor_day_groups = {
            pd.Timestamp(day).normalize(): grp.reset_index(drop=True)
            for day, grp in neighbor_df.groupby("day")
        }

        self.station_feature_cols = [
            c
            for c in (
                station_feature_cols
                or [
                    c
                    for c in DEFAULT_STATION_FEATURE_CANDIDATES
                    if c in self.samples.columns
                ]
            )
            if c in self.samples.columns
        ]
        self.station_mask_cols = [
            f"{name}_is_valid"
            for name in self.target_names
            if name in self.station_feature_cols
        ]
        self.station_node_dim = len(self.station_feature_cols) + len(
            self.station_mask_cols
        )

        example_bg = self._background_path(self.samples.iloc[0]["day"])
        self.background_feature_names = (
            background_feature_names or _discover_band_names(example_bg, "bg")
        )
        if terrain_tif is not None:
            self.terrain_feature_names = _discover_band_names(terrain_tif, "terrain")
        else:
            self.terrain_feature_names = []
        self.grid_feature_names = (
            list(self.background_feature_names)
            + list(self.terrain_feature_names)
            + ["doy_sin", "doy_cos", "latitude", "longitude"]
        )
        self.grid_node_dim = len(self.grid_feature_names)
        self.edge_dim = 7

        # Feature stats come from station samples; grid features share the same semantics.
        self.grid_norm_stats = self._compute_norm_stats(
            self.samples,
            [c for c in self.grid_feature_names if c in self.samples.columns],
        )
        self.station_norm_stats = self._compute_norm_stats(
            self.samples,
            [c for c in self.station_feature_cols if c in self.samples.columns],
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _background_path(self, day: pd.Timestamp) -> str:
        return os.path.join(
            self.background_dir,
            self.background_pattern.format(date=pd.Timestamp(day).strftime("%Y%m%d")),
        )

    @staticmethod
    def _compute_norm_stats(
        df: pd.DataFrame, feature_cols: list[str]
    ) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for col in feature_cols:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if vals.empty:
                continue
            stats[col] = {
                "mean": float(vals.mean()),
                "std": float(max(vals.std(), 1e-8)),
            }
        return stats

    @staticmethod
    def _apply_norm(
        arr: np.ndarray,
        cols: list[str],
        stats: dict[str, dict[str, float]],
    ) -> np.ndarray:
        return _apply_norm(arr, cols, stats)

    def _sample_grid_patch(
        self, day: pd.Timestamp, lon: float, lat: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict]:
        bg = self.raster_cache.get(self._background_path(day))
        bg_data = bg["data"]
        bg_tf = bg["transform"]
        bg_crs = bg["crs"]
        if bg_data.shape[0] != len(self.background_feature_names):
            raise ValueError(
                "Background feature names do not match raster band count: "
                f"{bg_data.shape[0]} vs {len(self.background_feature_names)}"
            )

        to_bg = Transformer.from_crs("EPSG:4326", bg_crs, always_xy=True)
        x_target, y_target = to_bg.transform(lon, lat)
        r_center, c_center = rowcol(bg_tf, x_target, y_target)
        r0 = max(0, r_center - self.grid_radius_cells)
        r1 = min(bg_data.shape[1], r_center + self.grid_radius_cells + 1)
        c0 = max(0, c_center - self.grid_radius_cells)
        c1 = min(bg_data.shape[2], c_center + self.grid_radius_cells + 1)

        rows = np.arange(r0, r1)
        cols = np.arange(c0, c1)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        rr_flat = rr.ravel()
        cc_flat = cc.ravel()

        bg_feats = bg_data[:, rr_flat, cc_flat].T.astype("float32")

        x_proj, y_proj = xy(bg_tf, rr_flat, cc_flat, offset="center")
        x_proj = np.asarray(x_proj, dtype="float64")
        y_proj = np.asarray(y_proj, dtype="float64")
        to_ll = Transformer.from_crs(bg_crs, "EPSG:4326", always_xy=True)
        grid_lon, grid_lat = to_ll.transform(x_proj, y_proj)
        grid_lon = np.asarray(grid_lon, dtype="float32")
        grid_lat = np.asarray(grid_lat, dtype="float32")

        feats = [bg_feats]

        if self.terrain_tif is not None:
            terrain = self.raster_cache.get(self.terrain_tif)
            t_data = terrain["data"]
            t_tf = terrain["transform"]
            t_crs = terrain["crs"]
            if t_data.shape[0] != len(self.terrain_feature_names):
                raise ValueError(
                    "Terrain feature names do not match raster band count: "
                    f"{t_data.shape[0]} vs {len(self.terrain_feature_names)}"
                )
            if str(t_crs) == str(bg_crs):
                tr, tc = rowcol(t_tf, x_proj, y_proj)
            else:
                to_terrain = Transformer.from_crs(bg_crs, t_crs, always_xy=True)
                tx, ty = to_terrain.transform(x_proj, y_proj)
                tr, tc = rowcol(t_tf, tx, ty)
            tr = np.clip(np.asarray(tr, dtype=int), 0, t_data.shape[1] - 1)
            tc = np.clip(np.asarray(tc, dtype=int), 0, t_data.shape[2] - 1)
            terrain_feats = t_data[:, tr, tc].T.astype("float32")
            feats.append(terrain_feats)

        doy_sin, doy_cos = _doy_features(day)
        temporal = np.full((len(rr_flat), 2), [doy_sin, doy_cos], dtype="float32")
        loc = np.column_stack([grid_lat, grid_lon]).astype("float32")

        feats.extend([temporal, loc])
        grid_x = np.concatenate(feats, axis=1)

        patch_h = r1 - r0
        patch_w = c1 - c0
        center_index = int((r_center - r0) * patch_w + (c_center - c0))

        meta = {
            "rows": rr_flat,
            "cols": cc_flat,
            "patch_h": patch_h,
            "patch_w": patch_w,
            "bg_feature_to_idx": {
                name: i for i, name in enumerate(self.background_feature_names)
            },
            "grid_lon": grid_lon,
            "grid_lat": grid_lat,
        }
        return grid_x, grid_lat, grid_lon, center_index, meta

    def _build_station_nodes(
        self,
        target_row: pd.Series,
        day_group: pd.DataFrame | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if day_group is None or day_group.empty:
            return (
                np.zeros((0, self.station_node_dim), dtype="float32"),
                np.zeros(0, dtype="float32"),
                np.zeros(0, dtype="float32"),
                np.zeros(0, dtype="float32"),
            )

        target_lat = float(target_row["latitude"])
        target_lon = float(target_row["longitude"])
        mask_not_self = day_group["fid"] != str(target_row["fid"])
        cand = day_group.loc[mask_not_self].copy()
        if cand.empty:
            return (
                np.zeros((0, self.station_node_dim), dtype="float32"),
                np.zeros(0, dtype="float32"),
                np.zeros(0, dtype="float32"),
                np.zeros(0, dtype="float32"),
            )

        d_km = _haversine_km(
            np.full(len(cand), target_lat),
            np.full(len(cand), target_lon),
            cand["latitude"].to_numpy(dtype=float),
            cand["longitude"].to_numpy(dtype=float),
        )
        cand["__dist_km"] = d_km
        cand = cand[cand["__dist_km"] <= self.station_radius_km]
        if cand.empty:
            return (
                np.zeros((0, self.station_node_dim), dtype="float32"),
                np.zeros(0, dtype="float32"),
                np.zeros(0, dtype="float32"),
                np.zeros(0, dtype="float32"),
            )

        cand = cand.sort_values("__dist_km").head(self.max_neighbor_stations)
        base = cand[self.station_feature_cols].apply(pd.to_numeric, errors="coerce")
        base_arr = base.fillna(0.0).to_numpy(dtype="float32")

        mask_arr = []
        for name in self.target_names:
            if name in self.station_feature_cols:
                mask_arr.append(cand[name].notna().to_numpy(dtype="float32"))
        if mask_arr:
            mask_stack = np.column_stack(mask_arr).astype("float32")
            station_x = np.concatenate([base_arr, mask_stack], axis=1)
        else:
            station_x = base_arr

        station_x = self._apply_norm(
            station_x,
            self.station_feature_cols + self.station_mask_cols,
            self.station_norm_stats,
        )
        return (
            station_x.astype("float32"),
            cand["latitude"].to_numpy(dtype="float32"),
            cand["longitude"].to_numpy(dtype="float32"),
            cand["elevation"].fillna(0.0).to_numpy(dtype="float32"),
        )

    def _build_grid_edges(
        self,
        patch_h: int,
        patch_w: int,
        grid_lat: np.ndarray,
        grid_lon: np.ndarray,
        grid_x: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat_map = {name: i for i, name in enumerate(self.grid_feature_names)}
        elev_idx = feat_map.get("elevation")
        ugrd_idx = feat_map.get("ugrd_hrrr")
        vgrd_idx = feat_map.get("vgrd_hrrr")

        idx = np.arange(patch_h * patch_w, dtype=np.int64).reshape(patch_h, patch_w)
        h_src = np.concatenate([idx[:, :-1].ravel(), idx[:, 1:].ravel()])
        h_dst = np.concatenate([idx[:, 1:].ravel(), idx[:, :-1].ravel()])
        v_src = np.concatenate([idx[:-1, :].ravel(), idx[1:, :].ravel()])
        v_dst = np.concatenate([idx[1:, :].ravel(), idx[:-1, :].ravel()])
        src_all = np.concatenate([h_src, v_src])
        dst_all = np.concatenate([h_dst, v_dst])

        if len(src_all) == 0:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, self.edge_dim), dtype=torch.float32),
            )

        s_lat_r = np.radians(grid_lat[src_all])
        s_lon_r = np.radians(grid_lon[src_all])
        d_lat_r = np.radians(grid_lat[dst_all])
        d_lon_r = np.radians(grid_lon[dst_all])

        dlat = d_lat_r - s_lat_r
        dlon = d_lon_r - s_lon_r
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(s_lat_r) * np.cos(d_lat_r) * np.sin(dlon / 2) ** 2
        )
        dist_km = 6371.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

        bx = np.sin(dlon) * np.cos(d_lat_r)
        by = np.cos(s_lat_r) * np.sin(d_lat_r) - np.sin(s_lat_r) * np.cos(
            d_lat_r
        ) * np.cos(dlon)
        brng = np.arctan2(bx, by)
        br_sin = np.sin(brng)
        br_cos = np.cos(brng)

        delta_elev = (
            (grid_x[src_all, elev_idx] - grid_x[dst_all, elev_idx]).astype("float32")
            if elev_idx is not None
            else np.zeros(len(src_all), dtype="float32")
        )

        if ugrd_idx is not None and vgrd_idx is not None:
            u = grid_x[dst_all, ugrd_idx].astype("float64")
            v = grid_x[dst_all, vgrd_idx].astype("float64")
            theta = np.arctan2(-u, -v)
            uw_cos = np.cos(theta - brng).astype("float32")
            uw_sin = np.sin(theta - brng).astype("float32")
        else:
            uw_cos = np.zeros(len(src_all), dtype="float32")
            uw_sin = np.zeros(len(src_all), dtype="float32")

        edge_attr = np.column_stack(
            [
                (dist_km / 5.0).astype("float32"),
                br_sin.astype("float32"),
                br_cos.astype("float32"),
                delta_elev,
                np.zeros(len(src_all), dtype="float32"),
                uw_cos,
                uw_sin,
            ]
        )
        return (
            torch.from_numpy(np.stack([src_all, dst_all])),
            torch.from_numpy(edge_attr),
        )

    def _build_station_grid_edges(
        self,
        station_lat: np.ndarray,
        station_lon: np.ndarray,
        station_elev: np.ndarray,
        grid_lat: np.ndarray,
        grid_lon: np.ndarray,
        grid_x: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(station_lat) == 0 or len(grid_lat) == 0:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, self.edge_dim), dtype=torch.float32),
            )

        feat_map = {name: i for i, name in enumerate(self.grid_feature_names)}
        elev_idx = feat_map.get("elevation")
        ugrd_idx = feat_map.get("ugrd_hrrr")
        vgrd_idx = feat_map.get("vgrd_hrrr")

        n_s = len(station_lat)
        n_g = len(grid_lat)

        # All (station, grid) pairs: shape (n_s * n_g,)
        s_idx = np.repeat(np.arange(n_s, dtype=np.int64), n_g)
        g_idx = np.tile(np.arange(n_g, dtype=np.int64), n_s)

        s_lat_r = np.radians(station_lat[s_idx].astype("float64"))
        s_lon_r = np.radians(station_lon[s_idx].astype("float64"))
        g_lat_r = np.radians(grid_lat[g_idx].astype("float64"))
        g_lon_r = np.radians(grid_lon[g_idx].astype("float64"))

        dlat = g_lat_r - s_lat_r
        dlon = g_lon_r - s_lon_r
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(s_lat_r) * np.cos(g_lat_r) * np.sin(dlon / 2) ** 2
        )
        dist_km = 6371.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

        bx = np.sin(dlon) * np.cos(g_lat_r)
        by = np.cos(s_lat_r) * np.sin(g_lat_r) - np.sin(s_lat_r) * np.cos(
            g_lat_r
        ) * np.cos(dlon)
        brng = np.arctan2(bx, by)
        br_sin = np.sin(brng)
        br_cos = np.cos(brng)

        delta_elev = (
            (station_elev[s_idx] - grid_x[g_idx, elev_idx]).astype("float32")
            if elev_idx is not None
            else np.zeros(n_s * n_g, dtype="float32")
        )

        if ugrd_idx is not None and vgrd_idx is not None:
            u = grid_x[g_idx, ugrd_idx].astype("float64")
            v = grid_x[g_idx, vgrd_idx].astype("float64")
            theta = np.arctan2(-u, -v)
            uw_cos = np.cos(theta - brng).astype("float32")
            uw_sin = np.sin(theta - brng).astype("float32")
        else:
            uw_cos = np.zeros(n_s * n_g, dtype="float32")
            uw_sin = np.zeros(n_s * n_g, dtype="float32")

        edge_attr = np.column_stack(
            [
                (dist_km / max(self.station_radius_km, 1.0)).astype("float32"),
                br_sin.astype("float32"),
                br_cos.astype("float32"),
                delta_elev,
                np.zeros(n_s * n_g, dtype="float32"),
                uw_cos,
                uw_sin,
            ]
        )
        return (
            torch.from_numpy(np.stack([s_idx, g_idx])),
            torch.from_numpy(edge_attr),
        )

    def __getitem__(self, idx: int) -> HeteroData:
        row = self.samples.iloc[idx]
        day = pd.Timestamp(row["day"]).normalize()
        target_lat = float(row["latitude"])
        target_lon = float(row["longitude"])

        grid_x_raw, grid_lat, grid_lon, center_idx, meta = self._sample_grid_patch(
            day, target_lon, target_lat
        )
        grid_x = self._apply_norm(
            grid_x_raw, self.grid_feature_names, self.grid_norm_stats
        )

        day_group = self._neighbor_day_groups.get(day)
        station_x, station_lat, station_lon, station_elev = self._build_station_nodes(
            row, day_group
        )

        gg_edge_index, gg_edge_attr = self._build_grid_edges(
            meta["patch_h"], meta["patch_w"], grid_lat, grid_lon, grid_x_raw
        )
        sg_edge_index, sg_edge_attr = self._build_station_grid_edges(
            station_lat, station_lon, station_elev, grid_lat, grid_lon, grid_x_raw
        )

        n_grid = grid_x.shape[0]
        y = np.zeros((n_grid, len(self.target_names)), dtype="float32")
        target_mask = np.zeros((n_grid, len(self.target_names)), dtype=bool)
        for i, name in enumerate(self.target_names):
            val = row.get(name, np.nan)
            if pd.notna(val):
                y[center_idx, i] = float(val)
                target_mask[center_idx, i] = True

        data = HeteroData()
        data["grid"].x = torch.from_numpy(grid_x.astype("float32"))
        data["grid"].y = torch.from_numpy(y)
        data["grid"].target_mask = torch.from_numpy(target_mask)
        data["grid"].center_idx = torch.tensor([center_idx], dtype=torch.long)
        data["station"].x = torch.from_numpy(station_x.astype("float32"))
        data["grid", "neighbors", "grid"].edge_index = gg_edge_index
        data["grid", "neighbors", "grid"].edge_attr = gg_edge_attr
        data["station", "influences", "grid"].edge_index = sg_edge_index
        data["station", "influences", "grid"].edge_attr = sg_edge_attr
        return data
