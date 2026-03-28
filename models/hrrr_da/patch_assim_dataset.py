"""
Patch-based HRRR assimilation dataset for E0.

Each sample is a (day, patch) pair:
- A 64×64 km raster patch (HRRR background + terrain + Landsat) centered on a station.
- All stations whose pixel coordinates fall within the patch provide supervision targets.
- No observation channels are used (E0: raster-only bias learning).

The model predicts a full correction field over the patch; the loss is computed only at
station pixel locations. TV regularization keeps the field smooth between stations.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import rasterio
import torch
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from torch.utils.data import Dataset

from models.hrrr_da.hetero_dataset import (
    DEFAULT_TARGET_NAMES,
    _RasterCache,
    _discover_band_names,
    _doy_features,
)

_LANDSAT_BANDS = 7  # B2, B3, B4, B5, B6, B7, B10 per period
_LANDSAT_PERIODS = 5
# DOY upper bounds for periods P0..P3; P4 catches the remainder.
# Derived from ee_remote_sensing.py:
#   P0: Jan 1 – Mar 1  (DOY < 60)
#   P1: Mar 1 – May 1  (DOY < 121)
#   P2: May 1 – Jul 15 (DOY < 196)
#   P3: Jul 15 – Sep 30 (DOY < 273)
#   P4: Sep 30 – Dec 31 (DOY >= 273)
_LANDSAT_PERIOD_DOY_BOUNDS = [60, 121, 196, 273]


def _landsat_period(doy: int) -> int:
    for p, bound in enumerate(_LANDSAT_PERIOD_DOY_BOUNDS):
        if doy < bound:
            return p
    return 4


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


def _compute_raster_norm_stats(
    path: str, band_names: list[str]
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    with rasterio.open(path) as src:
        data = src.read().astype("float32")
    for i, name in enumerate(band_names):
        band = data[i].ravel()
        valid = band[np.isfinite(band)]
        if len(valid) == 0:
            continue
        stats[name] = {
            "mean": float(valid.mean()),
            "std": float(max(valid.std(), 1e-8)),
        }
    return stats


def _compute_rsun_norm_stats(rsun_path: str) -> dict[str, float]:
    """Sample 4 quarterly bands from the 365-band rsun TIF to get mean/std."""
    sample_doys = [1, 91, 182, 274]
    vals = []
    with rasterio.open(rsun_path) as src:
        for doy in sample_doys:
            band_idx = max(1, min(doy, src.count))
            data = src.read(band_idx).astype("float32").ravel()
            valid = data[np.isfinite(data) & (data > 0)]
            if len(valid):
                vals.append(valid)
    if not vals:
        return {"mean": 0.0, "std": 1.0}
    all_vals = np.concatenate(vals)
    return {"mean": float(all_vals.mean()), "std": float(max(all_vals.std(), 1e-8))}


def _compute_cdr_norm_stats(
    cdr_dir: str,
    cdr_pattern: str,
    samples: pd.DataFrame,
    band_names: list[str],
    n_sample: int = 20,
) -> dict[str, dict[str, float]]:
    """Sample a few daily CDR TIFs to compute per-band mean/std."""
    unique_days = samples["day"].drop_duplicates().sort_values()
    step = max(1, len(unique_days) // n_sample)
    sampled_days = unique_days.iloc[::step].head(n_sample)

    accum = {name: [] for name in band_names}
    for day in sampled_days:
        path = os.path.join(
            cdr_dir, cdr_pattern.format(date=pd.Timestamp(day).strftime("%Y%m%d"))
        )
        if not os.path.exists(path):
            continue
        with rasterio.open(path) as src:
            data = src.read().astype("float32")
        for i, name in enumerate(band_names):
            if i < data.shape[0]:
                valid = data[i].ravel()
                valid = valid[np.isfinite(valid)]
                if len(valid):
                    accum[name].append(valid)

    stats: dict[str, dict[str, float]] = {}
    for name, arrays in accum.items():
        if arrays:
            cat = np.concatenate(arrays)
            stats[name] = {
                "mean": float(cat.mean()),
                "std": float(max(cat.std(), 1e-8)),
            }
    return stats


class HRRRPatchDataset(Dataset):
    """
    Dataset for raster-patch HRRR bias learning.

    Each sample returns:
        x_patch     (C, patch_size, patch_size)  float32  — normalized raster input
        sta_rows    (N_sta,)                      int64    — row index within patch
        sta_cols    (N_sta,)                      int64    — col index within patch
        sta_targets (N_sta, n_targets)            float32  — delta values
        sta_valid   (N_sta, n_targets)            bool     — which pairs are supervised
    """

    def __init__(
        self,
        table_path: str,
        background_dir: str,
        background_pattern: str = "HRRR_1km_{date}.tif",
        static_tifs: list[str] | None = None,
        landsat_tif: str | None = None,
        target_names: list[str] | None = None,
        train_days: set | None = None,
        target_include_fids: set[str] | None = None,
        target_exclude_fids: set[str] | None = None,
        supervision_exclude_fids: set[str] | None = None,
        rsun_tif: str | None = None,
        cdr_dir: str | None = None,
        cdr_pattern: str = "CDR_005deg_{date}.tif",
        holdout_fids: set[str] | None = None,
        drop_bands: list[str] | None = None,
        norm_stats: dict | None = None,
        patch_size: int = 64,
        cache_size: int = 8,
    ):
        super().__init__()
        self._holdout_fids = holdout_fids or set()
        self._drop_bands = set(drop_bands or [])
        self._provided_norm_stats = norm_stats
        self.background_dir = background_dir
        self.background_pattern = background_pattern
        self.static_tifs = list(static_tifs) if static_tifs else []
        self.landsat_tif = landsat_tif
        self.rsun_tif = rsun_tif
        self.cdr_dir = cdr_dir
        self.cdr_pattern = cdr_pattern
        self.target_names = target_names or list(DEFAULT_TARGET_NAMES)
        self.patch_size = patch_size

        actual_cache = max(
            cache_size,
            len(self.static_tifs)
            + 2
            + (1 if landsat_tif else 0)
            + (1 if cdr_dir else 0),
        )
        self.raster_cache = _RasterCache(max_items=actual_cache)

        # --- Patch-center table ---
        df = pd.read_parquet(table_path)
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        df["fid"] = df["fid"].astype(str)
        df["day"] = pd.to_datetime(df["day"]).dt.normalize()

        day_set: set | None = None
        if train_days is not None:
            day_set = {pd.Timestamp(d).normalize() for d in train_days}
            df = df[df["day"].isin(day_set)]
        if target_include_fids is not None:
            df = df[df["fid"].isin(target_include_fids)]
        if target_exclude_fids is not None:
            df = df[~df["fid"].isin(target_exclude_fids)]

        sample_mask = df[self.target_names].notna().any(axis=1)
        df = df[sample_mask].copy()

        bg_exists = df["day"].map(lambda d: os.path.exists(self._background_path(d)))
        df = df[bg_exists].copy()
        if df.empty:
            raise ValueError("No supervised samples remain after raster/day filtering.")

        self.samples = df.sort_values("day").reset_index(drop=True)

        # --- Neighbor pool for in-patch supervision ---
        neighbor_df = pd.read_parquet(table_path)
        if isinstance(neighbor_df.index, pd.MultiIndex):
            neighbor_df = neighbor_df.reset_index()
        neighbor_df["fid"] = neighbor_df["fid"].astype(str)
        neighbor_df["day"] = pd.to_datetime(neighbor_df["day"]).dt.normalize()
        if day_set is not None:
            neighbor_df = neighbor_df[neighbor_df["day"].isin(day_set)]
        if supervision_exclude_fids is not None:
            neighbor_df = neighbor_df[
                ~neighbor_df["fid"].isin(supervision_exclude_fids)
            ]

        self._neighbor_day_groups = {
            pd.Timestamp(day).normalize(): grp.reset_index(drop=True)
            for day, grp in neighbor_df.groupby("day")
        }

        # --- Band names ---
        example_bg = self._background_path(self.samples.iloc[0]["day"])
        all_bg_names = _discover_band_names(example_bg, "bg")
        # Filter out drop_bands
        if self._drop_bands:
            self._bg_keep_indices = [
                i for i, n in enumerate(all_bg_names) if n not in self._drop_bands
            ]
            self.background_feature_names: list[str] = [
                all_bg_names[i] for i in self._bg_keep_indices
            ]
        else:
            self._bg_keep_indices = list(range(len(all_bg_names)))
            self.background_feature_names: list[str] = all_bg_names

        self._static_tif_band_names: list[list[str]] = []
        self.static_feature_names: list[str] = []
        for tif_path in self.static_tifs:
            prefix = os.path.splitext(os.path.basename(tif_path))[0]
            names = _discover_band_names(tif_path, prefix)
            self._static_tif_band_names.append(names)
            self.static_feature_names.extend(names)

        self._all_landsat_band_names: list[str] = []
        self._landsat_period_names: list[list[str]] = [
            [] for _ in range(_LANDSAT_PERIODS)
        ]
        if self.landsat_tif:
            all_ls = _discover_band_names(self.landsat_tif, "landsat")
            self._all_landsat_band_names = all_ls
            for p in range(_LANDSAT_PERIODS):
                self._landsat_period_names[p] = all_ls[
                    p * _LANDSAT_BANDS : (p + 1) * _LANDSAT_BANDS
                ]

        # P0 names as the representative slot; all periods have the same length (7)
        _rep_landsat = self._landsat_period_names[0] if self.landsat_tif else []

        # --- rsun (1 band, DOY-selected from 365-band TIF) ---
        self._rsun_meta: dict | None = None
        if self.rsun_tif:
            with rasterio.open(self.rsun_tif) as src:
                self._rsun_meta = {
                    "transform": src.transform,
                    "crs": str(src.crs),
                    "count": src.count,
                    "height": src.height,
                    "width": src.width,
                }

        # --- CDR (daily 5-band TIFs at native 0.05-deg) ---
        self._cdr_band_names: list[str] = []
        if self.cdr_dir:
            example_cdr = self._cdr_path(self.samples.iloc[0]["day"])
            if os.path.exists(example_cdr):
                self._cdr_band_names = _discover_band_names(example_cdr, "cdr")

        # Full channel list
        self.feature_names: list[str] = (
            list(self.background_feature_names)
            + list(self.static_feature_names)
            + list(_rep_landsat)
            + (["rsun"] if self.rsun_tif else [])
            + list(self._cdr_band_names)
            + ["doy_sin", "doy_cos", "latitude", "longitude"]
        )
        self.in_channels = len(self.feature_names)

        # --- Norm stats (use provided if available, else compute from data) ---
        if self._provided_norm_stats is not None:
            self.norm_stats = dict(self._provided_norm_stats)
        else:
            parquet_cols = [
                c
                for c in self.background_feature_names
                + self.static_feature_names
                + ["latitude", "longitude"]
                if c in self.samples.columns
            ]
            self.norm_stats = _compute_norm_stats(self.samples, parquet_cols)

        # Compute raster/Landsat/rsun/CDR stats only if not provided
        if self._provided_norm_stats is None:
            for tif_path, tif_names in zip(
                self.static_tifs, self._static_tif_band_names
            ):
                uncovered = [n for n in tif_names if n not in self.norm_stats]
                if uncovered:
                    raster_stats = _compute_raster_norm_stats(tif_path, tif_names)
                    self.norm_stats.update(
                        {n: raster_stats[n] for n in uncovered if n in raster_stats}
                    )

            if self.landsat_tif:
                ls_stats = _compute_raster_norm_stats(
                    self.landsat_tif, self._all_landsat_band_names
                )
                self.norm_stats.update(ls_stats)

            if self.rsun_tif:
                self.norm_stats["rsun"] = _compute_rsun_norm_stats(self.rsun_tif)

        if self._provided_norm_stats is None and self.cdr_dir and self._cdr_band_names:
            cdr_stats = _compute_cdr_norm_stats(
                self.cdr_dir, self.cdr_pattern, self.samples, self._cdr_band_names
            )
            self.norm_stats.update(cdr_stats)

    def __len__(self) -> int:
        return len(self.samples)

    def _background_path(self, day: pd.Timestamp) -> str:
        return os.path.join(
            self.background_dir,
            self.background_pattern.format(date=pd.Timestamp(day).strftime("%Y%m%d")),
        )

    def _cdr_path(self, day: pd.Timestamp) -> str:
        return os.path.join(
            self.cdr_dir,
            self.cdr_pattern.format(date=pd.Timestamp(day).strftime("%Y%m%d")),
        )

    def __getitem__(self, idx: int):
        row = self.samples.iloc[idx]
        day = pd.Timestamp(row["day"]).normalize()
        doy = day.dayofyear
        period = _landsat_period(doy)

        # --- HRRR background patch ---
        bg = self.raster_cache.get(self._background_path(day))
        bg_data: np.ndarray = bg["data"]  # (C_bg, domain_H, domain_W)
        bg_tf = bg["transform"]
        bg_crs = bg["crs"]

        domain_H, domain_W = bg_data.shape[1], bg_data.shape[2]
        H = W = self.patch_size

        to_bg = Transformer.from_crs("EPSG:4326", bg_crs, always_xy=True)
        x_center, y_center = to_bg.transform(
            float(row["longitude"]), float(row["latitude"])
        )
        r_center, c_center = rowcol(bg_tf, x_center, y_center)

        # Clamp so the patch always fits within the domain
        r0 = int(np.clip(int(r_center) - H // 2, 0, domain_H - H))
        c0 = int(np.clip(int(c_center) - W // 2, 0, domain_W - W))
        r1 = r0 + H
        c1 = c0 + W

        bg_patch = bg_data[self._bg_keep_indices, r0:r1, c0:c1].astype(
            "float32"
        )  # (C_bg_filtered, H, W)

        # Projected pixel-center coordinates for static/Landsat lookup
        rows = np.arange(r0, r1)
        cols = np.arange(c0, c1)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        rr_flat = rr.ravel()
        cc_flat = cc.ravel()
        x_proj, y_proj = xy(bg_tf, rr_flat, cc_flat, offset="center")
        x_proj = np.asarray(x_proj, dtype="float64")
        y_proj = np.asarray(y_proj, dtype="float64")

        to_ll = Transformer.from_crs(bg_crs, "EPSG:4326", always_xy=True)
        pixel_lon, pixel_lat = to_ll.transform(x_proj, y_proj)
        pixel_lon = np.asarray(pixel_lon, dtype="float32").reshape(H, W)
        pixel_lat = np.asarray(pixel_lat, dtype="float32").reshape(H, W)

        # --- Static TIF patches ---
        static_patches: list[np.ndarray] = []
        for tif_path, tif_names in zip(self.static_tifs, self._static_tif_band_names):
            static = self.raster_cache.get(tif_path)
            s_data = static["data"]
            s_tf = static["transform"]
            s_crs = static["crs"]
            if str(s_crs) == str(bg_crs):
                s_rows = np.clip(rr_flat, 0, s_data.shape[1] - 1).astype(int)
                s_cols = np.clip(cc_flat, 0, s_data.shape[2] - 1).astype(int)
            else:
                to_s = Transformer.from_crs(bg_crs, s_crs, always_xy=True)
                sx, sy = to_s.transform(x_proj, y_proj)
                s_rows_raw, s_cols_raw = rowcol(s_tf, sx, sy)
                s_rows = np.clip(
                    np.asarray(s_rows_raw, dtype=int), 0, s_data.shape[1] - 1
                )
                s_cols = np.clip(
                    np.asarray(s_cols_raw, dtype=int), 0, s_data.shape[2] - 1
                )
            patch = np.nan_to_num(
                s_data[:, s_rows, s_cols].astype("float32"), nan=0.0
            ).reshape(len(tif_names), H, W)
            static_patches.append(patch)

        # --- Landsat period patch (7 bands) ---
        landsat_patch: np.ndarray | None = None
        active_landsat_names: list[str] = []
        if self.landsat_tif:
            active_landsat_names = self._landsat_period_names[period]
            landsat = self.raster_cache.get(self.landsat_tif)
            ls_data = landsat["data"]  # (35, domain_H, domain_W)
            ls_tf = landsat["transform"]
            ls_crs = landsat["crs"]
            b0 = period * _LANDSAT_BANDS
            b1 = b0 + _LANDSAT_BANDS
            if str(ls_crs) == str(bg_crs):
                ls_rows = np.clip(rr_flat, 0, ls_data.shape[1] - 1).astype(int)
                ls_cols = np.clip(cc_flat, 0, ls_data.shape[2] - 1).astype(int)
            else:
                to_ls = Transformer.from_crs(bg_crs, ls_crs, always_xy=True)
                lsx, lsy = to_ls.transform(x_proj, y_proj)
                ls_rows_raw, ls_cols_raw = rowcol(ls_tf, lsx, lsy)
                ls_rows = np.clip(
                    np.asarray(ls_rows_raw, dtype=int), 0, ls_data.shape[1] - 1
                )
                ls_cols = np.clip(
                    np.asarray(ls_cols_raw, dtype=int), 0, ls_data.shape[2] - 1
                )
            landsat_patch = np.nan_to_num(
                ls_data[b0:b1, ls_rows, ls_cols].astype("float32"), nan=0.0
            ).reshape(_LANDSAT_BANDS, H, W)

        # --- rsun patch (1 band, DOY-selected) ---
        rsun_patch: np.ndarray | None = None
        if self.rsun_tif and self._rsun_meta is not None:
            # Cache the band within a day group (DayGroupedSampler reuses DOY)
            cached_doy = getattr(self, "_rsun_cache_doy", -1)
            if cached_doy != doy:
                with rasterio.open(self.rsun_tif) as rsun_src:
                    band_idx = max(1, min(doy, rsun_src.count))
                    self._rsun_cache_band = rsun_src.read(band_idx).astype("float32")
                    self._rsun_cache_doy = doy
            rsun_band = self._rsun_cache_band  # (H_dom, W_dom)
            rsun_crs = self._rsun_meta["crs"]
            rsun_tf = self._rsun_meta["transform"]
            if rsun_crs == str(bg_crs):
                rs_rows = np.clip(rr_flat, 0, rsun_band.shape[0] - 1).astype(int)
                rs_cols = np.clip(cc_flat, 0, rsun_band.shape[1] - 1).astype(int)
            else:
                to_rsun = Transformer.from_crs(bg_crs, rsun_crs, always_xy=True)
                rx, ry = to_rsun.transform(x_proj, y_proj)
                rs_rows_raw, rs_cols_raw = rowcol(rsun_tf, rx, ry)
                rs_rows = np.clip(
                    np.asarray(rs_rows_raw, dtype=int), 0, rsun_band.shape[0] - 1
                )
                rs_cols = np.clip(
                    np.asarray(rs_cols_raw, dtype=int), 0, rsun_band.shape[1] - 1
                )
            rsun_patch = np.nan_to_num(rsun_band[rs_rows, rs_cols], nan=0.0).reshape(
                1, H, W
            )

        # --- CDR patch (5 bands, daily, native 0.05-deg) ---
        cdr_patch: np.ndarray | None = None
        if self.cdr_dir and self._cdr_band_names:
            cdr_path = self._cdr_path(day)
            n_cdr = len(self._cdr_band_names)
            if os.path.exists(cdr_path):
                cdr = self.raster_cache.get(cdr_path)
                cdr_data = cdr["data"]  # (5, cdr_H, cdr_W)
                cdr_tf = cdr["transform"]
                # CDR is EPSG:4326 — use already-computed pixel lat/lon
                cdr_rows_raw, cdr_cols_raw = rowcol(
                    cdr_tf,
                    pixel_lon.ravel().astype("float64"),
                    pixel_lat.ravel().astype("float64"),
                )
                cdr_rows = np.clip(
                    np.asarray(cdr_rows_raw, dtype=int), 0, cdr_data.shape[1] - 1
                )
                cdr_cols = np.clip(
                    np.asarray(cdr_cols_raw, dtype=int), 0, cdr_data.shape[2] - 1
                )
                cdr_patch = np.nan_to_num(
                    cdr_data[:, cdr_rows, cdr_cols].astype("float32"), nan=0.0
                ).reshape(n_cdr, H, W)
            else:
                cdr_patch = np.zeros((n_cdr, H, W), dtype="float32")

        # --- Position / time channels ---
        doy_sin, doy_cos = _doy_features(day)
        pos_time = np.stack(
            [
                np.full((H, W), doy_sin, dtype="float32"),
                np.full((H, W), doy_cos, dtype="float32"),
                pixel_lat,
                pixel_lon,
            ]
        )  # (4, H, W)

        # --- Stack and normalize ---
        parts: list[np.ndarray] = [bg_patch] + static_patches
        if landsat_patch is not None:
            parts.append(landsat_patch)
        if rsun_patch is not None:
            parts.append(rsun_patch)
        if cdr_patch is not None:
            parts.append(cdr_patch)
        parts.append(pos_time)
        x_patch = np.concatenate(parts, axis=0)  # (C, H, W)

        channel_names = (
            list(self.background_feature_names)
            + list(self.static_feature_names)
            + active_landsat_names
            + (["rsun"] if self.rsun_tif else [])
            + list(self._cdr_band_names)
            + ["doy_sin", "doy_cos", "latitude", "longitude"]
        )
        for i, name in enumerate(channel_names):
            if name in self.norm_stats:
                s = self.norm_stats[name]
                x_patch[i] = (x_patch[i] - s["mean"]) / s["std"]

        # --- In-patch station supervision targets ---
        sta_rows_list: list[int] = []
        sta_cols_list: list[int] = []
        sta_targets_list: list[list[float]] = []
        sta_valid_list: list[list[bool]] = []
        sta_holdout_list: list[bool] = []
        sta_is_center_list: list[bool] = []
        center_fid = str(row["fid"])

        day_group = self._neighbor_day_groups.get(day)
        if day_group is not None and not day_group.empty:
            sta_lons = day_group["longitude"].to_numpy(dtype="float64")
            sta_lats = day_group["latitude"].to_numpy(dtype="float64")
            sta_x, sta_y = to_bg.transform(sta_lons, sta_lats)
            sta_r_raw, sta_c_raw = rowcol(bg_tf, sta_x, sta_y)
            sta_r = np.asarray(sta_r_raw, dtype=int)
            sta_c = np.asarray(sta_c_raw, dtype=int)

            in_patch = (sta_r >= r0) & (sta_r < r1) & (sta_c >= c0) & (sta_c < c1)
            for j in np.where(in_patch)[0]:
                sta_rows_list.append(int(sta_r[j] - r0))
                sta_cols_list.append(int(sta_c[j] - c0))
                fid_j = str(day_group.iloc[j]["fid"])
                sta_holdout_list.append(fid_j in self._holdout_fids)
                sta_is_center_list.append(fid_j == center_fid)
                tgt = []
                vld = []
                for name in self.target_names:
                    val = day_group.iloc[j].get(name, float("nan"))
                    is_valid = pd.notna(val)
                    tgt.append(float(val) if is_valid else 0.0)
                    vld.append(bool(is_valid))
                sta_targets_list.append(tgt)
                sta_valid_list.append(vld)

        # Fallback: use the patch-center station if no in-patch stations were found
        if not sta_rows_list:
            r_in = int(np.clip(int(r_center) - r0, 0, H - 1))
            c_in = int(np.clip(int(c_center) - c0, 0, W - 1))
            sta_rows_list.append(r_in)
            sta_cols_list.append(c_in)
            sta_holdout_list.append(center_fid in self._holdout_fids)
            sta_is_center_list.append(True)
            tgt = []
            vld = []
            for name in self.target_names:
                val = row.get(name, float("nan"))
                is_valid = pd.notna(val)
                tgt.append(float(val) if is_valid else 0.0)
                vld.append(bool(is_valid))
            sta_targets_list.append(tgt)
            sta_valid_list.append(vld)

        return (
            torch.from_numpy(x_patch),
            torch.tensor(sta_rows_list, dtype=torch.long),
            torch.tensor(sta_cols_list, dtype=torch.long),
            torch.tensor(sta_targets_list, dtype=torch.float32),
            torch.tensor(sta_valid_list, dtype=torch.bool),
            torch.tensor(sta_holdout_list, dtype=torch.bool),
            torch.tensor(sta_is_center_list, dtype=torch.bool),
        )


def collate_patch(batch):
    """Collate a list of HRRRPatchDataset samples, padding station arrays to max N_sta."""
    # Support 5-tuple (legacy), 6-tuple, or 7-tuple (with holdout + is_center)
    n_fields = len(batch[0])
    if n_fields == 7:
        (
            x_list,
            rows_list,
            cols_list,
            tgts_list,
            valid_list,
            holdout_list,
            center_list,
        ) = zip(*batch)
    elif n_fields == 6:
        x_list, rows_list, cols_list, tgts_list, valid_list, holdout_list = zip(*batch)
        center_list = None
    else:
        x_list, rows_list, cols_list, tgts_list, valid_list = zip(*batch)
        holdout_list = None
        center_list = None

    x = torch.stack(x_list)  # (B, C, H, W)
    n_targets = tgts_list[0].shape[1]
    max_sta = max(r.shape[0] for r in rows_list)
    B = len(batch)

    sta_rows = torch.zeros(B, max_sta, dtype=torch.long)
    sta_cols = torch.zeros(B, max_sta, dtype=torch.long)
    sta_targets = torch.zeros(B, max_sta, n_targets, dtype=torch.float32)
    sta_valid = torch.zeros(B, max_sta, n_targets, dtype=torch.bool)

    sta_holdout = torch.zeros(B, max_sta, dtype=torch.bool)
    sta_is_center = torch.zeros(B, max_sta, dtype=torch.bool)

    for i, (rows, cols, tgts, vld) in enumerate(
        zip(rows_list, cols_list, tgts_list, valid_list)
    ):
        n = rows.shape[0]
        sta_rows[i, :n] = rows
        sta_cols[i, :n] = cols
        sta_targets[i, :n] = tgts
        sta_valid[i, :n] = vld
        if holdout_list is not None:
            sta_holdout[i, :n] = holdout_list[i]
        if center_list is not None:
            sta_is_center[i, :n] = center_list[i]

    return x, sta_rows, sta_cols, sta_targets, sta_valid, sta_holdout, sta_is_center
