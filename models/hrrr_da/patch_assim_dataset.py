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
        patch_size: int = 64,
        cache_size: int = 8,
    ):
        super().__init__()
        self.background_dir = background_dir
        self.background_pattern = background_pattern
        self.static_tifs = list(static_tifs) if static_tifs else []
        self.landsat_tif = landsat_tif
        self.target_names = target_names or list(DEFAULT_TARGET_NAMES)
        self.patch_size = patch_size

        actual_cache = max(
            cache_size, len(self.static_tifs) + 2 + (1 if landsat_tif else 0)
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
        self.background_feature_names: list[str] = _discover_band_names(
            example_bg, "bg"
        )

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

        # Full channel list: HRRR + static + 7 Landsat (period-selected) + pos/time
        self.feature_names: list[str] = (
            list(self.background_feature_names)
            + list(self.static_feature_names)
            + list(_rep_landsat)
            + ["doy_sin", "doy_cos", "latitude", "longitude"]
        )
        self.in_channels = len(self.feature_names)

        # --- Norm stats ---
        parquet_cols = [
            c
            for c in self.background_feature_names
            + self.static_feature_names
            + ["latitude", "longitude"]
            if c in self.samples.columns
        ]
        self.norm_stats = _compute_norm_stats(self.samples, parquet_cols)

        # Raster-based stats for static TIF bands absent from the parquet
        for tif_path, tif_names in zip(self.static_tifs, self._static_tif_band_names):
            uncovered = [n for n in tif_names if n not in self.norm_stats]
            if uncovered:
                raster_stats = _compute_raster_norm_stats(tif_path, tif_names)
                self.norm_stats.update(
                    {n: raster_stats[n] for n in uncovered if n in raster_stats}
                )

        # All 35 Landsat bands get stats from the raster
        if self.landsat_tif:
            ls_stats = _compute_raster_norm_stats(
                self.landsat_tif, self._all_landsat_band_names
            )
            self.norm_stats.update(ls_stats)

    def __len__(self) -> int:
        return len(self.samples)

    def _background_path(self, day: pd.Timestamp) -> str:
        return os.path.join(
            self.background_dir,
            self.background_pattern.format(date=pd.Timestamp(day).strftime("%Y%m%d")),
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

        bg_patch = bg_data[:, r0:r1, c0:c1].astype("float32")  # (C_bg, H, W)

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
        parts.append(pos_time)
        x_patch = np.concatenate(parts, axis=0)  # (C, H, W)

        channel_names = (
            list(self.background_feature_names)
            + list(self.static_feature_names)
            + active_landsat_names
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
        )


def collate_patch(batch):
    """Collate a list of HRRRPatchDataset samples, padding station arrays to max N_sta."""
    x_list, rows_list, cols_list, tgts_list, valid_list = zip(*batch)

    x = torch.stack(x_list)  # (B, C, H, W)
    n_targets = tgts_list[0].shape[1]
    max_sta = max(r.shape[0] for r in rows_list)
    B = len(batch)

    sta_rows = torch.zeros(B, max_sta, dtype=torch.long)
    sta_cols = torch.zeros(B, max_sta, dtype=torch.long)
    sta_targets = torch.zeros(B, max_sta, n_targets, dtype=torch.float32)
    sta_valid = torch.zeros(B, max_sta, n_targets, dtype=torch.bool)

    for i, (rows, cols, tgts, vld) in enumerate(
        zip(rows_list, cols_list, tgts_list, valid_list)
    ):
        n = rows.shape[0]
        sta_rows[i, :n] = rows
        sta_cols[i, :n] = cols
        sta_targets[i, :n] = tgts
        sta_valid[i, :n] = vld

    return x, sta_rows, sta_cols, sta_targets, sta_valid
