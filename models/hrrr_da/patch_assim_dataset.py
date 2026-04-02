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

import hashlib
import json
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
from prep.paths import NVME_CACHE

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


def _background_path_for(
    background_dir: str, background_pattern: str, day: pd.Timestamp
) -> str:
    return os.path.join(
        background_dir,
        background_pattern.format(date=pd.Timestamp(day).strftime("%Y%m%d")),
    )


def _normalise_day_tokens(days: set | None) -> list[str]:
    if days is None:
        return []
    return [
        pd.Timestamp(d).normalize().strftime("%Y-%m-%d")
        for d in sorted(pd.Timestamp(d).normalize() for d in days)
    ]


def _normalise_str_tokens(values: set[str] | None) -> list[str]:
    if values is None:
        return []
    return sorted(str(v) for v in values)


def _patch_metadata_root() -> str:
    if os.path.isdir(os.path.dirname(NVME_CACHE)):
        return os.path.join(NVME_CACHE, "patch_assim_metadata")
    return os.path.join("/tmp", "dads_patch_assim_metadata")


def _patch_metadata_dir(
    table_path: str,
    background_dir: str,
    background_pattern: str,
    target_names: list[str],
    train_days: set | None,
    target_include_fids: set[str] | None,
    target_exclude_fids: set[str] | None,
    supervision_exclude_fids: set[str] | None,
    centers_per_day: int | None,
    center_sampling_seed: int,
) -> str:
    stat = os.stat(table_path)
    payload = {
        "version": 1,
        "table_path": os.path.realpath(table_path),
        "table_size": stat.st_size,
        "table_mtime_ns": stat.st_mtime_ns,
        "background_dir": os.path.realpath(background_dir),
        "background_pattern": background_pattern,
        "target_names": list(target_names),
        "train_days": _normalise_day_tokens(train_days),
        "target_include_fids": _normalise_str_tokens(target_include_fids),
        "target_exclude_fids": _normalise_str_tokens(target_exclude_fids),
        "supervision_exclude_fids": _normalise_str_tokens(supervision_exclude_fids),
        "centers_per_day": centers_per_day,
        "center_sampling_seed": center_sampling_seed,
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return os.path.join(_patch_metadata_root(), digest)


def _project_rows_cols(
    df: pd.DataFrame, bg_crs, bg_tf
) -> tuple[np.ndarray, np.ndarray]:
    lons = df["longitude"].to_numpy(dtype="float64")
    lats = df["latitude"].to_numpy(dtype="float64")
    to_bg = Transformer.from_crs("EPSG:4326", bg_crs, always_xy=True)
    x, y = to_bg.transform(lons, lats)
    rows, cols = rowcol(bg_tf, x, y)
    return np.asarray(rows, dtype="int32"), np.asarray(cols, dtype="int32")


def _build_day_slices(days: pd.Series) -> dict[pd.Timestamp, tuple[int, int]]:
    if days.empty:
        return {}
    day_values = pd.to_datetime(days).dt.normalize().to_numpy()
    unique_days, starts = np.unique(day_values, return_index=True)
    stops = np.append(starts[1:], len(day_values))
    return {
        pd.Timestamp(day).normalize(): (int(start), int(stop))
        for day, start, stop in zip(unique_days, starts, stops, strict=False)
    }


def _write_day_slices(
    path: str, day_slices: dict[pd.Timestamp, tuple[int, int]]
) -> None:
    payload = {
        pd.Timestamp(day).strftime("%Y-%m-%d"): [int(start), int(stop)]
        for day, (start, stop) in day_slices.items()
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _read_day_slices(path: str) -> dict[pd.Timestamp, tuple[int, int]]:
    with open(path) as f:
        payload = json.load(f)
    return {
        pd.Timestamp(day).normalize(): (int(bounds[0]), int(bounds[1]))
        for day, bounds in payload.items()
    }


def _load_or_build_patch_metadata(
    table_path: str,
    background_dir: str,
    background_pattern: str,
    target_names: list[str],
    train_days: set | None,
    target_include_fids: set[str] | None,
    target_exclude_fids: set[str] | None,
    supervision_exclude_fids: set[str] | None,
    centers_per_day: int | None,
    center_sampling_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[pd.Timestamp, tuple[int, int]], str]:
    cache_dir = _patch_metadata_dir(
        table_path=table_path,
        background_dir=background_dir,
        background_pattern=background_pattern,
        target_names=target_names,
        train_days=train_days,
        target_include_fids=target_include_fids,
        target_exclude_fids=target_exclude_fids,
        supervision_exclude_fids=supervision_exclude_fids,
        centers_per_day=centers_per_day,
        center_sampling_seed=center_sampling_seed,
    )
    samples_path = os.path.join(cache_dir, "samples.parquet")
    neighbors_path = os.path.join(cache_dir, "neighbors.parquet")
    day_slices_path = os.path.join(cache_dir, "neighbor_day_slices.json")

    if (
        os.path.exists(samples_path)
        and os.path.exists(neighbors_path)
        and os.path.exists(day_slices_path)
    ):
        print(f"[patch_meta] cache hit: {cache_dir}")
        samples = pd.read_parquet(samples_path)
        neighbors = pd.read_parquet(neighbors_path)
        samples["day"] = pd.to_datetime(samples["day"]).dt.normalize()
        neighbors["day"] = pd.to_datetime(neighbors["day"]).dt.normalize()
        return samples, neighbors, _read_day_slices(day_slices_path), cache_dir

    print(f"[patch_meta] building cache: {cache_dir}")
    df = pd.read_parquet(table_path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["fid"] = df["fid"].astype(str)
    df["day"] = pd.to_datetime(df["day"]).dt.normalize()

    day_set: set | None = None
    if train_days is not None:
        day_set = {pd.Timestamp(d).normalize() for d in train_days}
        df = df[df["day"].isin(day_set)]

    samples = df
    if target_include_fids is not None:
        samples = samples[samples["fid"].isin(target_include_fids)]
    if target_exclude_fids is not None:
        samples = samples[~samples["fid"].isin(target_exclude_fids)]

    sample_mask = samples[target_names].notna().any(axis=1)
    samples = samples[sample_mask].copy()

    bg_exists = samples["day"].map(
        lambda d: os.path.exists(
            _background_path_for(background_dir, background_pattern, d)
        )
    )
    samples = samples[bg_exists].copy()
    if samples.empty:
        raise ValueError("No supervised samples remain after raster/day filtering.")

    samples = samples.sort_values(["day", "fid"]).reset_index(drop=True)
    if centers_per_day is not None and centers_per_day > 0:
        rng = np.random.default_rng(center_sampling_seed)
        keep_idx: list[int] = []
        for _, grp in samples.groupby("day", sort=False):
            idxs = grp.index.to_numpy()
            if len(idxs) > centers_per_day:
                idxs = np.sort(rng.choice(idxs, centers_per_day, replace=False))
            keep_idx.extend(idxs.tolist())
        samples = samples.loc[keep_idx].reset_index(drop=True)

    valid_days = set(samples["day"].unique())
    neighbors = df[df["day"].isin(valid_days)].copy()
    if supervision_exclude_fids is not None:
        neighbors = neighbors[~neighbors["fid"].isin(supervision_exclude_fids)]
    neighbors = neighbors.sort_values(["day", "fid"]).reset_index(drop=True)

    example_bg = _background_path_for(
        background_dir, background_pattern, samples.iloc[0]["day"]
    )
    with rasterio.open(example_bg) as src:
        bg_crs = src.crs
        bg_tf = src.transform

    sample_rows, sample_cols = _project_rows_cols(samples, bg_crs, bg_tf)
    neighbor_rows, neighbor_cols = _project_rows_cols(neighbors, bg_crs, bg_tf)
    samples["bg_row"] = sample_rows
    samples["bg_col"] = sample_cols
    neighbors["bg_row"] = neighbor_rows
    neighbors["bg_col"] = neighbor_cols

    day_slices = _build_day_slices(neighbors["day"])

    os.makedirs(cache_dir, exist_ok=True)
    samples.to_parquet(samples_path, index=False)
    neighbors.to_parquet(neighbors_path, index=False)
    _write_day_slices(day_slices_path, day_slices)
    return samples, neighbors, day_slices, cache_dir


def _tile_index_cache_path(
    cache_dir: str,
    patch_size: int,
    tile_stride: int,
    tiles_per_day: int | None,
    tile_sampling_seed: int,
) -> str:
    tpd = "all" if tiles_per_day is None else str(int(tiles_per_day))
    filename = (
        f"tile_index_ps{int(patch_size)}"
        f"_st{int(tile_stride)}"
        f"_tpd{tpd}"
        f"_seed{int(tile_sampling_seed)}.parquet"
    )
    return os.path.join(cache_dir, filename)


def _load_or_build_tile_index(
    station_samples: pd.DataFrame,
    cache_dir: str,
    domain_h: int,
    domain_w: int,
    patch_size: int,
    tile_stride: int,
    tiles_per_day: int | None,
    tile_sampling_seed: int,
) -> pd.DataFrame:
    cache_path = _tile_index_cache_path(
        cache_dir=cache_dir,
        patch_size=patch_size,
        tile_stride=tile_stride,
        tiles_per_day=tiles_per_day,
        tile_sampling_seed=tile_sampling_seed,
    )
    if os.path.exists(cache_path):
        tile_df = pd.read_parquet(cache_path)
        tile_df["day"] = pd.to_datetime(tile_df["day"]).dt.normalize()
        return tile_df

    max_r0 = max(domain_h - patch_size, 0)
    max_c0 = max(domain_w - patch_size, 0)
    tile_rows = np.clip(
        (station_samples["bg_row"].to_numpy(dtype="int32") // tile_stride)
        * tile_stride,
        0,
        max_r0,
    )
    tile_cols = np.clip(
        (station_samples["bg_col"].to_numpy(dtype="int32") // tile_stride)
        * tile_stride,
        0,
        max_c0,
    )

    tile_df = pd.DataFrame(
        {
            "day": pd.to_datetime(station_samples["day"]).dt.normalize(),
            "tile_r0": tile_rows.astype("int32"),
            "tile_c0": tile_cols.astype("int32"),
        }
    )
    tile_df = (
        tile_df.groupby(["day", "tile_r0", "tile_c0"], as_index=False)
        .size()
        .rename(columns={"size": "n_station_samples"})
        .sort_values(["day", "tile_r0", "tile_c0"])
        .reset_index(drop=True)
    )

    if tiles_per_day is not None and tiles_per_day > 0:
        rng = np.random.default_rng(tile_sampling_seed)
        keep_idx: list[int] = []
        for _, grp in tile_df.groupby("day", sort=False):
            idxs = grp.index.to_numpy()
            if len(idxs) > tiles_per_day:
                idxs = np.sort(rng.choice(idxs, tiles_per_day, replace=False))
            keep_idx.extend(idxs.tolist())
        tile_df = tile_df.loc[keep_idx].reset_index(drop=True)

    tile_df.to_parquet(cache_path, index=False)
    return tile_df


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
        centers_per_day: int | None = None,
        center_sampling_seed: int = 42,
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
        self._centers_per_day = centers_per_day
        self._center_sampling_seed = center_sampling_seed

        actual_cache = max(
            cache_size,
            len(self.static_tifs)
            + 2
            + (1 if landsat_tif else 0)
            + (1 if cdr_dir else 0),
        )
        self.raster_cache = _RasterCache(max_items=actual_cache)

        (
            self.samples,
            self._neighbor_df,
            self._neighbor_day_slices,
            self.metadata_cache_dir,
        ) = _load_or_build_patch_metadata(
            table_path=table_path,
            background_dir=self.background_dir,
            background_pattern=self.background_pattern,
            target_names=self.target_names,
            train_days=train_days,
            target_include_fids=target_include_fids,
            target_exclude_fids=target_exclude_fids,
            supervision_exclude_fids=supervision_exclude_fids,
            centers_per_day=self._centers_per_day,
            center_sampling_seed=self._center_sampling_seed,
        )

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

        # --- Pre-compute domain geometry (eliminates per-sample CRS work) ---
        example_bg = self.raster_cache.get(
            self._background_path(self.samples.iloc[0]["day"])
        )
        bg_crs = example_bg["crs"]
        bg_tf = example_bg["transform"]
        self._domain_H = example_bg["data"].shape[1]
        self._domain_W = example_bg["data"].shape[2]

        # Assert pixel-alignment for all co-registered rasters
        for tif_path in self.static_tifs:
            s = self.raster_cache.get(tif_path)
            assert str(s["crs"]) == str(bg_crs), (
                f"Static TIF CRS mismatch: {tif_path} has {s['crs']}, expected {bg_crs}"
            )
            assert s["transform"] == bg_tf, f"Static TIF transform mismatch: {tif_path}"
            assert s["data"].shape[1:] == (self._domain_H, self._domain_W), (
                f"Static TIF shape mismatch: {tif_path} has "
                f"{s['data'].shape[1:]}, expected {(self._domain_H, self._domain_W)}"
            )
        if self.landsat_tif:
            ls = self.raster_cache.get(self.landsat_tif)
            assert str(ls["crs"]) == str(bg_crs), (
                f"Landsat CRS mismatch: {ls['crs']} vs {bg_crs}"
            )
            assert ls["transform"] == bg_tf, "Landsat transform mismatch"
            assert ls["data"].shape[1:] == (self._domain_H, self._domain_W), (
                f"Landsat shape mismatch: {ls['data'].shape[1:]}"
            )
        if self.rsun_tif and self._rsun_meta is not None:
            assert self._rsun_meta["crs"] == str(bg_crs), (
                f"rsun CRS mismatch: {self._rsun_meta['crs']} vs {bg_crs}"
            )
            assert self._rsun_meta["transform"] == bg_tf, (
                "rsun transform mismatch vs HRRR background"
            )
            assert (self._rsun_meta["height"], self._rsun_meta["width"]) == (
                self._domain_H,
                self._domain_W,
            ), (
                f"rsun shape mismatch: {self._rsun_meta['height']}x{self._rsun_meta['width']}"
            )

        # CDR alignment: verify geometry consistency across sampled days
        if self.cdr_dir and self._cdr_band_names:
            sample_days = self.samples["day"].drop_duplicates().sort_values()
            step = max(1, len(sample_days) // 10)
            cdr_ref_crs = cdr_ref_tf = cdr_ref_shape = None
            for check_day in sample_days.iloc[::step].head(10):
                cp = self._cdr_path(check_day)
                if not os.path.exists(cp):
                    continue
                with rasterio.open(cp) as src:
                    if cdr_ref_crs is None:
                        cdr_ref_crs = str(src.crs)
                        cdr_ref_tf = src.transform
                        cdr_ref_shape = (src.height, src.width)
                    else:
                        assert str(src.crs) == cdr_ref_crs, (
                            f"CDR CRS varies across days: {cp}"
                        )
                        assert src.transform == cdr_ref_tf, (
                            f"CDR transform varies across days: {cp}"
                        )
                        assert (src.height, src.width) == cdr_ref_shape, (
                            f"CDR shape varies across days: {cp}"
                        )

        # Domain-wide lat/lon grids (for CDR lookup + position channels)
        to_ll = Transformer.from_crs(bg_crs, "EPSG:4326", always_xy=True)
        rr_all, cc_all = np.meshgrid(
            np.arange(self._domain_H), np.arange(self._domain_W), indexing="ij"
        )
        x_all, y_all = xy(bg_tf, rr_all.ravel(), cc_all.ravel(), offset="center")
        lon_all, lat_all = to_ll.transform(
            np.asarray(x_all, dtype="float64"), np.asarray(y_all, dtype="float64")
        )
        self._domain_lat = np.asarray(lat_all, dtype="float32").reshape(
            self._domain_H, self._domain_W
        )
        self._domain_lon = np.asarray(lon_all, dtype="float32").reshape(
            self._domain_H, self._domain_W
        )

        # CDR index maps (EPSG:5070 pixel → CDR row/col)
        self._cdr_row_map: np.ndarray | None = None
        self._cdr_col_map: np.ndarray | None = None
        if self.cdr_dir and self._cdr_band_names:
            example_cdr_path = self._cdr_path(self.samples.iloc[0]["day"])
            if os.path.exists(example_cdr_path):
                with rasterio.open(example_cdr_path) as cdr_src:
                    cdr_tf = cdr_src.transform
                    cdr_h, cdr_w = cdr_src.height, cdr_src.width
                cdr_r, cdr_c = rowcol(
                    cdr_tf,
                    lon_all.astype("float64"),
                    lat_all.astype("float64"),
                )
                self._cdr_row_map = np.clip(
                    np.asarray(cdr_r, dtype="int32").reshape(
                        self._domain_H, self._domain_W
                    ),
                    0,
                    cdr_h - 1,
                )
                self._cdr_col_map = np.clip(
                    np.asarray(cdr_c, dtype="int32").reshape(
                        self._domain_H, self._domain_W
                    ),
                    0,
                    cdr_w - 1,
                )

        # Pre-projected center / neighbor station coordinates come from cached metadata.
        self._center_rows = self.samples["bg_row"].to_numpy(dtype="int32")
        self._center_cols = self.samples["bg_col"].to_numpy(dtype="int32")
        self._neighbor_rows = self._neighbor_df["bg_row"].to_numpy(dtype="int32")
        self._neighbor_cols = self._neighbor_df["bg_col"].to_numpy(dtype="int32")

        # Eager-load static rasters so DataLoader workers share via COW
        for tif_path in self.static_tifs:
            self.raster_cache.get(tif_path)
        if self.landsat_tif:
            self.raster_cache.get(self.landsat_tif)

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

    def _neighbor_day_data(
        self, day: pd.Timestamp
    ) -> tuple[pd.DataFrame | None, np.ndarray | None, np.ndarray | None]:
        bounds = self._neighbor_day_slices.get(pd.Timestamp(day).normalize())
        if bounds is None:
            return None, None, None
        start, stop = bounds
        return (
            self._neighbor_df.iloc[start:stop],
            self._neighbor_rows[start:stop],
            self._neighbor_cols[start:stop],
        )

    def _build_patch_sample(
        self,
        day: pd.Timestamp,
        r0: int,
        c0: int,
        center_fid: str | None = None,
        center_rc: tuple[int, int] | None = None,
        center_row: pd.Series | None = None,
    ):
        day = pd.Timestamp(day).normalize()
        doy = day.dayofyear
        period = _landsat_period(doy)
        H = W = self.patch_size
        r1, c1 = r0 + H, c0 + W

        # --- HRRR background patch (cache hit for same-day samples) ---
        bg = self.raster_cache.get(self._background_path(day))
        bg_patch = bg["data"][self._bg_keep_indices, r0:r1, c0:c1].astype("float32")

        # --- Static TIF patches (pixel-aligned, just slice) ---
        static_patches: list[np.ndarray] = []
        for tif_path, tif_names in zip(self.static_tifs, self._static_tif_band_names):
            s = self.raster_cache.get(tif_path)
            patch = np.nan_to_num(s["data"][:, r0:r1, c0:c1].astype("float32"), nan=0.0)
            static_patches.append(patch)

        # --- Landsat period patch (pixel-aligned, just slice) ---
        landsat_patch: np.ndarray | None = None
        active_landsat_names: list[str] = []
        if self.landsat_tif:
            active_landsat_names = self._landsat_period_names[period]
            ls = self.raster_cache.get(self.landsat_tif)
            b0 = period * _LANDSAT_BANDS
            b1 = b0 + _LANDSAT_BANDS
            landsat_patch = np.nan_to_num(
                ls["data"][b0:b1, r0:r1, c0:c1].astype("float32"), nan=0.0
            )

        # --- rsun patch (pixel-aligned, DOY-cached band read) ---
        rsun_patch: np.ndarray | None = None
        if self.rsun_tif and self._rsun_meta is not None:
            cached_doy = getattr(self, "_rsun_cache_doy", -1)
            if cached_doy != doy:
                with rasterio.open(self.rsun_tif) as rsun_src:
                    band_idx = max(1, min(doy, rsun_src.count))
                    self._rsun_cache_band = rsun_src.read(band_idx).astype("float32")
                    self._rsun_cache_doy = doy
            rsun_patch = np.nan_to_num(
                self._rsun_cache_band[r0:r1, c0:c1], nan=0.0
            ).reshape(1, H, W)

        # --- CDR patch (pre-computed index maps) ---
        cdr_patch: np.ndarray | None = None
        if self.cdr_dir and self._cdr_band_names:
            cdr_path = self._cdr_path(day)
            n_cdr = len(self._cdr_band_names)
            if os.path.exists(cdr_path) and self._cdr_row_map is not None:
                cdr = self.raster_cache.get(cdr_path)
                cr = self._cdr_row_map[r0:r1, c0:c1]
                cc = self._cdr_col_map[r0:r1, c0:c1]
                cdr_patch = np.nan_to_num(
                    cdr["data"][:, cr, cc].astype("float32"), nan=0.0
                )
            else:
                cdr_patch = np.zeros((n_cdr, H, W), dtype="float32")

        # --- Position / time channels (slice pre-computed grids) ---
        pixel_lat = self._domain_lat[r0:r1, c0:c1]
        pixel_lon = self._domain_lon[r0:r1, c0:c1]
        doy_sin, doy_cos = _doy_features(day)
        pos_time = np.stack(
            [
                np.full((H, W), doy_sin, dtype="float32"),
                np.full((H, W), doy_cos, dtype="float32"),
                pixel_lat,
                pixel_lon,
            ]
        )

        # --- Stack and normalize ---
        parts: list[np.ndarray] = [bg_patch] + static_patches
        if landsat_patch is not None:
            parts.append(landsat_patch)
        if rsun_patch is not None:
            parts.append(rsun_patch)
        if cdr_patch is not None:
            parts.append(cdr_patch)
        parts.append(pos_time)
        x_patch = np.concatenate(parts, axis=0)

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

        # --- In-patch station supervision (pre-computed row/col) ---
        sta_rows_list: list[int] = []
        sta_cols_list: list[int] = []
        sta_targets_list: list[list[float]] = []
        sta_valid_list: list[list[bool]] = []
        sta_holdout_list: list[bool] = []
        sta_is_center_list: list[bool] = []

        day_group, sta_r, sta_c = self._neighbor_day_data(day)
        if day_group is not None and sta_r is not None and not day_group.empty:
            in_patch = (sta_r >= r0) & (sta_r < r1) & (sta_c >= c0) & (sta_c < c1)
            for j in np.where(in_patch)[0]:
                sta_rows_list.append(int(sta_r[j] - r0))
                sta_cols_list.append(int(sta_c[j] - c0))
                fid_j = str(day_group.iloc[j]["fid"])
                sta_holdout_list.append(fid_j in self._holdout_fids)
                sta_is_center_list.append(
                    center_fid is not None and fid_j == center_fid
                )
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
            if center_fid is None or center_rc is None or center_row is None:
                raise RuntimeError(
                    f"Tile patch for {day.strftime('%Y-%m-%d')} at ({r0}, {c0}) "
                    "had no supervised stations."
                )
            r_center, c_center = center_rc
            r_in = int(np.clip(r_center - r0, 0, H - 1))
            c_in = int(np.clip(c_center - c0, 0, H - 1))
            sta_rows_list.append(r_in)
            sta_cols_list.append(c_in)
            sta_holdout_list.append(center_fid in self._holdout_fids)
            sta_is_center_list.append(True)
            tgt = []
            vld = []
            for name in self.target_names:
                val = center_row.get(name, float("nan"))
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

    def __getitem__(self, idx: int):
        row = self.samples.iloc[idx]
        day = pd.Timestamp(row["day"]).normalize()

        # Patch bounds from pre-computed center pixel
        r_center = int(self._center_rows[idx])
        c_center = int(self._center_cols[idx])
        r0 = int(
            np.clip(
                r_center - self.patch_size // 2, 0, self._domain_H - self.patch_size
            )
        )
        c0 = int(
            np.clip(
                c_center - self.patch_size // 2, 0, self._domain_W - self.patch_size
            )
        )
        return self._build_patch_sample(
            day=day,
            r0=r0,
            c0=c0,
            center_fid=str(row["fid"]),
            center_rc=(r_center, c_center),
            center_row=row,
        )


class HRRRDayTileBatchDataset(HRRRPatchDataset):
    """Day-centric tile batch dataset for fast Stage B iteration.

    One dataset sample = one day.  ``__getitem__`` loads that day's rasters
    once and returns *K* tile-patch samples.  A dedicated collate function
    (``collate_day_tile_batch``) flattens them into the standard Stage B
    batch format so ``LitPatchAssim`` needs no changes.

    ``set_epoch(epoch)`` varies the per-day tile selection each epoch.
    """

    def __init__(
        self,
        *args,
        tile_stride: int | None = None,
        tiles_per_day: int | None = None,
        tile_sampling_seed: int = 42,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        stride = tile_stride if tile_stride is not None else self.patch_size
        if stride <= 0:
            raise ValueError(f"tile_stride must be > 0, got {stride}")
        if stride > self.patch_size:
            raise ValueError(
                f"tile_stride must be <= patch_size ({self.patch_size}), got {stride}"
            )

        self.tile_stride = int(stride)
        self.tiles_per_day = tiles_per_day
        self.tile_sampling_seed = tile_sampling_seed

        # Build FULL tile index (no per-day cap) so set_epoch() can resample
        station_samples = self.samples
        self._tile_index = _load_or_build_tile_index(
            station_samples=station_samples,
            cache_dir=self.metadata_cache_dir,
            domain_h=self._domain_H,
            domain_w=self._domain_W,
            patch_size=self.patch_size,
            tile_stride=self.tile_stride,
            tiles_per_day=None,
            tile_sampling_seed=self.tile_sampling_seed,
        )
        self._tile_r0 = self._tile_index["tile_r0"].to_numpy(dtype="int32")
        self._tile_c0 = self._tile_index["tile_c0"].to_numpy(dtype="int32")
        self._tile_day_slices = _build_day_slices(self._tile_index["day"])

        # Rebuild samples as one-row-per-day for DayGroupedSampler compatibility
        day_list = sorted(self._tile_index["day"].unique())
        self.samples = pd.DataFrame({"day": day_list})

        self._center_rows = None
        self._center_cols = None
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Update epoch so tile choices change across epochs."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        day = pd.Timestamp(self.samples.iloc[idx]["day"]).normalize()

        bounds = self._tile_day_slices.get(day)
        if bounds is None:
            return []
        start, stop = bounds
        tile_r0 = self._tile_r0[start:stop]
        tile_c0 = self._tile_c0[start:stop]
        n_tiles = len(tile_r0)

        if self.tiles_per_day is not None and n_tiles > self.tiles_per_day:
            rng = np.random.default_rng(
                self.tile_sampling_seed + self._epoch * 100_000 + idx
            )
            chosen = np.sort(rng.choice(n_tiles, self.tiles_per_day, replace=False))
            tile_r0 = tile_r0[chosen]
            tile_c0 = tile_c0[chosen]
            n_tiles = self.tiles_per_day

        samples = []
        for i in range(n_tiles):
            sample = self._build_patch_sample(
                day=day,
                r0=int(tile_r0[i]),
                c0=int(tile_c0[i]),
            )
            samples.append(sample)
        return samples


class HRRRTilePatchDataset(HRRRPatchDataset):
    """Tile-indexed variant of ``HRRRPatchDataset`` for Stage B training.

    The patch geometry is fixed on a regular tile lattice. One dataset sample is one
    occupied ``(day, tile)`` pair, and supervision is provided by all stations that
    fall inside the tile.
    """

    def __init__(
        self,
        *args,
        tile_stride: int | None = None,
        tiles_per_day: int | None = None,
        tile_sampling_seed: int = 42,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        stride = tile_stride if tile_stride is not None else self.patch_size
        if stride <= 0:
            raise ValueError(f"tile_stride must be > 0, got {stride}")
        if stride > self.patch_size:
            raise ValueError(
                f"tile_stride must be <= patch_size ({self.patch_size}), got {stride}"
            )

        self.tile_stride = int(stride)
        self.tiles_per_day = tiles_per_day
        self.tile_sampling_seed = tile_sampling_seed

        station_samples = self.samples
        self.samples = _load_or_build_tile_index(
            station_samples=station_samples,
            cache_dir=self.metadata_cache_dir,
            domain_h=self._domain_H,
            domain_w=self._domain_W,
            patch_size=self.patch_size,
            tile_stride=self.tile_stride,
            tiles_per_day=self.tiles_per_day,
            tile_sampling_seed=self.tile_sampling_seed,
        )
        self._tile_rows = self.samples["tile_r0"].to_numpy(dtype="int32")
        self._tile_cols = self.samples["tile_c0"].to_numpy(dtype="int32")

        # Center arrays are not used in tile mode and keeping them doubles the
        # indexed coordinate payload in memory.
        self._center_rows = None
        self._center_cols = None

    def __getitem__(self, idx: int):
        row = self.samples.iloc[idx]
        return self._build_patch_sample(
            day=pd.Timestamp(row["day"]).normalize(),
            r0=int(self._tile_rows[idx]),
            c0=int(self._tile_cols[idx]),
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


def collate_day_tile_batch(batch):
    """Flatten day-centric batches into the standard Stage B collated format.

    Each item in *batch* is a list of K sample tuples (one day's worth of tiles).
    We flatten all day payloads into a single sample list and delegate to
    ``collate_patch``.
    """
    flat = []
    for day_samples in batch:
        flat.extend(day_samples)
    return collate_patch(flat)
