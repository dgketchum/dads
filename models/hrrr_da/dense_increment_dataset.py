"""
Dense-increment dataset for grid backbone pretraining (Stage A).

Each sample is a (day, tile) pair drawn from a random position on the
full PNW domain.  The target is the dense teacher increment
``URMA − HRRR`` at every pixel in the tile.

Tile origins are epoch-diverse: ``set_epoch(epoch)`` changes the RNG seed
used to draw tile positions, so each epoch sees different tiles for the
same day while remaining deterministic within an epoch.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd
import rasterio
import torch
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from torch.utils.data import Dataset

from models.hrrr_da.hetero_dataset import (
    _RasterCache,
    _discover_band_names,
    _doy_features,
)
from models.hrrr_da.patch_assim_dataset import (
    _LANDSAT_BANDS,
    _LANDSAT_PERIODS,
    _compute_cdr_norm_stats,
    _compute_raster_norm_stats,
    _compute_rsun_norm_stats,
    _landsat_period,
)

# URMA band name → HRRR band name for the variables we want increments of.
_DEFAULT_INCREMENT_MAP: dict[str, str] = {
    "tmax_c": "tmax_hrrr",
    "tmin_c": "tmin_hrrr",
}


class DenseIncrementDataset(Dataset):
    """Random-tile dataset returning ``(x_patch, y_dense, y_valid)``.

    Parameters
    ----------
    background_dir, teacher_dir : str
        Directories of pixel-aligned daily HRRR and URMA 1 km COGs.
    increment_map : dict[str, str]
        ``{urma_band_name: hrrr_band_name}``.  The teacher target for each
        entry is ``urma[band] − hrrr[band]``.
    tiles_per_day : int
        Number of random tiles drawn per available day.
    """

    def __init__(
        self,
        background_dir: str,
        background_pattern: str,
        teacher_dir: str,
        teacher_pattern: str = "URMA_1km_{date}.tif",
        static_tifs: list[str] | None = None,
        landsat_tif: str | None = None,
        rsun_tif: str | None = None,
        cdr_dir: str | None = None,
        cdr_pattern: str = "CDR_005deg_{date}.tif",
        train_days: set | None = None,
        increment_map: dict[str, str] | None = None,
        drop_bands: list[str] | None = None,
        norm_stats: dict | None = None,
        patch_size: int = 64,
        tiles_per_day: int = 8,
        base_seed: int = 42,
        cache_size: int = 8,
    ):
        super().__init__()
        self.background_dir = background_dir
        self.background_pattern = background_pattern
        self.teacher_dir = teacher_dir
        self.teacher_pattern = teacher_pattern
        self.static_tifs = list(static_tifs) if static_tifs else []
        self.landsat_tif = landsat_tif
        self.rsun_tif = rsun_tif
        self.cdr_dir = cdr_dir
        self.cdr_pattern = cdr_pattern
        self.patch_size = patch_size
        self.tiles_per_day = tiles_per_day
        self.base_seed = base_seed
        self._epoch = 0
        self._drop_bands = set(drop_bands or [])

        self.increment_map = dict(increment_map or _DEFAULT_INCREMENT_MAP)
        self.target_names = [f"delta_{urma_name}" for urma_name in self.increment_map]

        actual_cache = max(
            cache_size,
            len(self.static_tifs)
            + 3
            + (1 if landsat_tif else 0)
            + (1 if cdr_dir else 0),
        )
        self.raster_cache = _RasterCache(max_items=actual_cache)

        # --- Discover available (HRRR ∩ URMA) days in train_days ---
        if train_days is not None:
            day_set = {pd.Timestamp(d).normalize() for d in train_days}
        else:
            day_set = None

        avail_days: list[pd.Timestamp] = []
        bg_files = sorted(os.listdir(background_dir))
        for fn in bg_files:
            if not fn.endswith(".tif"):
                continue
            # Extract date from pattern like HRRR_1km_20180101.tif
            date_str = fn.replace(".tif", "").split("_")[-1]
            if len(date_str) != 8:
                continue
            try:
                day = pd.Timestamp(date_str)
            except ValueError:
                continue
            if day_set is not None and day not in day_set:
                continue
            # Check URMA exists
            urma_path = self._teacher_path(day)
            if os.path.exists(urma_path):
                avail_days.append(day)

        if not avail_days:
            raise ValueError("No days found with both HRRR and URMA COGs.")

        self._days = sorted(avail_days)

        # Build a samples DataFrame for DayGroupedSampler compatibility
        rows = []
        for day in self._days:
            for tile_idx in range(tiles_per_day):
                rows.append({"day": day, "tile_idx": tile_idx})
        self.samples = pd.DataFrame(rows)

        # --- Band names ---
        example_bg = self._background_path(self._days[0])
        all_bg_names = _discover_band_names(example_bg, "bg")
        if self._drop_bands:
            self._bg_keep_indices = [
                i for i, n in enumerate(all_bg_names) if n not in self._drop_bands
            ]
            self.background_feature_names: list[str] = [
                all_bg_names[i] for i in self._bg_keep_indices
            ]
        else:
            self._bg_keep_indices = list(range(len(all_bg_names)))
            self.background_feature_names = list(all_bg_names)

        # URMA band index lookup
        example_urma = self._teacher_path(self._days[0])
        urma_band_names = _discover_band_names(example_urma, "urma")
        self._urma_band_indices: dict[str, int] = {
            name: i for i, name in enumerate(urma_band_names)
        }
        # HRRR band index lookup (from the full set, pre-drop)
        self._hrrr_band_indices: dict[str, int] = {
            name: i for i, name in enumerate(all_bg_names)
        }

        # Validate increment_map bands exist
        for urma_name, hrrr_name in self.increment_map.items():
            if urma_name not in self._urma_band_indices:
                raise ValueError(
                    f"URMA band '{urma_name}' not found. Available: {urma_band_names}"
                )
            if hrrr_name not in self._hrrr_band_indices:
                raise ValueError(
                    f"HRRR band '{hrrr_name}' not found. Available: {all_bg_names}"
                )

        # Static TIFs
        self._static_tif_band_names: list[list[str]] = []
        self.static_feature_names: list[str] = []
        for tif_path in self.static_tifs:
            prefix = os.path.splitext(os.path.basename(tif_path))[0]
            names = _discover_band_names(tif_path, prefix)
            self._static_tif_band_names.append(names)
            self.static_feature_names.extend(names)

        # Landsat
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
        _rep_landsat = self._landsat_period_names[0] if self.landsat_tif else []

        # rsun
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

        # CDR
        self._cdr_band_names: list[str] = []
        if self.cdr_dir:
            example_cdr = self._cdr_path(self._days[0])
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

        # --- Domain geometry ---
        bg_data = self.raster_cache.get(example_bg)
        bg_crs = bg_data["crs"]
        bg_tf = bg_data["transform"]
        self._domain_H = bg_data["data"].shape[1]
        self._domain_W = bg_data["data"].shape[2]

        # Lat/lon grids for position channels + CDR lookup
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

        # CDR index maps
        self._cdr_row_map: np.ndarray | None = None
        self._cdr_col_map: np.ndarray | None = None
        if self.cdr_dir and self._cdr_band_names:
            example_cdr_path = self._cdr_path(self._days[0])
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

        # --- Norm stats ---
        if norm_stats is not None:
            self.norm_stats = dict(norm_stats)
        else:
            self.norm_stats = self._compute_all_norm_stats(bg_data, all_bg_names)

        # Eager-load static rasters
        for tif_path in self.static_tifs:
            self.raster_cache.get(tif_path)
        if self.landsat_tif:
            self.raster_cache.get(self.landsat_tif)

    # ------------------------------------------------------------------ paths

    def _background_path(self, day: pd.Timestamp) -> str:
        return os.path.join(
            self.background_dir,
            self.background_pattern.format(date=day.strftime("%Y%m%d")),
        )

    def _teacher_path(self, day: pd.Timestamp) -> str:
        return os.path.join(
            self.teacher_dir,
            self.teacher_pattern.format(date=day.strftime("%Y%m%d")),
        )

    def _cdr_path(self, day: pd.Timestamp) -> str:
        return os.path.join(
            self.cdr_dir,
            self.cdr_pattern.format(date=day.strftime("%Y%m%d")),
        )

    # ------------------------------------------------------------------ norm

    def _compute_all_norm_stats(
        self, bg_data: dict, all_bg_names: list[str]
    ) -> dict[str, dict[str, float]]:
        """Compute per-channel norm stats from rasters (all training days).

        Uses streaming Welford accumulation so memory stays O(pixels-per-day)
        regardless of how many days are processed.
        """
        stats: dict[str, dict[str, float]] = {}

        # Background bands — stream all training days
        n_bands = len(self.background_feature_names)
        count = np.zeros(n_bands, dtype="float64")
        running_mean = np.zeros(n_bands, dtype="float64")
        running_m2 = np.zeros(n_bands, dtype="float64")

        print(
            f"Computing background norm stats over {len(self._days)} days...",
            flush=True,
        )
        for d_idx, sday in enumerate(self._days):
            bg_path = self._background_path(sday)
            if not os.path.exists(bg_path):
                continue
            sdata = self.raster_cache.get(bg_path)
            for b, (keep_i, name) in enumerate(
                zip(self._bg_keep_indices, self.background_feature_names)
            ):
                band = sdata["data"][keep_i].ravel().astype("float64")
                valid = band[np.isfinite(band)]
                if len(valid) == 0:
                    continue
                # Welford online update
                for val in [valid]:  # batch update
                    n_new = len(val)
                    new_mean = val.mean()
                    new_m2 = val.var() * n_new
                    n_old = count[b]
                    n_total = n_old + n_new
                    delta = new_mean - running_mean[b]
                    running_mean[b] = (
                        n_old * running_mean[b] + n_new * new_mean
                    ) / n_total
                    running_m2[b] += new_m2 + delta**2 * n_old * n_new / n_total
                    count[b] = n_total
            if (d_idx + 1) % 100 == 0:
                print(f"  norm stats: {d_idx + 1}/{len(self._days)} days", flush=True)

        for b, name in enumerate(self.background_feature_names):
            if count[b] > 0:
                std = float(np.sqrt(running_m2[b] / count[b]))
                stats[name] = {
                    "mean": float(running_mean[b]),
                    "std": max(std, 1e-8),
                }
        print("Background norm stats computed.", flush=True)

        # Static TIFs
        for tif_path, tif_names in zip(self.static_tifs, self._static_tif_band_names):
            uncovered = [n for n in tif_names if n not in stats]
            if uncovered:
                raster_stats = _compute_raster_norm_stats(tif_path, tif_names)
                stats.update(
                    {n: raster_stats[n] for n in uncovered if n in raster_stats}
                )

        # Landsat
        if self.landsat_tif:
            ls_stats = _compute_raster_norm_stats(
                self.landsat_tif, self._all_landsat_band_names
            )
            stats.update(ls_stats)

        # rsun
        if self.rsun_tif:
            stats["rsun"] = _compute_rsun_norm_stats(self.rsun_tif)

        # CDR
        if self.cdr_dir and self._cdr_band_names:
            cdr_stats = _compute_cdr_norm_stats(
                self.cdr_dir, self.cdr_pattern, self.samples, self._cdr_band_names
            )
            stats.update(cdr_stats)

        # Lat/lon
        lat_valid = self._domain_lat[np.isfinite(self._domain_lat)]
        lon_valid = self._domain_lon[np.isfinite(self._domain_lon)]
        if len(lat_valid):
            stats["latitude"] = {
                "mean": float(lat_valid.mean()),
                "std": float(max(lat_valid.std(), 1e-8)),
            }
        if len(lon_valid):
            stats["longitude"] = {
                "mean": float(lon_valid.mean()),
                "std": float(max(lon_valid.std(), 1e-8)),
            }

        return stats

    # ------------------------------------------------------------------ epoch

    def set_epoch(self, epoch: int) -> None:
        """Update the epoch counter so tile origins change each epoch."""
        self._epoch = epoch

    # ------------------------------------------------------------------ core

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples.iloc[idx]
        day = pd.Timestamp(row["day"]).normalize()
        tile_idx = int(row["tile_idx"])
        doy = day.dayofyear
        period = _landsat_period(doy)
        H = W = self.patch_size

        # --- Tile origin (epoch-diverse, deterministic across processes) ---
        key = f"{self.base_seed + self._epoch}:{day.strftime('%Y%m%d')}:{tile_idx}"
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        r0 = int(rng.integers(0, self._domain_H - H + 1))
        c0 = int(rng.integers(0, self._domain_W - W + 1))
        r1, c1 = r0 + H, c0 + W

        # --- HRRR background patch ---
        bg = self.raster_cache.get(self._background_path(day))
        bg_patch = bg["data"][self._bg_keep_indices, r0:r1, c0:c1].astype("float32")

        # --- URMA teacher patch (raw, for target computation) ---
        urma = self.raster_cache.get(self._teacher_path(day))

        # --- Dense teacher target: URMA − HRRR ---
        n_targets = len(self.increment_map)
        y_dense = np.empty((n_targets, H, W), dtype="float32")
        y_valid = np.empty((n_targets, H, W), dtype="bool")
        for t_idx, (urma_name, hrrr_name) in enumerate(self.increment_map.items()):
            urma_band = urma["data"][
                self._urma_band_indices[urma_name], r0:r1, c0:c1
            ].astype("float32")
            hrrr_band = bg["data"][
                self._hrrr_band_indices[hrrr_name], r0:r1, c0:c1
            ].astype("float32")
            delta = urma_band - hrrr_band
            valid = np.isfinite(delta)
            y_dense[t_idx] = np.where(valid, delta, 0.0)
            y_valid[t_idx] = valid

        # --- Static TIF patches ---
        static_patches: list[np.ndarray] = []
        for tif_path in self.static_tifs:
            s = self.raster_cache.get(tif_path)
            patch = np.nan_to_num(s["data"][:, r0:r1, c0:c1].astype("float32"), nan=0.0)
            static_patches.append(patch)

        # --- Landsat period patch ---
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

        # --- rsun patch ---
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

        # --- CDR patch ---
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

        # --- Position / time channels ---
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

        return (
            torch.from_numpy(x_patch),
            torch.from_numpy(y_dense),
            torch.from_numpy(y_valid),
        )


def collate_dense_increment(batch):
    """Simple stack collate — all tiles are the same shape."""
    x_list, y_list, v_list = zip(*batch)
    return torch.stack(x_list), torch.stack(y_list), torch.stack(v_list)
