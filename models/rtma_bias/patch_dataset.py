from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

from process.gridded.rtma_patch_sampler import (
    RtmaPatchConfig,
    _decode_rtma_raw_patch,
    doy_sin_cos,
    sample_rtma_patch,
)

# Per-worker cache of open rasterio readers (fallback path).
_SRC_CACHE: dict[str, rasterio.io.DatasetReader] = {}


def _cached_open(path: str) -> rasterio.io.DatasetReader:
    s = _SRC_CACHE.get(path)
    if s is None or s.closed:
        s = rasterio.open(path)
        _SRC_CACHE[path] = s
    return s


@dataclass(frozen=True)
class PatchDatasetConfig:
    tif_root: str
    rtma_channels: tuple[str, ...] = (
        "tmp_c",
        "dpt_c",
        "ugrd",
        "vgrd",
        "pres_kpa",
        "tcdc_pct",
        "prcp_mm",
        "ea_kpa",
    )
    add_doy: bool = True
    patch_size: int = 64
    preload: bool = True
    terrain_tif: str | None = None
    terrain_channels: tuple[str, ...] = (
        "elevation",
        "slope",
        "aspect_sin",
        "aspect_cos",
        "tpi_4",
        "tpi_10",
    )
    rsun_tif: str | None = None
    landsat_tif: str | None = None
    target_col: str = "delta_log_ea"
    preload_workers: int = 8


def _load_one_cog(
    tif_root: str,
    ymd: str,
    channels: tuple[str, ...],
) -> tuple[str, dict[str, np.ndarray]] | None:
    """Load a single COG and return (ymd, decoded_dict) or None if missing."""
    tif_path = os.path.join(tif_root, f"RTMA_{ymd}.tif")
    if not os.path.exists(tif_path):
        return None
    with rasterio.open(tif_path) as src:
        raw = src.read()  # (B, H, W) int32
        decoded = _decode_rtma_raw_patch(src, raw)
        keep = {c: decoded[c] for c in channels if c in decoded}
        keep["__transform__"] = np.array(src.transform, dtype="float64")
        keep["__shape__"] = np.array([src.height, src.width], dtype="int64")
    return ymd, keep


def _preload_cogs(
    tif_root: str,
    days: list[pd.Timestamp],
    channels: tuple[str, ...],
    max_workers: int = 8,
) -> dict[str, dict[str, np.ndarray]]:
    """Read each unique COG once and decode all bands into memory.

    Uses a thread pool (rasterio/GDAL releases the GIL during I/O) to
    overlap disk reads and LZW decompression.

    Returns ``{yyyymmdd: {channel_name: (H, W) float32 array}}``.
    """
    unique_days = sorted(set(d.strftime("%Y%m%d") for d in days))
    cache: dict[str, dict[str, np.ndarray]] = {}
    total = len(unique_days)
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_load_one_cog, tif_root, ymd, channels): ymd
            for ymd in unique_days
        }
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                ymd, keep = result
                cache[ymd] = keep
            done += 1
            if done % 50 == 0 or done == total:
                mem_gb = (
                    sum(sum(v.nbytes for v in d.values()) for d in cache.values()) / 1e9
                )
                print(
                    f"  preload: {done}/{total} COGs ({mem_gb:.1f} GB)",
                    flush=True,
                )
    return cache


def _compute_channel_stats(
    cache: dict[str, dict[str, np.ndarray]],
    channels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel (mean, std) from preloaded COG arrays.

    Returns two (C,) float64 arrays.  NaN pixels are excluded.
    """
    n_ch = len(channels)
    total = np.zeros(n_ch, dtype="float64")
    total_sq = np.zeros(n_ch, dtype="float64")
    count = np.zeros(n_ch, dtype="float64")

    for cog in cache.values():
        for i, ch in enumerate(channels):
            arr = cog.get(ch)
            if arr is None:
                continue
            valid = arr[np.isfinite(arr)]
            total[i] += valid.sum()
            total_sq[i] += (valid.astype("float64") ** 2).sum()
            count[i] += valid.size

    mean = np.where(count > 0, total / count, 0.0)
    var = np.where(count > 0, total_sq / count - mean**2, 1.0)
    std = np.sqrt(np.maximum(var, 0.0))
    std = np.where(std < 1e-8, 1.0, std)  # avoid division by zero
    return mean.astype("float32"), std.astype("float32")


# ---------------------------------------------------------------------------
# Terrain / RSUN helpers
# ---------------------------------------------------------------------------

_TERRAIN_BAND_NAMES = (
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "tpi_4",
    "tpi_10",
)


def _load_terrain(tif_path: str, channels: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Load terrain GeoTIFF bands into ``{name: (H, W) float32}``."""
    cache: dict[str, np.ndarray] = {}
    with rasterio.open(tif_path) as src:
        descs = [src.descriptions[i] for i in range(src.count)]
        for ch in channels:
            if ch in descs:
                band_idx = descs.index(ch) + 1
            elif ch in _TERRAIN_BAND_NAMES:
                band_idx = _TERRAIN_BAND_NAMES.index(ch) + 1
            else:
                raise ValueError(f"terrain channel '{ch}' not found in {tif_path}")
            cache[ch] = src.read(band_idx).astype("float32", copy=False)
        cache["__transform__"] = np.array(src.transform, dtype="float64")
        cache["__shape__"] = np.array([src.height, src.width], dtype="int64")
    mem = sum(v.nbytes for v in cache.values()) / 1e6
    print(
        f"Terrain loaded: {len(channels)} channels from {tif_path} ({mem:.0f} MB)",
        flush=True,
    )
    return cache


def _load_rsun(tif_path: str) -> dict[str, np.ndarray]:
    """Load 365-band RSUN GeoTIFF. Returns dict with 'data' and '__transform__'."""
    with rasterio.open(tif_path) as src:
        arr = src.read().astype("float32", copy=False)  # (365, H, W)
        tf = np.array(src.transform, dtype="float64")
    mem = arr.nbytes / 1e6
    print(
        f"RSUN loaded: {arr.shape[0]} bands from {tif_path} ({mem:.0f} MB)", flush=True
    )
    return {"data": arr, "__transform__": tf}


def _terrain_channel_stats(
    cache: dict[str, np.ndarray], channels: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (mean, std) from terrain cache."""
    means, stds = [], []
    for ch in channels:
        arr = cache[ch]
        valid = arr[np.isfinite(arr)]
        if valid.size > 0:
            means.append(float(valid.mean()))
            s = float(valid.std())
            stds.append(s if s > 1e-8 else 1.0)
        else:
            means.append(0.0)
            stds.append(1.0)
    return np.array(means, dtype="float32"), np.array(stds, dtype="float32")


def _rsun_channel_stats(rsun: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std across all 365 bands (single channel)."""
    valid = rsun["data"][np.isfinite(rsun["data"])]
    if valid.size > 0:
        m = float(valid.mean())
        s = float(valid.std())
        if s < 1e-8:
            s = 1.0
    else:
        m, s = 0.0, 1.0
    return np.array([m], dtype="float32"), np.array([s], dtype="float32")


def _rtma_to_aux_pixel(
    rtma_row: int,
    rtma_col: int,
    rtma_tf: np.ndarray,
    aux_tf: np.ndarray,
) -> tuple[int, int]:
    """Convert RTMA pixel (row, col) to auxiliary grid pixel coords."""
    # RTMA pixel center → lon/lat
    a, _b, c, _d, e, f = (
        rtma_tf[0],
        rtma_tf[1],
        rtma_tf[2],
        rtma_tf[3],
        rtma_tf[4],
        rtma_tf[5],
    )
    lon = c + (rtma_col + 0.5) * a
    lat = f + (rtma_row + 0.5) * e
    # lon/lat → auxiliary pixel
    aa, _bb, cc, _dd, ee, ff = (
        aux_tf[0],
        aux_tf[1],
        aux_tf[2],
        aux_tf[3],
        aux_tf[4],
        aux_tf[5],
    )
    ac = (lon - cc) / aa
    ar = (lat - ff) / ee
    return int(ar), int(ac)


def _extract_aux_patch(
    full_grid: np.ndarray,
    rtma_r0: int,
    rtma_c0: int,
    ps: int,
    rtma_tf: np.ndarray,
    aux_tf: np.ndarray,
) -> np.ndarray:
    """Extract a (ps, ps) patch from an auxiliary grid aligned to an RTMA window."""
    aH, aW = full_grid.shape[-2], full_grid.shape[-1]
    # Map top-left RTMA pixel to aux grid
    ar0, ac0 = _rtma_to_aux_pixel(rtma_r0, rtma_c0, rtma_tf, aux_tf)

    ndim = full_grid.ndim
    if ndim == 2:
        patch = np.zeros((ps, ps), dtype="float32")
    else:
        patch = np.zeros((full_grid.shape[0], ps, ps), dtype="float32")

    sr = max(0, ar0)
    er = min(aH, ar0 + ps)
    sc = max(0, ac0)
    ec = min(aW, ac0 + ps)
    if sr < er and sc < ec:
        pr = sr - ar0
        pc = sc - ac0
        if ndim == 2:
            patch[pr : pr + (er - sr), pc : pc + (ec - sc)] = full_grid[sr:er, sc:ec]
        else:
            patch[:, pr : pr + (er - sr), pc : pc + (ec - sc)] = full_grid[
                :, sr:er, sc:ec
            ]
    return patch


def _extract_terrain_patch(
    cache: dict[str, np.ndarray],
    channels: tuple[str, ...],
    r0: int,
    c0: int,
    ps: int,
    rtma_tf: np.ndarray,
    rtma_H: int,
    rtma_W: int,
) -> np.ndarray:
    """Extract terrain patch for given RTMA window coords. Returns (C, ps, ps)."""
    aux_tf = cache["__transform__"]
    stack = []
    for ch in channels:
        patch = _extract_aux_patch(cache[ch], r0, c0, ps, rtma_tf, aux_tf)
        stack.append(patch)
    return np.stack(stack, axis=0)


def _extract_rsun_patch(
    rsun: dict[str, np.ndarray],
    day: pd.Timestamp,
    r0: int,
    c0: int,
    ps: int,
    rtma_tf: np.ndarray,
    rtma_H: int,
    rtma_W: int,
) -> np.ndarray:
    """Extract RSUN patch for the given DOY. Returns (1, ps, ps)."""
    doy_idx = day.dayofyear - 1  # 0-based band index
    band = rsun["data"][doy_idx]  # (H, W)
    aux_tf = rsun["__transform__"]
    patch = _extract_aux_patch(band, r0, c0, ps, rtma_tf, aux_tf)
    return patch[None]  # (1, ps, ps)


# ---------------------------------------------------------------------------
# Landsat helpers
# ---------------------------------------------------------------------------

_LANDSAT_BANDS_PER_PERIOD = 7


def _load_landsat(tif_path: str) -> dict[str, np.ndarray]:
    """Load 35-band Landsat composite GeoTIFF.

    Returns dict with 'data' (35, H, W) float32 and '__transform__'.
    """
    with rasterio.open(tif_path) as src:
        arr = src.read().astype("float32", copy=False)  # (35, H, W)
        tf = np.array(src.transform, dtype="float64")
    mem = arr.nbytes / 1e6
    print(
        f"Landsat loaded: {arr.shape[0]} bands from {tif_path} ({mem:.0f} MB)",
        flush=True,
    )
    return {"data": arr, "__transform__": tf}


def _date_to_period(day: pd.Timestamp) -> int:
    """Map a date to Landsat composite period index (0-4)."""
    m, d = day.month, day.day
    if m <= 2:
        return 0
    if m <= 4:
        return 1
    if m <= 6 or (m == 7 and d < 15):
        return 2
    if m <= 8 or (m == 9 and d < 30) or (m == 7 and d >= 15):
        return 3
    return 4


def _extract_landsat_patch(
    landsat: dict[str, np.ndarray],
    day: pd.Timestamp,
    r0: int,
    c0: int,
    ps: int,
    rtma_tf: np.ndarray,
    rtma_H: int,
    rtma_W: int,
) -> np.ndarray:
    """Extract 7-band Landsat patch for the correct period. Returns (7, ps, ps)."""
    period = _date_to_period(day)
    b_start = period * _LANDSAT_BANDS_PER_PERIOD
    b_end = b_start + _LANDSAT_BANDS_PER_PERIOD
    bands = landsat["data"][b_start:b_end]  # (7, H, W)
    aux_tf = landsat["__transform__"]
    patch = _extract_aux_patch(bands, r0, c0, ps, rtma_tf, aux_tf)
    return patch  # (7, ps, ps)


def _landsat_channel_stats(
    landsat: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-spectral-band (mean, std) averaged across all 5 periods.

    Returns two (7,) float32 arrays.
    """
    data = landsat["data"]  # (35, H, W)
    means = np.zeros(7, dtype="float64")
    stds = np.zeros(7, dtype="float64")
    for b in range(7):
        # Gather this spectral band across all 5 periods
        indices = [p * 7 + b for p in range(5)]
        vals = data[indices]  # (5, H, W)
        valid = vals[np.isfinite(vals)]
        if valid.size > 0:
            means[b] = float(valid.mean())
            s = float(valid.std())
            stds[b] = s if s > 1e-8 else 1.0
        else:
            means[b] = 0.0
            stds[b] = 1.0
    return means.astype("float32"), stds.astype("float32")


class RtmaHumidityPatchDataset(Dataset):
    """
    Station-day patch dataset for humidity correction.

    Expects a patch index Parquet built by prep/build_rtma_patch_index.py with columns:
    - fid, day, latitude, longitude, delta_log_ea

    When ``config.preload`` is True (default), all unique COGs are read into
    memory once during ``__init__``.  ``__getitem__`` then performs a pure numpy
    slice (~0.2 ms) instead of a rasterio windowed read (~170 ms).
    """

    def __init__(self, patch_index_parquet: str, config: PatchDatasetConfig):
        self.config = config
        df = pd.read_parquet(patch_index_parquet)
        target_col = config.target_col
        req = {"fid", "day", "latitude", "longitude", target_col}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"patch index missing columns: {sorted(missing)}")
        df = df.copy()
        df["fid"] = df["fid"].astype(str)
        df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["day", "latitude", "longitude", target_col])
        self.df = df.reset_index(drop=True)

        self.rtma_cfg = RtmaPatchConfig(patch_size=int(config.patch_size))
        self._in_channels = len(config.rtma_channels) + (2 if config.add_doy else 0)

        # Pre-load COGs into memory when requested.
        self._cog_cache: dict[str, dict[str, np.ndarray]] | None = None
        if config.preload:
            print(
                f"Pre-loading COGs for {df['day'].nunique()} unique days …", flush=True
            )
            self._cog_cache = _preload_cogs(
                config.tif_root,
                df["day"].tolist(),
                config.rtma_channels,
                max_workers=config.preload_workers,
            )
            print(
                f"Pre-load complete: {len(self._cog_cache)} COGs in memory.", flush=True
            )

        # ---- Terrain (static 6-band grid) ----
        self._terrain_cache: dict[str, np.ndarray] | None = None
        if config.terrain_tif:
            self._terrain_cache = _load_terrain(
                config.terrain_tif, config.terrain_channels
            )
            self._in_channels += len(config.terrain_channels)

        # ---- RSUN (365-band DOY-indexed grid) ----
        self._rsun_cache: dict[str, np.ndarray] | None = None
        if config.rsun_tif:
            self._rsun_cache = _load_rsun(config.rsun_tif)
            self._in_channels += 1

        # ---- Landsat (35-band climatological composite) ----
        self._landsat_cache: dict[str, np.ndarray] | None = None
        if config.landsat_tif:
            self._landsat_cache = _load_landsat(config.landsat_tif)
            self._in_channels += _LANDSAT_BANDS_PER_PERIOD  # +7

        # Per-channel normalisation (mean, std).
        self._norm_mean: np.ndarray | None = None  # (C, 1, 1) float32
        self._norm_std: np.ndarray | None = None  # (C, 1, 1) float32
        if self._cog_cache is not None:
            rtma_mean, rtma_std = _compute_channel_stats(
                self._cog_cache, config.rtma_channels
            )
            # Append stats for doy channels (already ~N(0, 0.7); normalise anyway).
            if config.add_doy:
                rtma_mean = np.concatenate(
                    [rtma_mean, np.array([0.0, 0.0], dtype="float32")]
                )
                rtma_std = np.concatenate(
                    [rtma_std, np.array([1.0, 1.0], dtype="float32")]
                )

            # Terrain channel stats
            if self._terrain_cache is not None:
                t_mean, t_std = _terrain_channel_stats(
                    self._terrain_cache, config.terrain_channels
                )
                rtma_mean = np.concatenate([rtma_mean, t_mean])
                rtma_std = np.concatenate([rtma_std, t_std])

            # RSUN channel stats
            if self._rsun_cache is not None:
                r_mean, r_std = _rsun_channel_stats(self._rsun_cache)
                rtma_mean = np.concatenate([rtma_mean, r_mean])
                rtma_std = np.concatenate([rtma_std, r_std])

            # Landsat channel stats
            if self._landsat_cache is not None:
                l_mean, l_std = _landsat_channel_stats(self._landsat_cache)
                rtma_mean = np.concatenate([rtma_mean, l_mean])
                rtma_std = np.concatenate([rtma_std, l_std])

            self._norm_mean = rtma_mean.astype("float32")[:, None, None]
            self._norm_std = rtma_std.astype("float32")[:, None, None]
            print(
                f"Channel stats computed: {len(rtma_mean)} channels.",
                flush=True,
            )

    @property
    def in_channels(self) -> int:
        return int(self._in_channels)

    @property
    def norm_stats(self) -> dict:
        """Return normalisation stats as a plain dict (for JSON serialisation)."""
        channels = list(self.config.rtma_channels)
        if self.config.add_doy:
            channels += ["doy_sin", "doy_cos"]
        if self._terrain_cache is not None:
            channels += list(self.config.terrain_channels)
        if self._rsun_cache is not None:
            channels += ["rsun"]
        if self._landsat_cache is not None:
            channels += ["ls_b2", "ls_b3", "ls_b4", "ls_b5", "ls_b6", "ls_b7", "ls_b10"]
        return {
            "channels": channels,
            "mean": self._norm_mean[:, 0, 0].tolist()
            if self._norm_mean is not None
            else [],
            "std": self._norm_std[:, 0, 0].tolist()
            if self._norm_std is not None
            else [],
        }

    def save_norm_stats(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.norm_stats, f, indent=2)
        print(f"Norm stats saved to {path}", flush=True)

    def load_norm_stats(self, path: str) -> None:
        with open(path) as f:
            d = json.load(f)
        self._norm_mean = np.array(d["mean"], dtype="float32")[:, None, None]
        self._norm_std = np.array(d["std"], dtype="float32")[:, None, None]

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self._norm_mean is not None:
            x = (x - self._norm_mean) / self._norm_std
        return x

    @staticmethod
    def _tif_path(tif_root: str, day: pd.Timestamp) -> str:
        ymd = day.strftime("%Y%m%d")
        return os.path.join(str(tif_root).rstrip("/"), f"RTMA_{ymd}.tif")

    def __len__(self) -> int:
        return int(len(self.df))

    # ------------------------------------------------------------------
    # Fast path: numpy slice from pre-loaded COG arrays.
    # ------------------------------------------------------------------

    def _getitem_preloaded(self, idx: int):
        r = self.df.iloc[int(idx)]
        lon = float(r["longitude"])
        lat = float(r["latitude"])
        day = pd.Timestamp(r["day"])
        y = float(r[self.config.target_col])
        ymd = day.strftime("%Y%m%d")

        cog = self._cog_cache[ymd]
        tf = cog["__transform__"]
        shp = cog["__shape__"]
        H, W = int(shp[0]), int(shp[1])

        # Inverse affine: pixel coords from lon/lat.
        # transform = (a, b, c, d, e, f, 0, 0, 1)  — rasterio Affine
        a, _b, c, _d, e, f = tf[0], tf[1], tf[2], tf[3], tf[4], tf[5]
        col = (lon - c) / a
        row = (lat - f) / e

        ps = int(self.config.patch_size)
        half = ps // 2
        r0 = int(row) - half
        c0 = int(col) - half

        channels = self.config.rtma_channels
        stack = []
        for ch in channels:
            full = cog[ch]  # (H, W)
            # Handle boundary with zero-padding.
            patch = np.zeros((ps, ps), dtype="float32")
            # Compute overlap between the requested window and the image.
            sr = max(0, r0)
            er = min(H, r0 + ps)
            sc = max(0, c0)
            ec = min(W, c0 + ps)
            if sr < er and sc < ec:
                pr = sr - r0
                pc = sc - c0
                patch[pr : pr + (er - sr), pc : pc + (ec - sc)] = full[sr:er, sc:ec]
            stack.append(patch)

        x = np.stack(stack, axis=0)  # (C, H, W)

        if self.config.add_doy:
            ds, dc = doy_sin_cos(day)
            doy_s = np.full((ps, ps), ds, dtype="float32")
            doy_c = np.full((ps, ps), dc, dtype="float32")
            x = np.concatenate([x, doy_s[None], doy_c[None]], axis=0)

        # Terrain channels (static)
        if self._terrain_cache is not None:
            x = np.concatenate(
                [
                    x,
                    _extract_terrain_patch(
                        self._terrain_cache,
                        self.config.terrain_channels,
                        r0,
                        c0,
                        ps,
                        tf,
                        H,
                        W,
                    ),
                ],
                axis=0,
            )

        # RSUN channel (DOY-indexed)
        if self._rsun_cache is not None:
            x = np.concatenate(
                [x, _extract_rsun_patch(self._rsun_cache, day, r0, c0, ps, tf, H, W)],
                axis=0,
            )

        # Landsat channels (period-indexed, 7 bands)
        if self._landsat_cache is not None:
            x = np.concatenate(
                [
                    x,
                    _extract_landsat_patch(
                        self._landsat_cache, day, r0, c0, ps, tf, H, W
                    ),
                ],
                axis=0,
            )

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(
            "float32", copy=False
        )
        x = self._normalize(x)
        x_t = torch.from_numpy(x)
        y_t = torch.tensor([y], dtype=torch.float32)

        meta = {"fid": str(r["fid"]), "day": day, "lon": lon, "lat": lat}
        if "ea_rtma" in self.df.columns:
            meta["log_ea_rtma"] = float(np.log(r["ea_rtma"]))
        return x_t, y_t, meta

    # ------------------------------------------------------------------
    # Fallback path: on-the-fly COG reads (with file-handle caching).
    # ------------------------------------------------------------------

    def _getitem_ondisk(self, idx: int):
        r = self.df.iloc[int(idx)]
        lon = float(r["longitude"])
        lat = float(r["latitude"])
        day = pd.Timestamp(r["day"])
        y = float(r[self.config.target_col])

        tif_path = self._tif_path(self.config.tif_root, day)
        src = _cached_open(tif_path)
        x_rtma, chan_to_idx, _ = sample_rtma_patch(
            tif_path=tif_path,
            lon=lon,
            lat=lat,
            config=self.rtma_cfg,
            channels=self.config.rtma_channels,
            src=src,
        )

        x = x_rtma
        if self.config.add_doy:
            ds, dc = doy_sin_cos(day)
            h, w = x.shape[1], x.shape[2]
            doy_s = np.full((h, w), ds, dtype="float32")
            doy_c = np.full((h, w), dc, dtype="float32")
            x = np.concatenate([x, doy_s[None, :, :], doy_c[None, :, :]], axis=0)

        # Terrain / RSUN / Landsat for on-disk path: use RTMA file transform
        if (
            self._terrain_cache is not None
            or self._rsun_cache is not None
            or self._landsat_cache is not None
        ):
            ps = int(self.config.patch_size)
            half = ps // 2
            row_px, col_px = src.index(float(lon), float(lat))
            r0 = int(row_px) - half
            c0 = int(col_px) - half
            tf_arr = np.array(src.transform, dtype="float64")
            sH, sW = src.height, src.width

            if self._terrain_cache is not None:
                x = np.concatenate(
                    [
                        x,
                        _extract_terrain_patch(
                            self._terrain_cache,
                            self.config.terrain_channels,
                            r0,
                            c0,
                            ps,
                            tf_arr,
                            sH,
                            sW,
                        ),
                    ],
                    axis=0,
                )
            if self._rsun_cache is not None:
                x = np.concatenate(
                    [
                        x,
                        _extract_rsun_patch(
                            self._rsun_cache, day, r0, c0, ps, tf_arr, sH, sW
                        ),
                    ],
                    axis=0,
                )
            if self._landsat_cache is not None:
                x = np.concatenate(
                    [
                        x,
                        _extract_landsat_patch(
                            self._landsat_cache, day, r0, c0, ps, tf_arr, sH, sW
                        ),
                    ],
                    axis=0,
                )

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(
            "float32", copy=False
        )
        x = self._normalize(x)
        x_t = torch.from_numpy(x)
        y_t = torch.tensor([y], dtype=torch.float32)

        meta = {
            "fid": str(r["fid"]),
            "day": day,
            "lon": lon,
            "lat": lat,
            "tif": tif_path,
            "channels": chan_to_idx,
        }
        if "ea_rtma" in self.df.columns:
            meta["log_ea_rtma"] = float(np.log(r["ea_rtma"]))
        return x_t, y_t, meta

    def __getitem__(self, idx: int):
        if self._cog_cache is not None:
            return self._getitem_preloaded(idx)
        return self._getitem_ondisk(idx)
