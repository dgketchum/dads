from __future__ import annotations

import os
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


def _preload_cogs(
    tif_root: str,
    days: list[pd.Timestamp],
    channels: tuple[str, ...],
) -> dict[str, dict[str, np.ndarray]]:
    """Read each unique COG once and decode all bands into memory.

    Returns ``{yyyymmdd: {channel_name: (H, W) float32 array}}``.
    """
    unique_days = sorted(set(d.strftime("%Y%m%d") for d in days))
    cache: dict[str, dict[str, np.ndarray]] = {}
    total = len(unique_days)
    for i, ymd in enumerate(unique_days):
        tif_path = os.path.join(tif_root, f"RTMA_{ymd}.tif")
        if not os.path.exists(tif_path):
            continue
        with rasterio.open(tif_path) as src:
            raw = src.read()  # (B, H, W) int32
            decoded = _decode_rtma_raw_patch(src, raw)
            # Keep only the requested channels + geo transform info.
            keep = {c: decoded[c] for c in channels if c in decoded}
            # Store the transform for lon/lat → pixel conversion.
            keep["__transform__"] = np.array(src.transform, dtype="float64")
            keep["__shape__"] = np.array([src.height, src.width], dtype="int64")
        cache[ymd] = keep
        if (i + 1) % 50 == 0 or (i + 1) == total:
            mem_gb = sum(
                sum(v.nbytes for v in d.values()) for d in cache.values()
            ) / 1e9
            print(f"  preload: {i + 1}/{total} COGs ({mem_gb:.1f} GB)", flush=True)
    return cache


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
        req = {"fid", "day", "latitude", "longitude", "delta_log_ea"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"patch index missing columns: {sorted(missing)}")
        df = df.copy()
        df["fid"] = df["fid"].astype(str)
        df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["day", "latitude", "longitude", "delta_log_ea"])
        self.df = df.reset_index(drop=True)

        self.rtma_cfg = RtmaPatchConfig(patch_size=int(config.patch_size))
        self._in_channels = len(config.rtma_channels) + (2 if config.add_doy else 0)

        # Pre-load COGs into memory when requested.
        self._cog_cache: dict[str, dict[str, np.ndarray]] | None = None
        if config.preload:
            print(f"Pre-loading COGs for {df['day'].nunique()} unique days …", flush=True)
            self._cog_cache = _preload_cogs(
                config.tif_root,
                df["day"].tolist(),
                config.rtma_channels,
            )
            print(f"Pre-load complete: {len(self._cog_cache)} COGs in memory.", flush=True)

    @property
    def in_channels(self) -> int:
        return int(self._in_channels)

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
        y = float(r["delta_log_ea"])
        ymd = day.strftime("%Y%m%d")

        cog = self._cog_cache[ymd]
        tf = cog["__transform__"]
        shp = cog["__shape__"]
        H, W = int(shp[0]), int(shp[1])

        # Inverse affine: pixel coords from lon/lat.
        # transform = (a, b, c, d, e, f, 0, 0, 1)  — rasterio Affine
        a, b, c, d, e, f = tf[0], tf[1], tf[2], tf[3], tf[4], tf[5]
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

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)
        x_t = torch.from_numpy(x)
        y_t = torch.tensor([y], dtype=torch.float32)

        meta = {"fid": str(r["fid"]), "day": day, "lon": lon, "lat": lat}
        return x_t, y_t, meta

    # ------------------------------------------------------------------
    # Fallback path: on-the-fly COG reads (with file-handle caching).
    # ------------------------------------------------------------------

    def _getitem_ondisk(self, idx: int):
        r = self.df.iloc[int(idx)]
        lon = float(r["longitude"])
        lat = float(r["latitude"])
        day = pd.Timestamp(r["day"])
        y = float(r["delta_log_ea"])

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

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)
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
        return x_t, y_t, meta

    def __getitem__(self, idx: int):
        if self._cog_cache is not None:
            return self._getitem_preloaded(idx)
        return self._getitem_ondisk(idx)
