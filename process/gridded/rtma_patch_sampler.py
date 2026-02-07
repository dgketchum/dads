"""
RTMA daily COG patch sampler.

This supports the humidity MVP patch-based bias correction model:
- Inputs are dense gridded channels (RTMA + optional static rasters).
- Supervision is sparse (station residuals), so training consumes station-centered patches.

Assumptions
-----------
- RTMA COGs are EPSG:4326 with band order/descriptions matching the EE export:
  PRES, TMP, DPT, UGRD, VGRD, SPFH, WDIR, WIND, TCDC, ACPC01
- Scaling is implicit (see notes/RTMA_MVP_PROGRESS.md):
  - PRES stored as hPa ints (kPa = hPa / 10)
  - Most other fields stored as int(value * 100) (decode with / 100)

This module intentionally keeps decoding logic consistent with
process/gridded/rtma_station_daily.py to avoid drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
import rasterio
from rasterio.windows import Window

from process.gridded.rtma_station_daily import (
    _RTMA_DAILY_BAND_ORDER,
    _band_index,
    _sat_vapor_pressure_kpa_from_dewpoint_c,
    _wind_dir_deg_from_uv,
)


@dataclass(frozen=True)
class RtmaPatchConfig:
    patch_size: int = 64  # pixels; will be forced to even>=2 or odd>=3 by caller
    boundless: bool = True
    nodata_to_nan: bool = True


def _ensure_patch_size(patch_size: int) -> int:
    s = int(patch_size)
    if s < 2:
        raise ValueError("patch_size must be >= 2")
    return s


def _rtma_tif_path(tif_root: str, day_yyyymmdd: str) -> str:
    return f"{tif_root.rstrip('/')}/RTMA_{day_yyyymmdd}.tif"


@lru_cache(maxsize=1024)
def _band_name_to_index_from_desc(descs: tuple[str | None, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, d in enumerate(descs):
        if not d:
            continue
        out[str(d)] = i
    return out


def _read_window(
    src: rasterio.io.DatasetReader,
    window: Window,
    boundless: bool,
) -> np.ndarray:
    # rasterio returns (bands, H, W)
    fill = src.nodata
    if fill is None:
        # Prefer NaN for floats, but the source is Int32; use a sentinel that will be converted to NaN.
        fill = -2147483648
    arr = src.read(window=window, boundless=boundless, fill_value=fill)
    return arr


def _nanify(arr: np.ndarray, nodata: float | int | None) -> np.ndarray:
    out = arr.astype("float32", copy=False)
    if nodata is not None:
        out = np.where(out == float(nodata), np.nan, out)
    out = np.where(np.isfinite(out), out, np.nan)
    return out


def _decode_rtma_raw_patch(
    src: rasterio.io.DatasetReader,
    raw_bhw: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Decode a raw RTMA patch into canonical units.

    raw_bhw: (B, H, W) as returned by rasterio.
    Returns dict of decoded (H, W) float32 arrays.
    """
    if raw_bhw.ndim != 3:
        raise ValueError("expected raw patch shape (bands, H, W)")
    raw = _nanify(raw_bhw, nodata=src.nodata)

    # Prefer band descriptions when present; fall back to expected order.
    try:
        descs = tuple(src.descriptions or ())
    except Exception:
        descs = ()
    idx_desc = _band_name_to_index_from_desc(descs) if descs else {}
    idx = _band_index(src)

    def _get(name: str) -> np.ndarray:
        i = idx.get(name)
        if i is None:
            i = idx_desc.get(name)
        if i is None and len(_RTMA_DAILY_BAND_ORDER) == raw.shape[0]:
            i = _RTMA_DAILY_BAND_ORDER.index(name)
        if i is None or int(i) >= raw.shape[0]:
            return np.full(raw.shape[1:], np.nan, dtype="float32")
        return raw[int(i), :, :].astype("float32", copy=False)

    pres_hpa = _get("PRES")
    tmp_c = _get("TMP") / 100.0
    dpt_c = _get("DPT") / 100.0
    spfh = _get("SPFH") / 100.0
    ugrd = _get("UGRD") / 100.0
    vgrd = _get("VGRD") / 100.0
    wind = _get("WIND") / 100.0
    wdir = _get("WDIR") / 100.0
    tcdc_scaled = _get("TCDC") / 100.0
    prcp_mm = _get("ACPC01") / 100.0

    if np.isnan(wind).all():
        wind = np.sqrt(ugrd ** 2 + vgrd ** 2)
    if np.isnan(wdir).all():
        wdir = _wind_dir_deg_from_uv(ugrd, vgrd).astype("float32", copy=False)

    # Convert cloud cover to percent while tolerating fraction-like encodings.
    tcdc = tcdc_scaled.copy()
    finite_tcdc = tcdc[np.isfinite(tcdc)]
    if finite_tcdc.size and float(np.nanmax(finite_tcdc)) <= 1.5:
        tcdc = tcdc * 100.0

    # PRES appears stored as hPa ints in the EE-exported COGs.
    pres_kpa = pres_hpa / 10.0

    # Humidity baseline for MVP: derive from dewpoint.
    ea_kpa = _sat_vapor_pressure_kpa_from_dewpoint_c(dpt_c)
    ea_kpa = ea_kpa.astype("float32", copy=False)

    return {
        "pres_kpa": pres_kpa,
        "tmp_c": tmp_c,
        "dpt_c": dpt_c,
        "spfh": spfh,
        "ugrd": ugrd,
        "vgrd": vgrd,
        "wind_ms": wind,
        "wdir_deg": wdir,
        "tcdc_pct": tcdc,
        "prcp_mm": prcp_mm,
        "ea_kpa": ea_kpa,
    }


def sample_rtma_patch(
    tif_path: str,
    lon: float,
    lat: float,
    config: RtmaPatchConfig,
    channels: Iterable[str],
    src: rasterio.io.DatasetReader | None = None,
) -> tuple[np.ndarray, dict[str, int], tuple[int, int]]:
    """
    Returns (X, channel_to_idx, (row, col)).

    X: float32 array shaped (C, H, W).

    If *src* is provided (a pre-opened rasterio reader for *tif_path*), it is
    used directly and **not** closed.  This allows callers to cache open file
    handles and avoid repeated open/close overhead.
    """
    patch_size = _ensure_patch_size(config.patch_size)
    channels = list(channels)
    if not channels:
        raise ValueError("channels must be non-empty")

    owned = src is None
    if owned:
        src = rasterio.open(tif_path)
    try:
        row, col = src.index(float(lon), float(lat))
        # Window is defined in (col_off, row_off) pixel space.
        half = patch_size // 2
        w = Window(col_off=int(col - half), row_off=int(row - half), width=patch_size, height=patch_size)
        raw_bhw = _read_window(src, window=w, boundless=bool(config.boundless))
        decoded = _decode_rtma_raw_patch(src, raw_bhw=raw_bhw)
    finally:
        if owned:
            src.close()

    chan_to_idx: dict[str, int] = {}
    stack = []
    for i, c in enumerate(channels):
        if c not in decoded:
            raise KeyError(f"unknown channel '{c}'; available: {sorted(decoded.keys())}")
        chan_to_idx[c] = i
        stack.append(decoded[c])
    x = np.stack(stack, axis=0).astype("float32", copy=False)
    return x, chan_to_idx, (int(row), int(col))


def doy_sin_cos(day: np.datetime64 | str) -> tuple[float, float]:
    import pandas as pd
    ts = pd.Timestamp(day)
    doy = float(ts.dayofyear)
    ang = 2.0 * np.pi * doy / 365.25
    return float(np.sin(ang)), float(np.cos(ang))

