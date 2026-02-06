"""
RTMA / URMA station daily tables (fast tabular path).

Purpose
-------
The extraction script `extract/met_data/grid/rtma_extact.py` downloads hourly RTMA/URMA GRIB via Herbie
and writes *hourly* station extracts as monthly Parquets (one file per station per YYYYMM).

For the humidity MVP (ea-first) we want a *daily* station-aligned table that can be updated incrementally
and joined quickly to station observations.

This module converts the hourly monthly station Parquets into daily per-station Parquets with a stable
schema and canonical units, suitable for rapid joins and/or feeding into downstream table builders.

Output contract (per station file)
----------------------------------
Index: daily (UTC day; naive timestamps)
Columns (suffix `_{model}` where model in {rtma, urma}):
  - tmp_{model}   : air temperature at 2 m [degC]
  - dpt_{model}   : dewpoint temperature at 2 m [degC] (if present)
  - spfh_{model}  : specific humidity [kg/kg] (if present)
  - pres_{model}  : surface pressure [kPa] (if present)
  - ugrd_{model}  : u wind component [m/s] (if present)
  - vgrd_{model}  : v wind component [m/s] (if present)
  - wind_{model}  : wind speed [m/s] (computed if u/v present)
  - wdir_{model}  : wind direction [deg] (computed if u/v present)
  - tcdc_{model}  : total cloud cover [%] (if present)
  - prcp_{model}  : daily precip [mm/day] (from ACPC01 sum; if present)
  - ea_{model}    : actual vapor pressure [kPa] (computed from spfh + pres when available)

Notes
-----
- The raw station extract Parquets are not guaranteed to have a single canonical variable naming
  convention. We map from a set of common candidates to the canonical names above.
- Unit conversions use conservative heuristics and can be overridden with CLI flags.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


_DT_COL_CANDIDATES = ("dt", "time", "valid_time", "datetime", "DateTime")


@dataclass(frozen=True)
class UnitHints:
    tmp: str | None = None   # "K" or "C"
    dpt: str | None = None   # "K" or "C"
    pres: str | None = None  # "Pa" or "hPa" or "kPa"
    tcdc: str | None = None  # "fraction" or "percent"
    wdir: str | None = None  # "rad" or "deg"
    spfh: str | None = None  # "kgkg" or "gkg"
    prcp: str | None = None  # "mm" or "scaled_100"


def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def _parse_dt_col(df: pd.DataFrame) -> pd.Series:
    """Return a timezone-naive UTC datetime index."""
    dt_col = _first_present(df.columns, _DT_COL_CANDIDATES)
    if dt_col is None:
        raise KeyError(f"no datetime column found; expected one of {_DT_COL_CANDIDATES}")

    s = df[dt_col]
    if np.issubdtype(s.dtype, np.datetime64):
        out = pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)
    else:
        # Common raw formats:
        # - YYYYMMDDHH
        # - YYYY-MM-DD HH:MM:SS
        out = pd.to_datetime(s, utc=True, errors="coerce", format=None).dt.tz_convert(None)
        if out.isna().all():
            out = pd.to_datetime(s.astype(str), utc=True, errors="coerce", format="%Y%m%d%H").dt.tz_convert(None)

    if out.isna().any():
        out = out[out.notna()]
    return out


def _actual_vapor_pressure_kpa_from_q_and_p(q_kgkg: np.ndarray, pres_kpa: np.ndarray) -> np.ndarray:
    # ea = q * p / (0.622 + 0.378*q)
    q = np.asarray(q_kgkg, dtype="float64")
    p = np.asarray(pres_kpa, dtype="float64")
    denom = 0.622 + 0.378 * q
    denom = np.where(denom == 0, np.nan, denom)
    return (q * p) / denom


def _wind_dir_deg_from_uv(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological wind direction (direction FROM which wind blows), degrees [0, 360)."""
    u = np.asarray(u, dtype="float64")
    v = np.asarray(v, dtype="float64")
    # Meteorological convention: dir = atan2(-u, -v) in degrees, wrapped to [0, 360)
    deg = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    return deg


def _maybe_to_celsius(values: pd.Series, hint: str | None) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    if hint == "C":
        return v
    if hint == "K":
        return v - 273.15
    # heuristic
    med = float(np.nanmedian(v.values)) if np.isfinite(v.values).any() else np.nan
    if np.isfinite(med) and med > 200:
        return v - 273.15
    return v


def _maybe_to_kpa(values: pd.Series, hint: str | None) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    if hint == "kPa":
        return v
    if hint == "hPa":
        return v / 10.0
    if hint == "Pa":
        return v / 1000.0
    med = float(np.nanmedian(v.values)) if np.isfinite(v.values).any() else np.nan
    if not np.isfinite(med):
        return v
    # Pa typically ~100000, hPa ~1000, kPa ~100
    if med > 20000:
        return v / 1000.0
    if 500.0 <= med <= 1500.0:
        return v / 10.0
    return v


def _maybe_to_percent(values: pd.Series, hint: str | None) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    if hint == "percent":
        return v
    if hint == "fraction":
        return v * 100.0
    mx = float(np.nanmax(v.values)) if np.isfinite(v.values).any() else np.nan
    if np.isfinite(mx) and mx <= 1.5:
        return v * 100.0
    return v


def _maybe_to_degrees(values: pd.Series, hint: str | None) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    if hint == "deg":
        return v
    if hint == "rad":
        return np.degrees(v)
    mx = float(np.nanmax(v.values)) if np.isfinite(v.values).any() else np.nan
    if np.isfinite(mx) and mx <= 7.0:
        return np.degrees(v)
    return v


def _maybe_to_kgkg(values: pd.Series, hint: str | None) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    if hint == "kgkg":
        return v
    if hint == "gkg":
        return v / 1000.0
    med = float(np.nanmedian(v.values)) if np.isfinite(v.values).any() else np.nan
    if np.isfinite(med) and med > 0.2:
        # likely g/kg
        return v / 1000.0
    return v


def _maybe_unscale_prcp(values: pd.Series, hint: str | None) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    if hint == "mm":
        return v
    if hint == "scaled_100":
        return v / 100.0
    # heuristic: if 99th percentile looks like hundredths of mm, unscale
    if np.isfinite(v.values).any():
        q99 = float(np.nanpercentile(v.values, 99))
        if q99 > 500.0:
            return v / 100.0
    return v


def canonicalize_hourly(df_raw: pd.DataFrame, model: str, unit_hints: UnitHints) -> pd.DataFrame:
    """Pick and rename columns, convert units, and compute derived hourly fields."""
    model = str(model).lower()
    if model not in {"rtma", "urma"}:
        raise ValueError("model must be one of {'rtma', 'urma'}")

    dt = _parse_dt_col(df_raw)
    df = df_raw.loc[dt.index].copy()
    df.index = dt.values
    df.index.name = "time"

    # Candidate raw names for common RTMA/URMA variables.
    # We keep this permissive to cover both GRIB->cfgrib naming and any earlier exports.
    candidates = {
        "tmp": ("TMP", "tmp", "t2m", "t", "temperature", "air_temperature"),
        "dpt": ("DPT", "dpt", "d2m", "dewpoint", "dew_point_temperature"),
        "spfh": ("SPFH", "spfh", "q", "sph", "specific_humidity"),
        "pres": ("PRES", "pres", "sp", "surface_pressure", "pressure"),
        "ugrd": ("UGRD", "ugrd", "u10", "u", "u_wind", "u_component_of_wind"),
        "vgrd": ("VGRD", "vgrd", "v10", "v", "v_wind", "v_component_of_wind"),
        "wind": ("WIND", "wind", "ws", "wind_speed"),
        "wdir": ("WDIR", "wdir", "wd", "wind_direction"),
        "tcdc": ("TCDC", "tcdc", "tcc", "total_cloud_cover", "cloud_cover"),
        "acpc01": ("ACPC01", "acpc01", "prcp", "tp", "total_precipitation", "precipitation_amount"),
    }

    out = pd.DataFrame(index=df.index)
    col = _first_present(df.columns, candidates["tmp"])
    if col is not None:
        out[f"tmp_{model}"] = _maybe_to_celsius(df[col], unit_hints.tmp)
    col = _first_present(df.columns, candidates["dpt"])
    if col is not None:
        out[f"dpt_{model}"] = _maybe_to_celsius(df[col], unit_hints.dpt)
    col = _first_present(df.columns, candidates["spfh"])
    if col is not None:
        out[f"spfh_{model}"] = _maybe_to_kgkg(df[col], unit_hints.spfh)
    col = _first_present(df.columns, candidates["pres"])
    if col is not None:
        out[f"pres_{model}"] = _maybe_to_kpa(df[col], unit_hints.pres)
    col_u = _first_present(df.columns, candidates["ugrd"])
    if col_u is not None:
        out[f"ugrd_{model}"] = pd.to_numeric(df[col_u], errors="coerce")
    col_v = _first_present(df.columns, candidates["vgrd"])
    if col_v is not None:
        out[f"vgrd_{model}"] = pd.to_numeric(df[col_v], errors="coerce")
    col = _first_present(df.columns, candidates["wind"])
    if col is not None:
        out[f"wind_{model}"] = pd.to_numeric(df[col], errors="coerce")
    col = _first_present(df.columns, candidates["wdir"])
    if col is not None:
        out[f"wdir_{model}"] = _maybe_to_degrees(df[col], unit_hints.wdir)
    col = _first_present(df.columns, candidates["tcdc"])
    if col is not None:
        out[f"tcdc_{model}"] = _maybe_to_percent(df[col], unit_hints.tcdc)
    col = _first_present(df.columns, candidates["acpc01"])
    if col is not None:
        out[f"acpc01_{model}"] = _maybe_unscale_prcp(df[col], unit_hints.prcp)

    # Derive wind/wdir from u/v if missing.
    u_key = f"ugrd_{model}"
    v_key = f"vgrd_{model}"
    if (u_key in out.columns) and (v_key in out.columns):
        if f"wind_{model}" not in out.columns:
            out[f"wind_{model}"] = np.sqrt(out[u_key] ** 2 + out[v_key] ** 2)
        if f"wdir_{model}" not in out.columns:
            out[f"wdir_{model}"] = _wind_dir_deg_from_uv(out[u_key].values, out[v_key].values)

    # Derive ea from spfh + pres where available.
    q_key = f"spfh_{model}"
    p_key = f"pres_{model}"
    if (q_key in out.columns) and (p_key in out.columns):
        out[f"ea_{model}"] = _actual_vapor_pressure_kpa_from_q_and_p(
            out[q_key].values, out[p_key].values
        )

    return out


def hourly_to_daily(df_hourly: pd.DataFrame, model: str) -> pd.DataFrame:
    model = str(model).lower()
    agg = {}
    for c in df_hourly.columns:
        if c == f"acpc01_{model}":
            agg[c] = "sum"
        else:
            agg[c] = "mean"
    daily = df_hourly.resample("D").agg(agg)
    # Rename precip accumulation to prcp_{model} and drop raw accumulation name
    acc = f"acpc01_{model}"
    if acc in daily.columns:
        daily[f"prcp_{model}"] = daily[acc]
        daily.drop(columns=[acc], inplace=True)
    return daily


def _iter_station_files(raw_station_root: str) -> Iterable[tuple[str, list[str]]]:
    """Yield (fid, [parquet_files]) for each station subdir in raw_station_root."""
    for fid in sorted(os.listdir(raw_station_root)):
        station_dir = os.path.join(raw_station_root, fid)
        if not os.path.isdir(station_dir):
            continue
        files = [
            os.path.join(station_dir, fn)
            for fn in os.listdir(station_dir)
            if fn.endswith(".parquet") and fn.startswith(f"{fid}_")
        ]
        if files:
            yield fid, sorted(files)


def build_daily_tables(
    raw_station_root: str,
    out_daily_root: str,
    model: str,
    unit_hints: UnitHints,
    overwrite: bool = False,
) -> None:
    os.makedirs(out_daily_root, exist_ok=True)
    for fid, files in _iter_station_files(raw_station_root):
        out_file = os.path.join(out_daily_root, f"{fid}.parquet")
        if os.path.exists(out_file) and not overwrite:
            continue

        frames = []
        for p in files:
            try:
                df_raw = pd.read_parquet(p)
            except Exception:
                continue
            try:
                frames.append(canonicalize_hourly(df_raw, model=model, unit_hints=unit_hints))
            except Exception:
                continue

        if not frames:
            continue

        hourly = pd.concat(frames, axis=0).sort_index()
        hourly = hourly.loc[~hourly.index.duplicated(keep="first")]
        daily = hourly_to_daily(hourly, model=model)
        daily.to_parquet(out_file)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build daily RTMA/URMA per-station Parquets from hourly station extracts.")
    p.add_argument("--raw-station-root", required=True, help="Root dir containing per-station subdirs of monthly Parquets.")
    p.add_argument("--out-daily-root", required=True, help="Output directory for per-station daily Parquets.")
    p.add_argument("--model", required=True, choices=["rtma", "urma"], help="Model name for output suffixing.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output station files.")
    # Optional unit hints (override heuristics)
    p.add_argument("--tmp-units", choices=["K", "C"], default=None)
    p.add_argument("--dpt-units", choices=["K", "C"], default=None)
    p.add_argument("--pres-units", choices=["Pa", "hPa", "kPa"], default=None)
    p.add_argument("--tcdc-units", choices=["fraction", "percent"], default=None)
    p.add_argument("--wdir-units", choices=["rad", "deg"], default=None)
    p.add_argument("--spfh-units", choices=["kgkg", "gkg"], default=None)
    p.add_argument("--prcp-units", choices=["mm", "scaled_100"], default=None)
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    hints = UnitHints(
        tmp=a.tmp_units,
        dpt=a.dpt_units,
        pres=a.pres_units,
        tcdc=a.tcdc_units,
        wdir=a.wdir_units,
        spfh=a.spfh_units,
        prcp=a.prcp_units,
    )
    build_daily_tables(
        raw_station_root=a.raw_station_root,
        out_daily_root=a.out_daily_root,
        model=a.model,
        unit_hints=hints,
        overwrite=bool(a.overwrite),
    )


if __name__ == "__main__":
    main()

