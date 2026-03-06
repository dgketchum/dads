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
import glob
import os
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio


_DT_COL_CANDIDATES = ("dt", "time", "valid_time", "datetime", "DateTime")
_STATION_ID_COL_CANDIDATES = ("fid", "STAID", "station_id", "station", "id")
_STATION_LAT_COL_CANDIDATES = ("lat", "latitude", "LAT", "Latitude")
_STATION_LON_COL_CANDIDATES = ("lon", "longitude", "LON", "Longitude")
_RTMA_DAILY_BAND_ORDER = (
    "PRES",
    "TMP",
    "DPT",
    "UGRD",
    "VGRD",
    "SPFH",
    "WDIR",
    "WIND",
    "TCDC",
    "ACPC01",
)


@dataclass(frozen=True)
class UnitHints:
    tmp: str | None = None  # "K" or "C"
    dpt: str | None = None  # "K" or "C"
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
        raise KeyError(
            f"no datetime column found; expected one of {_DT_COL_CANDIDATES}"
        )

    s = df[dt_col]
    if np.issubdtype(s.dtype, np.datetime64):
        out = pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)
    else:
        # Common raw formats:
        # - YYYYMMDDHH
        # - YYYY-MM-DD HH:MM:SS
        out = pd.to_datetime(s, utc=True, errors="coerce", format=None).dt.tz_convert(
            None
        )
        if out.isna().all():
            out = pd.to_datetime(
                s.astype(str), utc=True, errors="coerce", format="%Y%m%d%H"
            ).dt.tz_convert(None)

    if out.isna().any():
        out = out[out.notna()]
    return out


def _actual_vapor_pressure_kpa_from_q_and_p(
    q_kgkg: np.ndarray, pres_kpa: np.ndarray
) -> np.ndarray:
    # ea = q * p / (0.622 + 0.378*q)
    q = np.asarray(q_kgkg, dtype="float64")
    p = np.asarray(pres_kpa, dtype="float64")
    denom = 0.622 + 0.378 * q
    denom = np.where(denom == 0, np.nan, denom)
    return (q * p) / denom


def _sat_vapor_pressure_kpa_from_dewpoint_c(dpt_c: np.ndarray) -> np.ndarray:
    td = np.asarray(dpt_c, dtype="float64")
    return 0.6108 * np.exp((17.27 * td) / (td + 237.3))


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


def canonicalize_hourly(
    df_raw: pd.DataFrame, model: str, unit_hints: UnitHints
) -> pd.DataFrame:
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
        "acpc01": (
            "ACPC01",
            "acpc01",
            "prcp",
            "tp",
            "total_precipitation",
            "precipitation_amount",
        ),
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
            out[f"wdir_{model}"] = _wind_dir_deg_from_uv(
                out[u_key].values, out[v_key].values
            )

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
    # Keep tmp_{model} as daily mean for backward compatibility, and add
    # tmax_{model} as daily maximum for true tmax residual targets.
    tmp_col = f"tmp_{model}"
    if tmp_col in df_hourly.columns:
        daily[f"tmax_{model}"] = df_hourly[tmp_col].resample("D").max()
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
                frames.append(
                    canonicalize_hourly(df_raw, model=model, unit_hints=unit_hints)
                )
            except Exception:
                continue

        if not frames:
            continue

        hourly = pd.concat(frames, axis=0).sort_index()
        hourly = hourly.loc[~hourly.index.duplicated(keep="first")]
        daily = hourly_to_daily(hourly, model=model)
        daily.to_parquet(out_file)


def _discover_station_col(
    df: pd.DataFrame, user_col: str | None, candidates: tuple[str, ...], kind: str
) -> str:
    if user_col:
        if user_col not in df.columns:
            raise KeyError(f"{kind} column not found: {user_col}")
        return user_col
    found = _first_present(df.columns, candidates)
    if found is None:
        raise KeyError(f"could not infer {kind} column; pass explicit --{kind}-col")
    return found


def _load_stations(
    stations_csv: str,
    station_id_col: str | None = None,
    lat_col: str | None = None,
    lon_col: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(stations_csv)
    id_col = _discover_station_col(
        df, station_id_col, _STATION_ID_COL_CANDIDATES, "station-id"
    )
    lat_key = _discover_station_col(df, lat_col, _STATION_LAT_COL_CANDIDATES, "lat")
    lon_key = _discover_station_col(df, lon_col, _STATION_LON_COL_CANDIDATES, "lon")
    out = df[[id_col, lat_key, lon_key]].copy()
    out.columns = ["fid", "lat", "lon"]
    out["fid"] = out["fid"].astype(str)
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["fid", "lat", "lon"])
    out = out.drop_duplicates(subset=["fid"]).reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"no valid station rows in {stations_csv}")
    return out


def _parse_day_from_tif_name(path: str) -> pd.Timestamp | None:
    name = os.path.basename(path)
    m = re.search(r"(\d{8})", name)
    if m is None:
        return None
    day = pd.to_datetime(m.group(1), format="%Y%m%d", errors="coerce")
    if pd.isna(day):
        return None
    return day.normalize()


def _iter_daily_tifs(
    tif_root: str, start_date: str | None = None, end_date: str | None = None
) -> list[tuple[pd.Timestamp, str]]:
    start = pd.to_datetime(start_date).normalize() if start_date else None
    end = pd.to_datetime(end_date).normalize() if end_date else None
    rows = []
    for p in sorted(glob.glob(os.path.join(tif_root, "RTMA_*.tif"))):
        day = _parse_day_from_tif_name(p)
        if day is None:
            continue
        if start is not None and day < start:
            continue
        if end is not None and day > end:
            continue
        rows.append((day, p))
    if not rows:
        raise RuntimeError(f"no daily RTMA tifs found in {tif_root}")
    return rows


def _band_index(src: rasterio.io.DatasetReader) -> dict[str, int]:
    names = []
    for i, desc in enumerate(src.descriptions):
        if desc:
            names.append((str(desc).strip().upper(), i))
    if names:
        return {k: i for k, i in names}
    if src.count == len(_RTMA_DAILY_BAND_ORDER):
        return {k: i for i, k in enumerate(_RTMA_DAILY_BAND_ORDER)}
    return {}


def _sample_daily_tif(
    src: rasterio.io.DatasetReader,
    coords: list[tuple[float, float]],
    model: str,
) -> dict[str, np.ndarray]:
    raw = np.asarray(list(src.sample(coords)), dtype="float64")
    if raw.ndim != 2:
        raw = np.atleast_2d(raw)
    nodata = src.nodata
    if nodata is not None:
        raw = np.where(raw == nodata, np.nan, raw)
    raw = np.where(np.isfinite(raw), raw, np.nan)
    idx = _band_index(src)
    model = str(model).lower()

    def _get(name: str) -> np.ndarray:
        i = idx.get(name)
        if i is None and len(_RTMA_DAILY_BAND_ORDER) == raw.shape[1]:
            i = _RTMA_DAILY_BAND_ORDER.index(name)
        if i is None or i >= raw.shape[1]:
            return np.full(raw.shape[0], np.nan, dtype="float64")
        return raw[:, int(i)].astype("float64", copy=False)

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
        wind = np.sqrt(ugrd**2 + vgrd**2)
    if np.isnan(wdir).all():
        wdir = _wind_dir_deg_from_uv(ugrd, vgrd)

    # Convert cloud cover to percent while tolerating fraction-like encodings.
    tcdc = tcdc_scaled.copy()
    finite_tcdc = tcdc[np.isfinite(tcdc)]
    if finite_tcdc.size and float(np.nanmax(finite_tcdc)) <= 1.5:
        tcdc = tcdc * 100.0

    out = {
        f"tmp_{model}": tmp_c,
        f"dpt_{model}": dpt_c,
        f"spfh_{model}": spfh,
        f"pres_{model}": pres_hpa / 10.0,  # PRES appears to be stored in hPa ints
        f"ugrd_{model}": ugrd,
        f"vgrd_{model}": vgrd,
        f"wind_{model}": wind,
        f"wdir_{model}": wdir,
        f"tcdc_{model}": tcdc,
        f"prcp_{model}": prcp_mm,
        f"ea_{model}": _sat_vapor_pressure_kpa_from_dewpoint_c(dpt_c),
    }
    return out


def build_daily_tables_from_tifs(
    stations_csv: str,
    tif_root: str,
    out_daily_root: str,
    model: str = "rtma",
    overwrite: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    station_chunk_size: int = 2000,
    station_id_col: str | None = None,
    lat_col: str | None = None,
    lon_col: str | None = None,
) -> None:
    os.makedirs(out_daily_root, exist_ok=True)
    stations = _load_stations(
        stations_csv=stations_csv,
        station_id_col=station_id_col,
        lat_col=lat_col,
        lon_col=lon_col,
    )
    tif_rows = _iter_daily_tifs(
        tif_root=tif_root, start_date=start_date, end_date=end_date
    )

    n = len(stations)
    step = max(1, int(station_chunk_size))
    for start_i in range(0, n, step):
        stop_i = min(n, start_i + step)
        chunk = stations.iloc[start_i:stop_i].reset_index(drop=True)
        fids = chunk["fid"].tolist()
        coords = list(
            zip(
                chunk["lon"].astype(float).tolist(), chunk["lat"].astype(float).tolist()
            )
        )
        rows_by_fid: dict[str, list[dict[str, float | pd.Timestamp]]] = {
            fid: [] for fid in fids
        }

        for day, tif_path in tif_rows:
            try:
                with rasterio.open(tif_path) as src:
                    decoded = _sample_daily_tif(src, coords=coords, model=model)
            except Exception:
                continue
            for i, fid in enumerate(fids):
                row = {"day": day}
                for c, arr in decoded.items():
                    row[c] = float(arr[i]) if np.isfinite(arr[i]) else np.nan
                rows_by_fid[fid].append(row)

        for fid, rows in rows_by_fid.items():
            if not rows:
                continue
            out_file = os.path.join(out_daily_root, f"{fid}.parquet")
            new_df = pd.DataFrame.from_records(rows).set_index("day").sort_index()
            new_df = new_df.loc[~new_df.index.duplicated(keep="last")]
            if os.path.exists(out_file) and not overwrite:
                try:
                    old = pd.read_parquet(out_file)
                    old.index = pd.to_datetime(old.index, errors="coerce")
                    old = old[old.index.notna()]
                    old.index = old.index.normalize()
                    merged = pd.concat([old, new_df], axis=0).sort_index()
                    merged = merged.loc[~merged.index.duplicated(keep="last")]
                    merged.to_parquet(out_file)
                except Exception:
                    new_df.to_parquet(out_file)
            else:
                new_df.to_parquet(out_file)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build daily RTMA/URMA per-station Parquets from hourly extracts or daily COGs."
    )
    p.add_argument(
        "--source",
        choices=["hourly", "tif"],
        default="hourly",
        help="Input source type.",
    )
    p.add_argument(
        "--raw-station-root",
        default=None,
        help="Root dir containing per-station subdirs of monthly Parquets (hourly mode).",
    )
    p.add_argument(
        "--stations",
        default=None,
        help="Station CSV with fid/lat/lon columns (tif mode).",
    )
    p.add_argument(
        "--tif-root",
        default=None,
        help="Directory of daily COGs named RTMA_YYYYMMDD.tif (tif mode).",
    )
    p.add_argument(
        "--out-daily-root",
        required=True,
        help="Output directory for per-station daily Parquets.",
    )
    p.add_argument(
        "--model",
        default="rtma",
        choices=["rtma", "urma"],
        help="Model name for output suffixing.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output station files.",
    )
    p.add_argument(
        "--start-date",
        default=None,
        help="Optional start date (YYYY-MM-DD) for tif mode.",
    )
    p.add_argument(
        "--end-date", default=None, help="Optional end date (YYYY-MM-DD) for tif mode."
    )
    p.add_argument(
        "--station-chunk-size",
        type=int,
        default=2000,
        help="Stations per chunk while sampling tifs.",
    )
    p.add_argument(
        "--station-id-col", default=None, help="Station id column in --stations CSV."
    )
    p.add_argument("--lat-col", default=None, help="Latitude column in --stations CSV.")
    p.add_argument(
        "--lon-col", default=None, help="Longitude column in --stations CSV."
    )
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
    if a.source == "hourly":
        if not a.raw_station_root:
            raise ValueError("--raw-station-root is required when --source hourly")
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
        return

    if not a.stations:
        raise ValueError("--stations is required when --source tif")
    if not a.tif_root:
        raise ValueError("--tif-root is required when --source tif")
    build_daily_tables_from_tifs(
        stations_csv=a.stations,
        tif_root=a.tif_root,
        out_daily_root=a.out_daily_root,
        model=a.model,
        overwrite=bool(a.overwrite),
        start_date=a.start_date,
        end_date=a.end_date,
        station_chunk_size=int(a.station_chunk_size),
        station_id_col=a.station_id_col,
        lat_col=a.lat_col,
        lon_col=a.lon_col,
    )


if __name__ == "__main__":
    main()
