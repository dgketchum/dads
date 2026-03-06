"""
Build a train-ready station-day patch index for RTMA bias correction.

Inputs
------
- station-day residual table (Parquet keyed by (fid, day)) from prep/build_station_day_table.py
- station metadata CSV with fid, latitude, longitude (e.g., MADIS station inventory)

Output
------
Parquet with one row per station-day suitable for patch sampling:
- fid, day, latitude, longitude
- delta_log_ea (required for humidity MVP)

This index is intentionally small; it keeps patch I/O on-the-fly so iteration is fast.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

QC_PROFILE_TO_LISTS: dict[str, list[str]] = {
    # Canonical non-wind operational profiles
    "t_rtma": [
        "rtma_t_rejectlist",
        "rtma_t_day_rejectlist",
        "rtma_t_night_rejectlist",
    ],
    "t_urma": [
        "urma2p5_t_day_rejectlist",
        "urma2p5_t_night_rejectlist",
    ],
    "ea_rtma": [
        "rtma_q_rejectlist",
        "rtma_q_day_rejectlist",
        "rtma_q_night_rejectlist",
    ],
    "ea_urma": [
        "urma2p5_q_day_rejectlist",
        "urma2p5_q_night_rejectlist",
    ],
    # Aliases
    "q_rtma": [
        "rtma_q_rejectlist",
        "rtma_q_day_rejectlist",
        "rtma_q_night_rejectlist",
    ],
    "q_urma": [
        "urma2p5_q_day_rejectlist",
        "urma2p5_q_night_rejectlist",
    ],
    "temp_rtma": [
        "rtma_t_rejectlist",
        "rtma_t_day_rejectlist",
        "rtma_t_night_rejectlist",
    ],
    "temp_urma": [
        "urma2p5_t_day_rejectlist",
        "urma2p5_t_night_rejectlist",
    ],
    "humidity_rtma": [
        "rtma_q_rejectlist",
        "rtma_q_day_rejectlist",
        "rtma_q_night_rejectlist",
    ],
    "humidity_urma": [
        "urma2p5_q_day_rejectlist",
        "urma2p5_q_night_rejectlist",
    ],
}


def _load_gsi_reject_set(paths: list[str]) -> set[str]:
    """Parse GSI-fix reject-list files and return a set of station IDs.

    Format: 3 header lines, then ``'STAID   | itype=...`` per line.
    Station ID is at ``line[1:9].strip()``.
    """
    ids: set[str] = set()
    for p in paths:
        p = str(Path(p).expanduser())
        with open(p) as fh:
            lines = fh.readlines()
        for line in lines[3:]:
            line = line.strip()
            if len(line) < 9:
                continue
            sid = line[1:9].strip()
            if sid:
                ids.add(sid)
    return ids


def _resolve_qc_profile_paths(
    profiles: list[str] | None, gsi_fix_dir: str | None
) -> list[str]:
    """Resolve named QC profiles to concrete GSI-fix reject-list paths."""
    if not profiles:
        return []

    if not gsi_fix_dir:
        raise ValueError("--qc-profile requires --gsi-fix-dir")

    root = Path(gsi_fix_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"--gsi-fix-dir does not exist: {root}")

    resolved: list[str] = []
    for raw in profiles:
        key = str(raw).strip().lower()
        if not key:
            continue
        if key not in QC_PROFILE_TO_LISTS:
            known = ", ".join(sorted(QC_PROFILE_TO_LISTS))
            raise ValueError(
                f"Unknown --qc-profile '{raw}'. Supported profiles: {known}"
            )
        for fn in QC_PROFILE_TO_LISTS[key]:
            p = root / fn
            if not p.exists():
                raise FileNotFoundError(f"QC profile '{key}' expects missing file: {p}")
            resolved.append(str(p))

    # Preserve order while de-duplicating.
    uniq: list[str] = []
    seen: set[str] = set()
    for p in resolved:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _read_station_day_table(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.MultiIndex) and df.index.nlevels >= 2:
        # Normalize MultiIndex names and day type.
        names = list(df.index.names[:2])
        if names != ["fid", "day"]:
            df.index = df.index.set_names(["fid", "day"] + list(df.index.names[2:]))
        fid = df.index.get_level_values(0).astype(str)
        day = pd.to_datetime(df.index.get_level_values(1), errors="coerce").normalize()
        df = df.copy()
        df.index = pd.MultiIndex.from_arrays([fid, day], names=["fid", "day"])
        df = df.sort_index()
        return df

    # Accept flat columns too, but normalize to (fid, day).
    if "fid" in df.columns and "day" in df.columns:
        df["fid"] = df["fid"].astype(str)
        df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["day"])
        df = df.set_index(["fid", "day"]).sort_index()
    else:
        raise ValueError(
            "station-day table must be keyed by (fid, day) or contain fid/day columns"
        )
    return df


def build_patch_index(
    station_day_parquet: str,
    stations_csv: str,
    out_parquet: str,
    min_ea_kpa: float = 1e-4,
    max_ea_kpa: float = 8.0,
    max_tmax_c: float = 60.0,
    min_tmax_c: float = -50.0,
    max_abs_delta: float | None = 3.0,
    max_abs_delta_tmax: float | None = 30.0,
    max_abs_delta_tmin: float | None = 30.0,
    max_abs_delta_wind: float | None = 15.0,
    require_tif_root: str | None = None,
    station_id_col: str = "fid",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    reject_lists: list[str] | None = None,
    target_cols: list[str] | None = None,
) -> str:
    if target_cols is None:
        target_cols = ["delta_log_ea"]
    sdf = _read_station_day_table(station_day_parquet)

    # Identify ea baseline column: prefer urma, fall back to rtma.
    ea_base_col = None
    for cand in ("ea_urma", "ea_rtma"):
        if cand in sdf.columns:
            ea_base_col = cand
            break

    delta_ea_col = None
    for cand in ("delta_ea_urma", "delta_ea_rtma"):
        if cand in sdf.columns:
            delta_ea_col = cand
            break

    # Determine which source columns we need.
    need: list[str] = []
    if "y_obs" in sdf.columns:
        need.append("y_obs")
    if "ea" in sdf.columns:
        need.append("ea")
    if ea_base_col:
        need.append(ea_base_col)
    if delta_ea_col:
        need.append(delta_ea_col)
    if "delta_tmax" in target_cols:
        if "tmax" in sdf.columns:
            need.append("tmax")
        for cand in ("tmax_urma", "tmax_rtma"):
            if cand in sdf.columns:
                need.append(cand)
        if "tmax_baseline_source" in sdf.columns:
            need.append("tmax_baseline_source")
        for cand in ("tmp_urma", "tmp_rtma"):
            if cand in sdf.columns:
                need.append(cand)
        if "delta_tmax" in sdf.columns:
            need.append("delta_tmax")
        has_tmax_base = any(c in sdf.columns for c in ("tmax_urma", "tmax_rtma"))
        if not has_tmax_base:
            raise ValueError(
                "tmax target requested but station-day table has neither "
                "'tmax_urma' nor 'tmax_rtma'. Rebuild daily artifacts first."
            )

    if "delta_tmin" in target_cols:
        if "tmin" in sdf.columns:
            need.append("tmin")
        for cand in ("tmin_urma", "tmin_rtma"):
            if cand in sdf.columns:
                need.append(cand)
        if "delta_tmin" in sdf.columns:
            need.append("delta_tmin")
        has_tmin_base = any(c in sdf.columns for c in ("tmin_urma", "tmin_rtma"))
        if not has_tmin_base:
            raise ValueError(
                "tmin target requested but station-day table has neither "
                "'tmin_urma' nor 'tmin_rtma'. Rebuild daily artifacts first."
            )

    if "delta_wind" in target_cols:
        if "wind" in sdf.columns:
            need.append("wind")
        for cand in ("wind_urma", "wind_rtma"):
            if cand in sdf.columns:
                need.append(cand)
        if "delta_wind" in sdf.columns:
            need.append("delta_wind")
        has_wind_base = any(c in sdf.columns for c in ("wind_urma", "wind_rtma"))
        if not has_wind_base:
            raise ValueError(
                "wind target requested but station-day table has neither "
                "'wind_urma' nor 'wind_rtma'. Rebuild daily artifacts first."
            )

    # Must have ea data one way or another
    has_ea = (
        "y_obs" in need or "ea" in need or delta_ea_col in need
    ) and ea_base_col in need
    if not has_ea:
        raise ValueError(
            "expected columns: (y_obs|ea, ea_urma|ea_rtma) or "
            "(delta_ea_urma|delta_ea_rtma, ea_urma|ea_rtma)"
        )

    sub = sdf[[c for c in need if c in sdf.columns]].copy()
    sub = sub.reset_index()

    stations = pd.read_csv(stations_csv)
    if station_id_col not in stations.columns:
        raise ValueError(f"stations_csv must include '{station_id_col}' column")
    if lat_col not in stations.columns or lon_col not in stations.columns:
        raise ValueError(
            f"stations_csv must include '{lat_col}' and '{lon_col}' columns"
        )
    stations[station_id_col] = stations[station_id_col].astype(str)

    merge_cols = [station_id_col, lat_col, lon_col]
    if "MGRS_TILE" in stations.columns:
        merge_cols.append("MGRS_TILE")

    rename_map = {station_id_col: "fid", lat_col: "latitude", lon_col: "longitude"}
    merged = sub.merge(
        stations[merge_cols].rename(columns=rename_map),
        how="left",
        on="fid",
    )
    merged = merged.dropna(subset=["latitude", "longitude"])

    # GSI reject-list filtering
    if reject_lists:
        reject_ids = _load_gsi_reject_set(reject_lists)
        n_before = len(merged)
        n_stations_before = merged["fid"].nunique()
        merged = merged.loc[~merged["fid"].isin(reject_ids)].copy()
        n_drop_rows = n_before - len(merged)
        n_drop_sta = n_stations_before - merged["fid"].nunique()
        print(
            f"  reject-list filter: dropped {n_drop_sta:,} stations, "
            f"{n_drop_rows:,} rows ({n_drop_rows / n_before * 100:.1f}%)"
        )

    # --- Physical bounds on raw obs (before computing residuals) ---
    if "ea" in merged.columns:
        ea_raw = pd.to_numeric(merged["ea"], errors="coerce")
        bad_ea = ~ea_raw.between(min_ea_kpa, max_ea_kpa)
        merged.loc[bad_ea, "ea"] = np.nan
    if "y_obs" in merged.columns:
        yo_raw = pd.to_numeric(merged["y_obs"], errors="coerce")
        bad_yo = ~yo_raw.between(min_ea_kpa, max_ea_kpa)
        merged.loc[bad_yo, "y_obs"] = np.nan
    if "tmax" in merged.columns:
        tmax_raw = pd.to_numeric(merged["tmax"], errors="coerce")
        bad_tmax = ~tmax_raw.between(min_tmax_c, max_tmax_c)
        merged.loc[bad_tmax, "tmax"] = np.nan
    if "tmin" in merged.columns:
        tmin_raw = pd.to_numeric(merged["tmin"], errors="coerce")
        bad_tmin = ~tmin_raw.between(-50.0, 60.0)
        merged.loc[bad_tmin, "tmin"] = np.nan
    if "wind" in merged.columns:
        wind_raw = pd.to_numeric(merged["wind"], errors="coerce")
        bad_wind = ~wind_raw.between(0.0, 35.0)
        merged.loc[bad_wind, "wind"] = np.nan
    n_bounds = (
        int(bad_ea.sum() if "ea" in merged.columns else 0)
        + int(bad_tmax.sum() if "tmax" in merged.columns else 0)
        + int(bad_tmin.sum() if "tmin" in merged.columns else 0)
        + int(bad_wind.sum() if "wind" in merged.columns else 0)
    )
    if n_bounds:
        print(f"  physical bounds: NaN'd {n_bounds:,} obs values outside limits")

    # --- Compute ea residuals ---
    if "ea" in merged.columns:
        ea_obs = pd.to_numeric(merged["ea"], errors="coerce")
    elif "y_obs" in merged.columns:
        ea_obs = pd.to_numeric(merged["y_obs"], errors="coerce")
    else:
        ea_obs = pd.to_numeric(merged[delta_ea_col], errors="coerce") + pd.to_numeric(
            merged[ea_base_col], errors="coerce"
        )
    ea_base = pd.to_numeric(merged[ea_base_col], errors="coerce")

    merged = merged.copy()
    merged["ea_obs"] = ea_obs
    merged["ea_base_num"] = ea_base

    # ea validity mask (required for delta_log_ea rows)
    ea_valid = (
        (merged["ea_obs"] > float(min_ea_kpa))
        & (merged["ea_base_num"] > float(min_ea_kpa))
        & np.isfinite(merged["ea_obs"])
        & np.isfinite(merged["ea_base_num"])
    )
    merged["delta_log_ea"] = np.where(
        ea_valid,
        np.log(np.maximum(merged["ea_obs"], 1e-10))
        - np.log(np.maximum(merged["ea_base_num"], 1e-10)),
        np.nan,
    )
    merged["log_ea_obs"] = np.where(
        ea_valid, np.log(np.maximum(merged["ea_obs"], 1e-10)), np.nan
    )
    merged["ea_rtma"] = np.where(ea_valid, merged["ea_base_num"], np.nan)

    # ea outlier filter
    if max_abs_delta is not None:
        ea_outlier = merged["delta_log_ea"].abs() > float(max_abs_delta)
        n_drop = ea_outlier.sum()
        if n_drop:
            print(
                f"  ea outlier filter: setting {n_drop:,} rows with |delta_log_ea| > {max_abs_delta} to NaN"
            )
        merged.loc[ea_outlier, ["delta_log_ea", "log_ea_obs", "ea_rtma"]] = np.nan

    # --- Compute tmax residuals (if requested) ---
    if "delta_tmax" in target_cols:
        if "delta_tmax" in merged.columns:
            merged["delta_tmax"] = pd.to_numeric(merged["delta_tmax"], errors="coerce")
            if "tmax_baseline_source" not in merged.columns:
                raise ValueError(
                    "station-day table has delta_tmax but missing tmax_baseline_source. "
                    "Rebuild station-day table with strict tmax provenance."
                )
            has_delta = merged["delta_tmax"].notna()
            src = merged["tmax_baseline_source"]
            missing_src = int((has_delta & src.isna()).sum())
            if missing_src:
                raise ValueError(
                    f"Found {missing_src:,} rows with delta_tmax but missing "
                    "tmax_baseline_source."
                )
            allowed_sources = {"tmax_rtma", "tmax_urma"}
            src_vals = src.loc[has_delta].dropna().astype(str).unique().tolist()
            bad = set(src_vals) - allowed_sources
            if bad:
                raise ValueError(
                    f"Unexpected tmax baseline sources: {sorted(bad)}. "
                    f"Allowed: {sorted(allowed_sources)}."
                )
        elif "tmax" in merged.columns:
            tmax_base_col = None
            for cand in ("tmax_urma", "tmax_rtma"):
                if cand in merged.columns:
                    tmax_base_col = cand
                    break
            if tmax_base_col is None:
                raise ValueError(
                    "Unable to construct delta_tmax: no tmax baseline column found."
                )
            tmax_obs = pd.to_numeric(merged["tmax"], errors="coerce")
            tmax_base = pd.to_numeric(merged[tmax_base_col], errors="coerce")
            merged["delta_tmax"] = tmax_obs - tmax_base
            merged["tmax_baseline_source"] = pd.Series(pd.NA, index=merged.index)
            has_delta = merged["delta_tmax"].notna()
            merged.loc[has_delta, "tmax_baseline_source"] = tmax_base_col
        else:
            raise ValueError(
                "Unable to construct delta_tmax: expected either precomputed delta_tmax "
                "+ tmax_baseline_source or raw (tmax, tmax_rtma) columns."
            )

        for col in ("tmax_urma", "tmax_rtma", "tmp_urma", "tmp_rtma"):
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")

        if max_abs_delta_tmax is not None:
            tmax_outlier = merged["delta_tmax"].abs() > float(max_abs_delta_tmax)
            n_drop = tmax_outlier.sum()
            if n_drop:
                print(
                    f"  tmax outlier filter: setting {n_drop:,} rows with |delta_tmax| > {max_abs_delta_tmax} to NaN"
                )
            merged.loc[tmax_outlier, "delta_tmax"] = np.nan

    # --- Compute tmin residuals (if requested) ---
    if "delta_tmin" in target_cols:
        if "delta_tmin" in merged.columns:
            merged["delta_tmin"] = pd.to_numeric(merged["delta_tmin"], errors="coerce")
        elif "tmin" in merged.columns:
            tmin_base_col = None
            for cand in ("tmin_urma", "tmin_rtma"):
                if cand in merged.columns:
                    tmin_base_col = cand
                    break
            if tmin_base_col is None:
                raise ValueError(
                    "Unable to construct delta_tmin: no tmin baseline column found."
                )
            tmin_obs = pd.to_numeric(merged["tmin"], errors="coerce")
            tmin_base = pd.to_numeric(merged[tmin_base_col], errors="coerce")
            merged["delta_tmin"] = tmin_obs - tmin_base
        else:
            raise ValueError(
                "Unable to construct delta_tmin: expected either precomputed delta_tmin "
                "or raw (tmin, tmin_urma) columns."
            )

        if max_abs_delta_tmin is not None:
            tmin_outlier = merged["delta_tmin"].abs() > float(max_abs_delta_tmin)
            n_drop = tmin_outlier.sum()
            if n_drop:
                print(
                    f"  tmin outlier filter: setting {n_drop:,} rows with |delta_tmin| > {max_abs_delta_tmin} to NaN"
                )
            merged.loc[tmin_outlier, "delta_tmin"] = np.nan

    # --- Compute wind residuals (if requested) ---
    if "delta_wind" in target_cols:
        if "delta_wind" in merged.columns:
            merged["delta_wind"] = pd.to_numeric(merged["delta_wind"], errors="coerce")
        elif "wind" in merged.columns:
            wind_base_col = None
            for cand in ("wind_urma", "wind_rtma"):
                if cand in merged.columns:
                    wind_base_col = cand
                    break
            if wind_base_col is None:
                raise ValueError(
                    "Unable to construct delta_wind: no wind baseline column found."
                )
            wind_obs = pd.to_numeric(merged["wind"], errors="coerce")
            wind_base = pd.to_numeric(merged[wind_base_col], errors="coerce")
            merged["delta_wind"] = wind_obs - wind_base
        else:
            raise ValueError(
                "Unable to construct delta_wind: expected either precomputed delta_wind "
                "or raw (wind, wind_urma) columns."
            )

        if max_abs_delta_wind is not None:
            wind_outlier = merged["delta_wind"].abs() > float(max_abs_delta_wind)
            n_drop = wind_outlier.sum()
            if n_drop:
                print(
                    f"  wind outlier filter: setting {n_drop:,} rows with |delta_wind| > {max_abs_delta_wind} to NaN"
                )
            merged.loc[wind_outlier, "delta_wind"] = np.nan

    merged["day"] = pd.to_datetime(merged["day"], errors="coerce").dt.normalize()
    merged = merged.dropna(subset=["day"])

    # Keep rows with at least one valid target
    merged = merged.dropna(subset=target_cols, how="all")

    # Optional filter: only keep rows that have a corresponding TIF on disk.
    if require_tif_root:
        root = str(require_tif_root).rstrip("/")
        # Detect naming convention from first .tif file in directory.
        first_files = [f for f in sorted(os.listdir(root)) if f.endswith(".tif")][:1]
        if first_files and first_files[0].startswith("URMA_1km_"):
            tif_fmt = "URMA_1km_{}.tif"
        elif first_files and first_files[0].startswith("RTMA_1km_"):
            tif_fmt = "RTMA_1km_{}.tif"
        else:
            tif_fmt = "RTMA_{}.tif"
        ymd = merged["day"].dt.strftime("%Y%m%d")
        paths = ymd.map(lambda s: os.path.join(root, tif_fmt.format(s)))
        exists = paths.map(os.path.exists)
        merged = merged.loc[exists.values].copy()

    out_cols = [
        "fid",
        "day",
        "latitude",
        "longitude",
        "delta_log_ea",
        "log_ea_obs",
        "ea_rtma",
    ]
    if "delta_tmax" in target_cols:
        out_cols.append("delta_tmax")
        if "tmax_baseline_source" in merged.columns:
            out_cols.append("tmax_baseline_source")
        for col in ("tmax_urma", "tmax_rtma", "tmp_urma", "tmp_rtma"):
            if col in merged.columns:
                out_cols.append(col)
    if "delta_tmin" in target_cols:
        out_cols.append("delta_tmin")
    if "delta_wind" in target_cols:
        out_cols.append("delta_wind")
    if "MGRS_TILE" in merged.columns:
        out_cols.append("MGRS_TILE")

    out = merged[out_cols].copy()
    os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
    out.to_parquet(out_parquet, index=False)

    # Summary stats
    n_total = len(out)
    n_ea = out["delta_log_ea"].notna().sum()
    parts = [f"{n_total:,} rows", f"ea={n_ea:,}"]
    for tcol in target_cols:
        if tcol != "delta_log_ea" and tcol in out.columns:
            parts.append(f"{tcol}={out[tcol].notna().sum():,}")
    print(f"  patch index: {', '.join(parts)}")

    return out_parquet


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a station-day patch index for RTMA correction training."
    )
    p.add_argument(
        "--station-day",
        required=True,
        help="Parquet keyed by (fid, day) from build_station_day_table.",
    )
    p.add_argument(
        "--stations-csv",
        required=True,
        help="CSV with columns fid, latitude, longitude.",
    )
    p.add_argument("--out", required=True, help="Output Parquet path.")
    p.add_argument(
        "--min-ea-kpa",
        type=float,
        default=1e-4,
        help="Minimum ea threshold to allow log residuals.",
    )
    p.add_argument(
        "--max-ea-kpa",
        type=float,
        default=8.0,
        help="Physical upper bound for observed ea (kPa). Default 8.0.",
    )
    p.add_argument(
        "--max-tmax-c",
        type=float,
        default=60.0,
        help="Physical upper bound for observed tmax (C). Default 60.0.",
    )
    p.add_argument(
        "--min-tmax-c",
        type=float,
        default=-50.0,
        help="Physical lower bound for observed tmax (C). Default -50.0.",
    )
    p.add_argument(
        "--max-abs-delta",
        type=float,
        default=3.0,
        help="Drop rows with |delta_log_ea| above this threshold (default 3.0).",
    )
    p.add_argument(
        "--require-tif-root",
        default=None,
        help="If set, filter rows to those with RTMA_YYYYMMDD.tif present.",
    )
    p.add_argument(
        "--station-id-col",
        default="fid",
        help="Station id column in --stations-csv (default fid).",
    )
    p.add_argument(
        "--lat-col", default="latitude", help="Latitude column in --stations-csv."
    )
    p.add_argument(
        "--lon-col", default="longitude", help="Longitude column in --stations-csv."
    )
    p.add_argument(
        "--reject-list",
        nargs="*",
        default=None,
        help="GSI-fix reject-list file(s). Stations in these lists are dropped.",
    )
    p.add_argument(
        "--qc-profile",
        action="append",
        default=None,
        help=(
            "Named reject-list profile(s). Repeat flag or use comma-separated values. "
            "Examples: ea_rtma, ea_urma, t_rtma, t_urma."
        ),
    )
    p.add_argument(
        "--gsi-fix-dir",
        default=None,
        help="Path to GSI-fix directory used to resolve --qc-profile list files.",
    )
    p.add_argument(
        "--target-cols",
        default=None,
        help="Comma-separated target columns (default: delta_log_ea). "
        "Use delta_log_ea,delta_tmax for multi-task.",
    )
    p.add_argument(
        "--max-abs-delta-tmax",
        type=float,
        default=30.0,
        help="Drop rows with |delta_tmax| above this threshold (default 30.0).",
    )
    p.add_argument(
        "--max-abs-delta-tmin",
        type=float,
        default=30.0,
        help="Drop rows with |delta_tmin| above this threshold (default 30.0).",
    )
    p.add_argument(
        "--max-abs-delta-wind",
        type=float,
        default=15.0,
        help="Drop rows with |delta_wind| above this threshold (default 15.0).",
    )
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    target_cols = None
    if a.target_cols:
        target_cols = [c.strip() for c in a.target_cols.split(",")]

    profiles: list[str] = []
    if a.qc_profile:
        for item in a.qc_profile:
            for tok in str(item).split(","):
                tok = tok.strip()
                if tok:
                    profiles.append(tok)
    profile_lists = _resolve_qc_profile_paths(profiles, a.gsi_fix_dir)

    reject_lists: list[str] = []
    if a.reject_list:
        reject_lists.extend(a.reject_list)
    reject_lists.extend(profile_lists)

    if reject_lists:
        # Keep order, normalize paths, de-duplicate.
        norm: list[str] = []
        seen: set[str] = set()
        for p in reject_lists:
            path = str(Path(p).expanduser())
            if path in seen:
                continue
            seen.add(path)
            norm.append(path)
        reject_lists = norm
        if profiles:
            print(f"  qc profiles: {', '.join(profiles)}")
        print(f"  reject-list files: {len(reject_lists)}")
    else:
        reject_lists = []

    build_patch_index(
        station_day_parquet=a.station_day,
        stations_csv=a.stations_csv,
        out_parquet=a.out,
        min_ea_kpa=float(a.min_ea_kpa),
        max_ea_kpa=float(a.max_ea_kpa),
        max_tmax_c=float(a.max_tmax_c),
        min_tmax_c=float(a.min_tmax_c),
        max_abs_delta=float(a.max_abs_delta) if a.max_abs_delta else None,
        max_abs_delta_tmax=float(a.max_abs_delta_tmax)
        if a.max_abs_delta_tmax
        else None,
        max_abs_delta_tmin=float(a.max_abs_delta_tmin)
        if a.max_abs_delta_tmin
        else None,
        max_abs_delta_wind=float(a.max_abs_delta_wind)
        if a.max_abs_delta_wind
        else None,
        require_tif_root=a.require_tif_root,
        station_id_col=str(a.station_id_col),
        lat_col=str(a.lat_col),
        lon_col=str(a.lon_col),
        reject_lists=reject_lists or None,
        target_cols=target_cols,
    )


if __name__ == "__main__":
    main()
