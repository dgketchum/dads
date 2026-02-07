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

import numpy as np
import pandas as pd


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
        raise ValueError("station-day table must be keyed by (fid, day) or contain fid/day columns")
    return df


def build_patch_index(
    station_day_parquet: str,
    stations_csv: str,
    out_parquet: str,
    min_ea_kpa: float = 1e-4,
    require_tif_root: str | None = None,
    station_id_col: str = "fid",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> str:
    sdf = _read_station_day_table(station_day_parquet)
    # Keep only humidity MVP columns.
    need = []
    if "y_obs" in sdf.columns and "ea_rtma" in sdf.columns:
        need = ["y_obs", "ea_rtma"]
    elif "delta_ea_rtma" in sdf.columns and "ea_rtma" in sdf.columns:
        need = ["delta_ea_rtma", "ea_rtma"]
    else:
        raise ValueError("expected columns: (y_obs, ea_rtma) or (delta_ea_rtma, ea_rtma)")

    sub = sdf[need].copy()
    sub = sub.reset_index()

    stations = pd.read_csv(stations_csv)
    if station_id_col not in stations.columns:
        raise ValueError(f"stations_csv must include '{station_id_col}' column")
    if lat_col not in stations.columns or lon_col not in stations.columns:
        raise ValueError(f"stations_csv must include '{lat_col}' and '{lon_col}' columns")
    stations[station_id_col] = stations[station_id_col].astype(str)

    merged = sub.merge(
        stations[[station_id_col, lat_col, lon_col]].rename(
            columns={station_id_col: "fid", lat_col: "latitude", lon_col: "longitude"}
        ),
        how="left",
        on="fid",
    )
    merged = merged.dropna(subset=["latitude", "longitude"])

    # Compute delta_log_ea.
    if "y_obs" in merged.columns:
        ea_obs = pd.to_numeric(merged["y_obs"], errors="coerce")
    else:
        # delta_ea_rtma + ea_rtma
        ea_obs = pd.to_numeric(merged["delta_ea_rtma"], errors="coerce") + pd.to_numeric(
            merged["ea_rtma"], errors="coerce"
        )
    ea_rtma = pd.to_numeric(merged["ea_rtma"], errors="coerce")

    # Keep a numeric ea_obs column so any filtering happens before log() to avoid warnings.
    merged = merged.copy()
    merged["ea_obs"] = ea_obs

    m = (
        (merged["ea_obs"] > float(min_ea_kpa))
        & (ea_rtma > float(min_ea_kpa))
        & np.isfinite(merged["ea_obs"])
        & np.isfinite(ea_rtma)
    )
    merged = merged.loc[m].copy()
    merged["delta_log_ea"] = np.log(merged["ea_obs"]) - np.log(merged["ea_rtma"])

    merged["day"] = pd.to_datetime(merged["day"], errors="coerce").dt.normalize()
    merged = merged.dropna(subset=["day"])

    # Optional filter: only keep rows that have a corresponding RTMA tif on disk.
    if require_tif_root:
        root = str(require_tif_root).rstrip("/")
        # Convert day to yyyymmdd
        ymd = merged["day"].dt.strftime("%Y%m%d")
        paths = ymd.map(lambda s: os.path.join(root, f"RTMA_{s}.tif"))
        exists = paths.map(os.path.exists)
        merged = merged.loc[exists.values].copy()

    out = merged[["fid", "day", "latitude", "longitude", "delta_log_ea"]].copy()
    os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
    out.to_parquet(out_parquet, index=False)
    return out_parquet


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a station-day patch index for RTMA correction training.")
    p.add_argument("--station-day", required=True, help="Parquet keyed by (fid, day) from build_station_day_table.")
    p.add_argument("--stations-csv", required=True, help="CSV with columns fid, latitude, longitude.")
    p.add_argument("--out", required=True, help="Output Parquet path.")
    p.add_argument("--min-ea-kpa", type=float, default=1e-4, help="Minimum ea threshold to allow log residuals.")
    p.add_argument("--require-tif-root", default=None, help="If set, filter rows to those with RTMA_YYYYMMDD.tif present.")
    p.add_argument("--station-id-col", default="fid", help="Station id column in --stations-csv (default fid).")
    p.add_argument("--lat-col", default="latitude", help="Latitude column in --stations-csv.")
    p.add_argument("--lon-col", default="longitude", help="Longitude column in --stations-csv.")
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    build_patch_index(
        station_day_parquet=a.station_day,
        stations_csv=a.stations_csv,
        out_parquet=a.out,
        min_ea_kpa=float(a.min_ea_kpa),
        require_tif_root=a.require_tif_root,
        station_id_col=str(a.station_id_col),
        lat_col=str(a.lat_col),
        lon_col=str(a.lon_col),
    )


if __name__ == "__main__":
    main()
