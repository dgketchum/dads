"""
Build a single station-day table for MVP experiments.

This is a *fast tabular* artifact used to:
- verify RTMA/URMA update coverage quickly
- enforce leakage-safe feature masks deterministically
- drive simple baselines (RTMA vs corrected) without depending on per-station Parquet conventions

The output is a single Parquet keyed by (fid, day) with one row per station-day.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Inputs:
    obs_dir: str
    obs_col: str
    rtma_daily_dir: str | None = None
    urma_daily_dir: str | None = None
    stations_csv: str | None = None
    station_id_col: str = "fid"


def _read_station_daily_parquet(path: str, day_mode: str = "local") -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        # common pattern: a date column exists but index is 0..N
        for c in ("time", "date", "day", "dt"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
                df = df.set_index(c)
                break
    idx = pd.to_datetime(df.index, errors="coerce")
    df = df[idx.notna()].copy()
    idx = idx[idx.notna()]
    df.index = idx
    df = df.sort_index()

    mode = str(day_mode or "local").lower().strip()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        # MADIS daily files are typically local-day aggregates with tz-aware
        # indexes at local midnight. For joining to UTC-day RTMA, we provide a
        # midpoint mapping option that maps each local day to the UTC day that
        # contains its midpoint (local noon).
        local_day = df.index.normalize()
        if mode == "utc_midpoint":
            idx2 = (local_day + pd.Timedelta(hours=12)).tz_convert("UTC").normalize().tz_localize(None)
        elif mode == "utc_start":
            idx2 = local_day.tz_convert("UTC").normalize().tz_localize(None)
        elif mode == "local":
            idx2 = local_day.tz_localize(None)
        else:
            raise ValueError("day_mode must be one of: local, utc_midpoint, utc_start")
        df = df.copy()
        df.index = idx2
    else:
        # Naive timestamps: treat as already day-aligned.
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce").normalize()
    return df


def build_station_day_table(
    inputs: Inputs,
    out_file: str,
    leakage_safe: bool = True,
    overwrite: bool = False,
    obs_day_mode: str = "local",
) -> str:
    if os.path.exists(out_file) and not overwrite:
        return out_file

    if not os.path.isdir(inputs.obs_dir):
        raise FileNotFoundError(f"obs_dir not found: {inputs.obs_dir}")

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    allowed_fids: set[str] | None = None
    if inputs.stations_csv:
        stations = pd.read_csv(inputs.stations_csv)
        if inputs.station_id_col not in stations.columns:
            raise KeyError(f"station_id_col not found in stations_csv: {inputs.station_id_col}")
        allowed_fids = set(stations[inputs.station_id_col].astype(str).tolist())
        if not allowed_fids:
            raise RuntimeError("stations_csv produced empty fid set")

    rows = []
    for fn in sorted(os.listdir(inputs.obs_dir)):
        if not fn.endswith(".parquet"):
            continue
        fid = os.path.splitext(fn)[0]
        if allowed_fids is not None and fid not in allowed_fids:
            continue
        obs_path = os.path.join(inputs.obs_dir, fn)
        obs = _read_station_daily_parquet(obs_path, day_mode=obs_day_mode)
        if inputs.obs_col not in obs.columns:
            continue
        obs = obs[[inputs.obs_col]].rename(columns={inputs.obs_col: "y_obs"})

        merged = obs
        if inputs.rtma_daily_dir:
            p = os.path.join(inputs.rtma_daily_dir, f"{fid}.parquet")
            if os.path.exists(p):
                rtma = _read_station_daily_parquet(p)
                merged = merged.join(rtma, how="left")
        if inputs.urma_daily_dir:
            p = os.path.join(inputs.urma_daily_dir, f"{fid}.parquet")
            if os.path.exists(p):
                urma = _read_station_daily_parquet(p)
                merged = merged.join(urma, how="left")

        # Useful MVP targets: residuals against RTMA/URMA baselines (when present).
        if "ea_rtma" in merged.columns:
            merged["delta_ea_rtma"] = merged["y_obs"] - merged["ea_rtma"]
        if "ea_urma" in merged.columns:
            merged["delta_ea_urma"] = merged["y_obs"] - merged["ea_urma"]

        # optional leakage mask for ea MVP: keep ea_{model} baseline, but drop other humidity-proxy fields
        if leakage_safe:
            drop_substrings = ("spfh_", "dpt_", "rh_", "q_", "sh_", "dew")
            keep_cols = []
            for c in merged.columns:
                if c in ("y_obs", "ea_rtma", "ea_urma"):
                    keep_cols.append(c)
                    continue
                if any(s in c.lower() for s in drop_substrings):
                    continue
                keep_cols.append(c)
            merged = merged[keep_cols]

        merged = merged.copy()
        merged["fid"] = fid
        merged["day"] = merged.index.normalize()
        merged.reset_index(drop=True, inplace=True)
        rows.append(merged)

    if not rows:
        raise RuntimeError("no station rows written; check input dirs and obs_col")

    out = pd.concat(rows, axis=0, ignore_index=True)
    out = out.dropna(subset=["y_obs"])
    out = out.set_index(["fid", "day"]).sort_index()
    out.to_parquet(out_file)
    return out_file


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a station-day Parquet table keyed by (fid, day).")
    p.add_argument("--obs-dir", required=True, help="Directory of per-station daily Parquets (index must be date-like).")
    p.add_argument("--obs-col", required=True, help="Observed target column to use as y_obs (e.g., ea_obs).")
    p.add_argument("--rtma-daily-dir", default=None, help="Directory of per-station daily RTMA Parquets.")
    p.add_argument("--urma-daily-dir", default=None, help="Directory of per-station daily URMA Parquets.")
    p.add_argument("--out-file", required=True, help="Output Parquet path.")
    p.add_argument("--stations-csv", default=None, help="Optional station inventory CSV to restrict fids processed.")
    p.add_argument("--station-id-col", default="fid", help="Station id column name in --stations-csv.")
    p.add_argument(
        "--obs-day-mode",
        default="local",
        choices=["local", "utc_midpoint", "utc_start"],
        help="How to map tz-aware obs daily indexes onto a joinable day key.",
    )
    p.add_argument("--no-leakage-mask", action="store_true", help="Disable humidity leakage-safe column mask.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite out-file if it exists.")
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    inputs = Inputs(
        obs_dir=a.obs_dir,
        obs_col=a.obs_col,
        rtma_daily_dir=a.rtma_daily_dir,
        urma_daily_dir=a.urma_daily_dir,
        stations_csv=a.stations_csv,
        station_id_col=str(a.station_id_col),
    )
    build_station_day_table(
        inputs=inputs,
        out_file=a.out_file,
        obs_day_mode=str(a.obs_day_mode),
        leakage_safe=not bool(a.no_leakage_mask),
        overwrite=bool(a.overwrite),
    )


if __name__ == "__main__":
    main()
