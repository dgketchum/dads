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

import numpy as np
import pandas as pd

from prep.obsmet_adapter import load_station_daily


def _compute_doorenbos_rsds(
    tcdc: pd.Series, lat: pd.Series, doy: pd.Series
) -> pd.Series:
    """Doorenbos & Pruitt (1977) solar radiation from cloud cover + FAO-56 Ra.

    Parameters
    ----------
    tcdc : Series, cloud cover 0–100 %
    lat  : Series, latitude in degrees
    doy  : Series, day-of-year (1–366)

    Returns
    -------
    Rs in W/m² (daily mean equivalent).
    """
    # Sunshine fraction from cloud cover (Doorenbos & Pruitt 1977)
    n_over_N = (-0.0083 * tcdc + 0.9659).clip(0, 1)

    # FAO-56 extraterrestrial radiation Ra (MJ/m²/day)
    lat_rad = np.radians(lat)
    decl = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
    cos_ws = (-np.tan(lat_rad) * np.tan(decl)).clip(-1.0, 1.0)
    ws = np.arccos(cos_ws)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * doy)
    Gsc = 0.0820  # MJ/m²/min
    Ra_mj = (
        (24 * 60 / np.pi)
        * Gsc
        * dr
        * (
            ws * np.sin(lat_rad) * np.sin(decl)
            + np.cos(lat_rad) * np.cos(decl) * np.sin(ws)
        )
    ).clip(lower=0)

    # Angstrom–Prescott: Rs = (a + b * n/N) * Ra
    Rs_mj = (0.25 + 0.50 * n_over_N) * Ra_mj

    # MJ/m²/day → W/m²  (1 MJ/m²/day = 1e6 / 86400 W/m² ≈ 11.574)
    return Rs_mj / 0.0864


@dataclass(frozen=True)
class Inputs:
    obsmet_source: str  # e.g. "madis"
    variables: str  # comma-separated (e.g. "ea,tmax,tmin,wind")
    obsmet_channel: str = "prod"
    rtma_daily_dir: str | None = None
    urma_daily_dir: str | None = None
    cdr_daily_dir: str | None = None
    stations_csv: str | None = None
    station_id_col: str = "fid"

    @property
    def obs_cols(self) -> list[str]:
        return [c.strip() for c in self.variables.split(",")]


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
            idx2 = (
                (local_day + pd.Timedelta(hours=12))
                .tz_convert("UTC")
                .normalize()
                .tz_localize(None)
            )
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


def _compute_residuals(
    out: pd.DataFrame, obs_cols: list[str], multi_target: bool
) -> pd.DataFrame:
    """Compute residuals against RTMA/URMA baselines on the flat station-day table."""
    # ea residuals
    if "ea_rtma" in out.columns and "ea" in obs_cols:
        out["delta_ea_rtma"] = out["ea"] - out["ea_rtma"]
    elif "ea_rtma" in out.columns and not multi_target:
        out["delta_ea_rtma"] = out["y_obs"] - out["ea_rtma"]
    if "ea_urma" in out.columns and "ea" in obs_cols:
        out["delta_ea_urma"] = out["ea"] - out["ea_urma"]
    elif "ea_urma" in out.columns and not multi_target:
        out["delta_ea_urma"] = out["y_obs"] - out["ea_urma"]

    # tmax residuals
    if "tmax" in obs_cols:
        tmax_base_col = None
        for cand in ("tmax_urma", "tmax_rtma"):
            if cand in out.columns:
                tmax_base_col = cand
                break
        if tmax_base_col is not None:
            tmax_obs = pd.to_numeric(out["tmax"], errors="coerce")
            tmax_baseline = pd.to_numeric(out[tmax_base_col], errors="coerce")
            out["delta_tmax"] = tmax_obs - tmax_baseline
            out["tmax_baseline_source"] = pd.Series(pd.NA, index=out.index)
            has_tmax = out["delta_tmax"].notna()
            out.loc[has_tmax, "tmax_baseline_source"] = tmax_base_col

    # tmin residuals
    if "tmin" in obs_cols:
        tmin_base_col = None
        for cand in ("tmin_urma", "tmin_rtma"):
            if cand in out.columns:
                tmin_base_col = cand
                break
        if tmin_base_col is not None:
            tmin_obs = pd.to_numeric(out["tmin"], errors="coerce")
            tmin_baseline = pd.to_numeric(out[tmin_base_col], errors="coerce")
            out["delta_tmin"] = tmin_obs - tmin_baseline

    # wind residuals
    if "wind" in obs_cols:
        wind_base_col = None
        for cand in ("wind_urma", "wind_rtma"):
            if cand in out.columns:
                wind_base_col = cand
                break
        if wind_base_col is not None:
            wind_obs = pd.to_numeric(out["wind"], errors="coerce")
            wind_baseline = pd.to_numeric(out[wind_base_col], errors="coerce")
            out["delta_wind"] = wind_obs - wind_baseline

    # rsds residuals
    if "rsds" in obs_cols:
        for cand in ("rsds_urma", "rsds_rtma"):
            if cand in out.columns:
                rsds_obs = pd.to_numeric(out["rsds"], errors="coerce")
                rsds_baseline = pd.to_numeric(out[cand], errors="coerce")
                out["delta_rsds"] = rsds_obs - rsds_baseline
                break

    return out


def build_station_day_table(
    inputs: Inputs,
    out_file: str,
    leakage_safe: bool = True,
    overwrite: bool = False,
) -> str:
    if os.path.exists(out_file) and not overwrite:
        return out_file

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    stations = None
    allowed_fids: set[str] | None = None
    if inputs.stations_csv:
        stations = pd.read_csv(inputs.stations_csv)
        if inputs.station_id_col not in stations.columns:
            raise KeyError(
                f"station_id_col not found in stations_csv: {inputs.station_id_col}"
            )
        allowed_fids = set(stations[inputs.station_id_col].astype(str).tolist())
        if not allowed_fids:
            raise RuntimeError("stations_csv produced empty fid set")

    obs_cols = inputs.obs_cols
    multi_target = len(obs_cols) > 1

    # --- Load obs via obsmet adapter ---
    obs_all = load_station_daily(
        inputs.obsmet_source,
        channel=inputs.obsmet_channel,
        variables=obs_cols,
        fids=allowed_fids,
    )

    # Map first obs col to y_obs
    if obs_cols[0] in obs_all.columns:
        obs_all["y_obs"] = obs_all[obs_cols[0]]
    else:
        obs_all["y_obs"] = float("nan")

    # --- Join RTMA/URMA/CDR baselines per station ---
    rows = []
    for fid, grp in obs_all.groupby(level="fid"):
        merged = grp.copy()
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
        if inputs.cdr_daily_dir:
            p = os.path.join(inputs.cdr_daily_dir, f"{fid}.parquet")
            if os.path.exists(p):
                cdr = _read_station_daily_parquet(p)
                merged = merged.join(cdr, how="left")
        merged = merged.reset_index(level="fid", drop=True)
        merged["fid"] = str(fid)
        merged["day"] = merged.index.normalize()
        merged.reset_index(drop=True, inplace=True)
        rows.append(merged)

    if not rows:
        raise RuntimeError("no station rows written; check input dirs and variables")

    out = pd.concat(rows, axis=0, ignore_index=True)

    # --- Doorenbos (1977) synthetic rsds baseline from URMA cloud cover ---
    if "rsds" in obs_cols and "tcdc_urma" in out.columns:
        lat_map = None
        if inputs.stations_csv and stations is not None:
            sid = inputs.station_id_col
            if sid in stations.columns and "latitude" in stations.columns:
                lat_map = stations.set_index(sid)["latitude"].to_dict()
        if lat_map is not None:
            out["_lat"] = out["fid"].map(lat_map)
            out["_doy"] = pd.to_datetime(out["day"]).dt.dayofyear
            out["rsds_urma"] = _compute_doorenbos_rsds(
                out["tcdc_urma"], out["_lat"], out["_doy"]
            )
            out.drop(columns=["_lat", "_doy"], inplace=True)

    # --- Residuals (shared for both modes) ---
    out = _compute_residuals(out, obs_cols, multi_target)

    # --- Leakage mask ---
    if leakage_safe:
        drop_substrings = ("spfh_", "dpt_", "rh_", "q_", "sh_", "dew")
        safe_cols = set(obs_cols) | {
            "y_obs",
            "fid",
            "day",
            "ea_rtma",
            "ea_urma",
            "tmp_rtma",
            "tmp_urma",
            "tmax_rtma",
            "tmax_urma",
            "tmin_rtma",
            "tmin_urma",
            "wind_rtma",
            "wind_urma",
            "delta_ea_rtma",
            "delta_ea_urma",
            "delta_tmax",
            "delta_tmin",
            "delta_wind",
            "rsds_urma",
            "rsds_rtma",
            "delta_rsds",
            "wind_sensor_ht",
            "tmax_baseline_source",
        }
        keep_cols = []
        for c in out.columns:
            if c in safe_cols:
                keep_cols.append(c)
                continue
            if any(s in c.lower() for s in drop_substrings):
                continue
            keep_cols.append(c)
        out = out[keep_cols]

    # Keep rows with at least one valid obs target.
    if multi_target:
        out = out.dropna(subset=obs_cols, how="all")
    else:
        out = out.dropna(subset=["y_obs"])

    # Validate tmax baseline provenance
    if "tmax" in obs_cols and "tmax_baseline_source" in out.columns:
        has_delta = (
            out["delta_tmax"].notna()
            if "delta_tmax" in out.columns
            else pd.Series(False, index=out.index)
        )
        missing_src = int((has_delta & out["tmax_baseline_source"].isna()).sum())
        if missing_src:
            raise RuntimeError(
                f"Found {missing_src:,} rows with delta_tmax but missing tmax_baseline_source."
            )
        allowed_sources = {"tmax_rtma", "tmax_urma"}
        src_vals = (
            out.loc[has_delta, "tmax_baseline_source"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        bad = set(src_vals) - allowed_sources
        if bad:
            raise RuntimeError(
                f"Unexpected tmax baseline sources found: {sorted(bad)}. "
                f"Allowed: {sorted(allowed_sources)}."
            )

    out = out.set_index(["fid", "day"]).sort_index()
    out.to_parquet(out_file)
    n_days = out.index.get_level_values("day").nunique()
    n_stations = out.index.get_level_values("fid").nunique()
    print(
        f"  wrote {len(out):,} rows ({n_stations:,} stations, {n_days} days) → {out_file}",
        flush=True,
    )
    return out_file


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a station-day Parquet table keyed by (fid, day)."
    )
    p.add_argument(
        "--obsmet-source",
        required=True,
        help="obsmet source identifier (e.g. madis, ghcnd).",
    )
    p.add_argument(
        "--obsmet-channel",
        default="prod",
        help="obsmet release channel (default: prod).",
    )
    p.add_argument(
        "--variables",
        required=True,
        help="Target variable(s), comma-separated (e.g. ea,tmax,tmin,wind).",
    )
    p.add_argument(
        "--rtma-daily-dir",
        default=None,
        help="Directory of per-station daily RTMA Parquets.",
    )
    p.add_argument(
        "--urma-daily-dir",
        default=None,
        help="Directory of per-station daily URMA Parquets.",
    )
    p.add_argument(
        "--cdr-daily-dir",
        default=None,
        help="Directory of per-station daily CDR Parquets (NOAA CDR surface reflectance).",
    )
    p.add_argument("--out-file", required=True, help="Output Parquet path.")
    p.add_argument(
        "--stations-csv",
        default=None,
        help="Optional station inventory CSV to restrict fids processed.",
    )
    p.add_argument(
        "--station-id-col",
        default="fid",
        help="Station id column name in --stations-csv.",
    )
    p.add_argument(
        "--no-leakage-mask",
        action="store_true",
        help="Disable humidity leakage-safe column mask.",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite out-file if it exists."
    )
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    inputs = Inputs(
        obsmet_source=a.obsmet_source,
        obsmet_channel=str(a.obsmet_channel),
        variables=a.variables,
        rtma_daily_dir=a.rtma_daily_dir,
        urma_daily_dir=a.urma_daily_dir,
        cdr_daily_dir=a.cdr_daily_dir,
        stations_csv=a.stations_csv,
        station_id_col=str(a.station_id_col),
    )
    build_station_day_table(
        inputs=inputs,
        out_file=a.out_file,
        leakage_safe=not bool(a.no_leakage_mask),
        overwrite=bool(a.overwrite),
    )


if __name__ == "__main__":
    main()
