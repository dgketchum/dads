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

# FAO-56 neutral log-law constant: ln(67.8 * 10 - 5.42)
_LN_10M = np.log(67.8 * 10 - 5.42)  # ln(672.58) ≈ 6.5116


def _load_provider_wind_heights(path: str) -> dict[str, float]:
    """Parse ``urma2p5_provider_windheight`` into {provider_8char: height_m}.

    Format per line: ``A8 2X A8 2X F5.2`` (Fortran fixed-width).
    Lines with a specific subprovider (not ``allsprvs``) are stored under
    ``"provider:subprov"``; ``allsprvs`` lines under just ``"provider"``.
    Lookup order at query time: specific key first, then provider-only, then
    default 10.0 m.
    """
    heights: dict[str, float] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            provider = line[:8].strip()
            subprov = line[10:18].strip()
            ht = float(line[20:].strip())
            if subprov == "allsprvs":
                heights[provider] = ht
            else:
                heights[f"{provider}:{subprov}"] = ht
    return heights


def _adjust_wind_to_10m(wind: pd.Series, zw: pd.Series) -> pd.Series:
    """FAO-56 neutral log-law adjustment from sensor height *zw* to 10 m AGL.

    ``u10 = uz * ln(672.58) / ln(67.8 * zw - 5.42)``

    Guard: heights <= 0.1 m are clipped (log argument would go non-positive).
    At zw=10 the factor is 1.0; at zw=2 it is ~1.34; at zw=3 ~1.23; at zw=6 ~1.09.
    """
    zw_safe = zw.clip(lower=0.2)
    factor = _LN_10M / np.log(67.8 * zw_safe - 5.42)
    return wind * factor


@dataclass(frozen=True)
class Inputs:
    obs_dir: str
    obs_col: str  # comma-separated for multi-target (e.g. "ea,tmax")
    obs_format: str = "station_por"  # "station_por" or "daily_all"
    rtma_daily_dir: str | None = None
    urma_daily_dir: str | None = None
    stations_csv: str | None = None
    station_id_col: str = "fid"
    provider_heights_path: str | None = None
    synoptic_crosswalk_path: str | None = None

    @property
    def obs_cols(self) -> list[str]:
        return [c.strip() for c in self.obs_col.split(",")]


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


def _read_daily_all_obs(
    obs_dir: str,
    allowed_fids: set[str] | None,
    obs_cols: list[str],
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    provider_heights_path: str | None = None,
    synoptic_crosswalk_path: str | None = None,
) -> pd.DataFrame:
    """Aggregate hourly obs from per-day parquets into station-day rows.

    Each file in obs_dir is ``YYYYMMDD.parquet`` with hourly rows from MADIS.
    Returns a DataFrame indexed by (fid, day) with columns from *obs_cols*.
    """
    day_files = sorted(
        fn for fn in os.listdir(obs_dir) if fn.endswith(".parquet") and fn[:8].isdigit()
    )
    # Filter to requested date range.
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Pre-load provider wind heights once (not per-day)
    prov_ht: dict[str, float] | None = None
    if provider_heights_path is not None and "wind" in obs_cols:
        prov_ht = _load_provider_wind_heights(provider_heights_path)
        print(
            f"  wind height correction: loaded {len(prov_ht)} provider entries",
            flush=True,
        )

    # Synoptic crosswalk: stationId → wind_sensor_ht (overrides provider default
    # for MesoWest sub-networks like AGRIMET, PGN, MT-MESO)
    synoptic_ht: dict[str, float] | None = None
    if synoptic_crosswalk_path is not None and "wind" in obs_cols:
        xw = pd.read_csv(
            synoptic_crosswalk_path, usecols=["stationId", "wind_sensor_ht"]
        )
        synoptic_ht = dict(zip(xw["stationId"], xw["wind_sensor_ht"]))
        print(
            f"  synoptic crosswalk: {len(synoptic_ht)} stations "
            f"({sum(1 for v in synoptic_ht.values() if v != 10.0)} non-10m)",
            flush=True,
        )

    all_chunks: list[pd.DataFrame] = []
    for fn in day_files:
        day_str = fn[:8]
        day_ts = pd.Timestamp(day_str)
        if day_ts < start or day_ts > end:
            continue

        df = pd.read_parquet(os.path.join(obs_dir, fn))

        # QC filter
        if "qc_passed" in df.columns:
            df = df.loc[df["qc_passed"]].copy()

        # Station filter
        if "stationId" in df.columns:
            df = df.rename(columns={"stationId": "fid"})
        if "fid" not in df.columns:
            continue
        df["fid"] = df["fid"].astype(str)
        if allowed_fids is not None:
            df = df.loc[df["fid"].isin(allowed_fids)]
        if df.empty:
            continue

        # Unit conversions: temperature and dewpoint from K to C
        if "temperature" in df.columns:
            df["temperature_C"] = df["temperature"] - 273.15
        if "dewpoint" in df.columns:
            df["dewpoint_C"] = df["dewpoint"] - 273.15

        # Wind height correction: adjust obs from sensor height to 10 m AGL
        if (
            prov_ht is not None
            and "windSpeed" in df.columns
            and "dataProvider" in df.columns
        ):
            prov8 = df["dataProvider"].str[:8]
            df["wind_sensor_ht"] = prov8.map(prov_ht).fillna(10.0)
            # Override with Synoptic crosswalk (handles MesoWest sub-networks)
            if synoptic_ht is not None:
                xw_ht = df["fid"].map(synoptic_ht)
                has_xw = xw_ht.notna()
                df.loc[has_xw, "wind_sensor_ht"] = xw_ht[has_xw]
            needs_adj = df["wind_sensor_ht"] != 10.0
            if needs_adj.any():
                df.loc[needs_adj, "windSpeed"] = _adjust_wind_to_10m(
                    df.loc[needs_adj, "windSpeed"],
                    df.loc[needs_adj, "wind_sensor_ht"],
                ).values

        # Per station-day aggregation
        agg = {}
        if "tmax" in obs_cols and "temperature_C" in df.columns:
            agg["tmax"] = ("temperature_C", "max")
        if "tmin" in obs_cols and "temperature_C" in df.columns:
            agg["tmin"] = ("temperature_C", "min")
        if "ea" in obs_cols and "dewpoint_C" in df.columns:
            # Tetens from dewpoint: ea = 0.6108 * exp(17.27 * Td / (Td + 237.3))
            df["ea_inst"] = 0.6108 * np.exp(
                17.27 * df["dewpoint_C"] / (df["dewpoint_C"] + 237.3)
            )
            agg["ea"] = ("ea_inst", "mean")
        if "wind" in obs_cols and "windSpeed" in df.columns:
            agg["wind"] = ("windSpeed", "mean")
            if "wind_sensor_ht" in df.columns:
                agg["wind_sensor_ht"] = ("wind_sensor_ht", "first")
        if not agg:
            continue

        grouped = df.groupby("fid").agg(**agg)

        # Physical bounds filtering
        if "tmax" in grouped.columns:
            grouped.loc[~grouped["tmax"].between(-50, 60), "tmax"] = np.nan
        if "tmin" in grouped.columns:
            grouped.loc[~grouped["tmin"].between(-50, 60), "tmin"] = np.nan
        if "ea" in grouped.columns:
            grouped.loc[~grouped["ea"].between(0.0001, 8.0), "ea"] = np.nan
        if "wind" in grouped.columns:
            grouped.loc[~grouped["wind"].between(0, 35), "wind"] = np.nan

        grouped["day"] = day_ts
        all_chunks.append(grouped.reset_index())

        if len(all_chunks) % 50 == 0:
            print(f"  daily_all: processed {len(all_chunks)} day files ...", flush=True)

    if not all_chunks:
        raise RuntimeError("No daily_all obs found in date range")

    out = pd.concat(all_chunks, ignore_index=True)
    print(
        f"  daily_all: {len(out):,} station-day rows from {len(all_chunks)} days, "
        f"{out['fid'].nunique():,} stations",
        flush=True,
    )
    return out.set_index(["fid", "day"]).sort_index()


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

    return out


def build_station_day_table(
    inputs: Inputs,
    out_file: str,
    leakage_safe: bool = True,
    overwrite: bool = False,
    obs_day_mode: str = "local",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
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
            raise KeyError(
                f"station_id_col not found in stations_csv: {inputs.station_id_col}"
            )
        allowed_fids = set(stations[inputs.station_id_col].astype(str).tolist())
        if not allowed_fids:
            raise RuntimeError("stations_csv produced empty fid set")

    obs_cols = inputs.obs_cols
    multi_target = len(obs_cols) > 1

    # --- daily_all mode: read per-day parquets, aggregate, then join baselines ---
    if inputs.obs_format == "daily_all":
        obs_all = _read_daily_all_obs(
            obs_dir=inputs.obs_dir,
            allowed_fids=allowed_fids,
            obs_cols=obs_cols,
            start_date=start_date,
            end_date=end_date,
            provider_heights_path=inputs.provider_heights_path,
            synoptic_crosswalk_path=inputs.synoptic_crosswalk_path,
        )
        # For backward compat: first obs col maps to y_obs
        if obs_cols[0] in obs_all.columns:
            obs_all["y_obs"] = obs_all[obs_cols[0]]
        else:
            obs_all["y_obs"] = float("nan")

        # Join RTMA/URMA baselines per station
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
            merged = merged.reset_index(level="fid", drop=True)
            merged["fid"] = str(fid)
            merged["day"] = merged.index.normalize()
            merged.reset_index(drop=True, inplace=True)
            rows.append(merged)

        if not rows:
            raise RuntimeError("no station rows written; check input dirs and obs_col")

        out = pd.concat(rows, axis=0, ignore_index=True)

    # --- station_por mode: read per-station parquets (original behavior) ---
    else:
        rows = []
        for fn in sorted(os.listdir(inputs.obs_dir)):
            if not fn.endswith(".parquet"):
                continue
            fid = os.path.splitext(fn)[0]
            if allowed_fids is not None and fid not in allowed_fids:
                continue
            obs_path = os.path.join(inputs.obs_dir, fn)
            obs = _read_station_daily_parquet(obs_path, day_mode=obs_day_mode)

            present = [c for c in obs_cols if c in obs.columns]
            if not present:
                continue

            keep = {}
            for c in obs_cols:
                if c in obs.columns:
                    keep[c] = obs[c]
                else:
                    keep[c] = pd.Series(float("nan"), index=obs.index)
            keep["y_obs"] = keep[obs_cols[0]]
            obs_sub = pd.DataFrame(keep, index=obs.index)

            merged = obs_sub
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

            merged = merged.copy()
            merged["fid"] = fid
            merged["day"] = merged.index.normalize()
            merged.reset_index(drop=True, inplace=True)
            rows.append(merged)

        if not rows:
            raise RuntimeError("no station rows written; check input dirs and obs_col")

        out = pd.concat(rows, axis=0, ignore_index=True)

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
        "--obs-dir",
        required=True,
        help="Directory of per-station daily Parquets or daily_all day files.",
    )
    p.add_argument(
        "--obs-col",
        required=True,
        help="Observed target column(s), comma-separated (e.g. ea,tmax,tmin,wind).",
    )
    p.add_argument(
        "--obs-format",
        default="station_por",
        choices=["station_por", "daily_all"],
        help="Obs directory layout: station_por (per-station) or daily_all (per-day).",
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
        "--provider-heights",
        default=None,
        help="Path to urma2p5_provider_windheight file for wind sensor height lookup.",
    )
    p.add_argument(
        "--synoptic-crosswalk",
        default=None,
        help="Path to Synoptic wind height crosswalk CSV (overrides provider defaults for MesoWest sub-networks).",
    )
    p.add_argument(
        "--obs-day-mode",
        default="local",
        choices=["local", "utc_midpoint", "utc_start"],
        help="How to map tz-aware obs daily indexes onto a joinable day key.",
    )
    p.add_argument(
        "--start-date",
        default="2024-01-01",
        help="Start date for daily_all obs (YYYY-MM-DD).",
    )
    p.add_argument(
        "--end-date",
        default="2024-12-31",
        help="End date for daily_all obs (YYYY-MM-DD).",
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
        obs_dir=a.obs_dir,
        obs_col=a.obs_col,
        obs_format=str(a.obs_format),
        rtma_daily_dir=a.rtma_daily_dir,
        urma_daily_dir=a.urma_daily_dir,
        stations_csv=a.stations_csv,
        station_id_col=str(a.station_id_col),
        provider_heights_path=a.provider_heights,
        synoptic_crosswalk_path=a.synoptic_crosswalk,
    )
    build_station_day_table(
        inputs=inputs,
        out_file=a.out_file,
        obs_day_mode=str(a.obs_day_mode),
        leakage_safe=not bool(a.no_leakage_mask),
        overwrite=bool(a.overwrite),
        start_date=a.start_date,
        end_date=a.end_date,
    )


if __name__ == "__main__":
    main()
