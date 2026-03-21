"""
Build a flat station-day table for HRRR bias correction GNN.

Joins HRRR daily baselines + obs (via obsmet adapter) + terrain + Sx into
one Parquet keyed by (fid, day).  Computes residual targets for all
supported variables; missing obs propagate as NaN (not dropped).

Output schema
-------------
Targets (may be NaN):
  delta_tmax, delta_tmin, delta_tair, delta_ea, delta_rsds,
  delta_w_par, delta_w_perp
Obs (may be NaN):
  tmax_obs, tmin_obs, tair_obs, td_obs, ea_obs, rsds_obs,
  wind_obs, wdir_obs, u_obs, v_obs
HRRR weather:
  ugrd_hrrr, vgrd_hrrr, wind_hrrr, tmp_hrrr, tmax_hrrr, tmin_hrrr,
  dpt_hrrr, ea_hrrr, dswrf_hrrr, pres_hrrr, tcdc_hrrr, hpbl_hrrr,
  spfh_hrrr, wdir_hrrr
Derived:
  tmp_dpt_diff
Terrain:
  elevation, slope, aspect_sin, aspect_cos, tpi_4, tpi_10
Sx:
  sx_*_2k, sx_*_10k, terrain_openness, terrain_directionality
Flow-terrain:
  flow_upslope, flow_cross, wind_aligned_sx
Temporal:
  doy_sin, doy_cos
Location:
  latitude, longitude
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import rasterio

from prep.obsmet_adapter import load_station_daily
from prep.paths import MVP_ROOT

EPS = 1e-6
# HRRR dswrf_hrrr is W/m²; MADIS rsds is MJ/m²/day
_WM2_TO_MJD = 86400.0 / 1e6

OBS_VARS = ["tmax", "tmin", "tair", "td", "wind", "wind_dir", "rsds"]

_OBS_TARGET_COLS = [
    "delta_tmax",
    "delta_tmin",
    "delta_tair",
    "delta_ea",
    "delta_rsds",
    "delta_w_par",
    "delta_w_perp",
]


def _read_station_daily(path: str) -> pd.DataFrame:
    """Read a per-station daily parquet, normalising the index to naive dates."""
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        for c in ("time", "date", "day", "dt"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
                df = df.set_index(c)
                break
    idx = pd.to_datetime(df.index, errors="coerce")
    df = df[idx.notna()].copy()
    df.index = idx[idx.notna()]
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.index.name = "day"
    return df


def _lerp_sx(
    sx_values: np.ndarray, azimuths_deg: np.ndarray, target_deg: float
) -> float:
    """Linearly interpolate Sx from precomputed azimuth bins."""
    n = len(azimuths_deg)
    step = 360.0 / n
    idx_lo = int(target_deg / step) % n
    idx_hi = (idx_lo + 1) % n
    az_lo = azimuths_deg[idx_lo]
    frac = (target_deg - az_lo) % 360.0 / step
    return float(sx_values[idx_lo] * (1.0 - frac) + sx_values[idx_hi] * frac)


def _sample_terrain_at_stations(
    terrain_tif: str, lons: np.ndarray, lats: np.ndarray
) -> pd.DataFrame:
    """Sample all bands of the terrain TIF at station lon/lat positions."""
    with rasterio.open(terrain_tif) as src:
        rows_cols = [src.index(lon, lat) for lon, lat in zip(lons, lats)]
        rows = np.clip([rc[0] for rc in rows_cols], 0, src.height - 1)
        cols = np.clip([rc[1] for rc in rows_cols], 0, src.width - 1)
        data = src.read()
        names = (
            list(src.descriptions)
            if src.descriptions
            else [f"band_{i}" for i in range(src.count)]
        )
    result = {}
    for i, name in enumerate(names):
        result[name] = data[i, rows, cols].astype("float32")
    return pd.DataFrame(result)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_hrrr_station_day_table(
    obsmet_source: str,
    hrrr_dir: str,
    sx_path: str,
    terrain_tif: str,
    stations_csv: str,
    out_path: str,
    overwrite: bool = False,
    obsmet_channel: str = "prod",
    por_path: str | None = None,
    n_stations: int | None = None,
) -> str:
    if os.path.exists(out_path) and not overwrite:
        print(f"Output exists: {out_path}")
        return out_path

    # Load station inventory
    stations = pd.read_csv(stations_csv)
    id_col = "station_id" if "station_id" in stations.columns else "fid"
    fids = set(stations[id_col].astype(str))
    print(f"Stations: {len(fids)}")

    # Load all obs variables via obsmet adapter
    obs_all = load_station_daily(
        obsmet_source,
        channel=obsmet_channel,
        por_path=por_path,
        variables=OBS_VARS,
        fids=fids,
    )

    # Vectorised u/v decomposition (NaN-safe: NaN wind_dir → NaN u/v)
    wdir_rad = np.deg2rad(obs_all["wind_dir"].values)
    obs_all["u"] = -obs_all["wind"] * np.sin(wdir_rad)
    obs_all["v"] = -obs_all["wind"] * np.cos(wdir_rad)
    obs_fids = set(obs_all.index.get_level_values("fid").unique())

    # Load Sx table
    sx_df = pd.read_parquet(sx_path)
    sx_df["fid"] = sx_df["fid"].astype(str)
    sx_df = sx_df.set_index("fid")
    sx_cols_2k = sorted(
        c for c in sx_df.columns if c.startswith("sx_") and c.endswith("_2k")
    )
    sx_cols_10k = sorted(
        c for c in sx_df.columns if c.startswith("sx_") and c.endswith("_10k")
    )
    sx_cols = sx_cols_2k + sx_cols_10k
    sx_azimuths_10k = np.array([float(c.split("_")[1]) for c in sx_cols_10k])
    print(f"Sx columns: {len(sx_cols)} ({len(sx_cols_2k)} 2k + {len(sx_cols_10k)} 10k)")

    # Sample terrain at station locations
    sta_lons = stations["longitude"].values if "longitude" in stations.columns else None
    sta_lats = stations["latitude"].values if "latitude" in stations.columns else None
    terrain_df = _sample_terrain_at_stations(terrain_tif, sta_lons, sta_lats)
    terrain_df["fid"] = stations[id_col].astype(str).values
    terrain_df = terrain_df.set_index("fid")

    # Join station-level static data: sx + terrain
    station_static = sx_df.join(terrain_df, how="inner")
    print(f"Stations with Sx + terrain: {len(station_static)}")

    # Process station by station
    all_rows = []
    n_qc = {
        "wind_range": 0,
        "wind_delta": 0,
        "temp_qc": 0,
        "ea_qc": 0,
        "rsds_qc": 0,
        "no_obs": 0,
    }
    processed = 0

    for fid in sorted(fids)[:n_stations] if n_stations else sorted(fids):
        hrrr_path = os.path.join(hrrr_dir, f"{fid}.parquet")
        if not os.path.exists(hrrr_path):
            continue
        if fid not in obs_fids:
            continue
        if fid not in station_static.index:
            continue

        obs = obs_all.loc[[fid]].droplevel(0)
        hrrr = _read_station_daily(hrrr_path)

        # Inner join: keep days where both obs row AND hrrr row exist.
        # Individual obs columns may still be NaN after the join.
        obs_cols = [
            c
            for c in [
                "tmax",
                "tmin",
                "tair",
                "td",
                "wind",
                "wind_dir",
                "rsds",
                "u",
                "v",
            ]
            if c in obs.columns
        ]
        merged = obs[obs_cols].join(hrrr, how="inner")
        if merged.empty:
            continue

        # Rename obs columns
        merged = merged.rename(
            columns={
                "tmax": "tmax_obs",
                "tmin": "tmin_obs",
                "tair": "tair_obs",
                "td": "td_obs",
                "wind": "wind_obs",
                "wind_dir": "wdir_obs",
                "rsds": "rsds_obs",
                "u": "u_obs",
                "v": "v_obs",
            }
        )

        # ---- Compute ea_obs from td_obs (Tetens, NaN-safe) ----
        if "td_obs" in merged.columns:
            td = merged["td_obs"]
            merged["ea_obs"] = 0.6108 * np.exp(17.27 * td / (td + 237.3))

        # ---- Delta columns (NaN propagates naturally) ----
        if "tmax_obs" in merged.columns and "tmax_hrrr" in merged.columns:
            merged["delta_tmax"] = merged["tmax_obs"] - merged["tmax_hrrr"]
        if "tmin_obs" in merged.columns and "tmin_hrrr" in merged.columns:
            merged["delta_tmin"] = merged["tmin_obs"] - merged["tmin_hrrr"]
        if "tair_obs" in merged.columns and "tmp_hrrr" in merged.columns:
            merged["delta_tair"] = merged["tair_obs"] - merged["tmp_hrrr"]
        if "ea_obs" in merged.columns and "ea_hrrr" in merged.columns:
            merged["delta_ea"] = merged["ea_obs"] - merged["ea_hrrr"]
        if "rsds_obs" in merged.columns and "dswrf_hrrr" in merged.columns:
            # dswrf_hrrr is W/m²; rsds_obs is MJ/m²/day
            merged["delta_rsds"] = (
                merged["rsds_obs"] - merged["dswrf_hrrr"] * _WM2_TO_MJD
            )

        # ---- Wind decomposition (NaN-safe via u_obs/v_obs) ----
        if "ugrd_hrrr" in merged.columns and "vgrd_hrrr" in merged.columns:
            wind_hrrr_spd = np.sqrt(
                merged["ugrd_hrrr"].values ** 2 + merged["vgrd_hrrr"].values ** 2
            )
            e_par_x = merged["ugrd_hrrr"].values / (wind_hrrr_spd + EPS)
            e_par_y = merged["vgrd_hrrr"].values / (wind_hrrr_spd + EPS)
            e_perp_x = -e_par_y
            e_perp_y = e_par_x

            if "u_obs" in merged.columns and "v_obs" in merged.columns:
                delta_u = merged["u_obs"].values - merged["ugrd_hrrr"].values
                delta_v = merged["v_obs"].values - merged["vgrd_hrrr"].values
                merged["delta_w_par"] = delta_u * e_par_x + delta_v * e_par_y
                merged["delta_w_perp"] = delta_u * e_perp_x + delta_v * e_perp_y
        else:
            # HRRR u/v missing — can't compute flow basis; set defaults
            e_par_x = np.ones(len(merged))
            e_par_y = np.zeros(len(merged))

        # ---- QC: set flagged values to NaN, do NOT drop rows ----

        # Temperature QC
        for dcol, ocol in [("delta_tmax", "tmax_obs"), ("delta_tmin", "tmin_obs")]:
            if dcol in merged.columns:
                bad = merged[dcol].abs() > 20
                n_qc["temp_qc"] += int(bad.sum())
                merged.loc[bad, [ocol, dcol]] = np.nan

        # ea QC
        if "ea_obs" in merged.columns and "delta_ea" in merged.columns:
            bad_ea = (merged["ea_obs"] < 0) | (merged["delta_ea"].abs() > 3)
            n_qc["ea_qc"] += int(bad_ea.sum())
            merged.loc[bad_ea, ["ea_obs", "delta_ea"]] = np.nan

        # rsds QC
        if "rsds_obs" in merged.columns:
            bad_rs = merged["rsds_obs"] < 0
            n_qc["rsds_qc"] += int(bad_rs.sum())
            merged.loc[bad_rs, ["rsds_obs", "delta_rsds"]] = np.nan

        # Wind QC (range + delta)
        if "wind_obs" in merged.columns:
            bad_range = (merged["wind_obs"] < 0) | (merged["wind_obs"] > 50)
            n_qc["wind_range"] += int(bad_range.sum())
            merged.loc[
                bad_range,
                [
                    "wind_obs",
                    "wdir_obs",
                    "u_obs",
                    "v_obs",
                    "delta_w_par",
                    "delta_w_perp",
                ],
            ] = np.nan

        if "delta_w_par" in merged.columns:
            # Recompute delta_u/delta_v for range check from current u/v
            if "u_obs" in merged.columns and "ugrd_hrrr" in merged.columns:
                du = merged["u_obs"] - merged["ugrd_hrrr"]
                dv = merged["v_obs"] - merged["vgrd_hrrr"]
                bad_delta = (du.abs() > 20) | (dv.abs() > 20)
                # Only flag where obs actually exists (not already NaN)
                bad_delta = bad_delta & merged["u_obs"].notna()
                n_qc["wind_delta"] += int(bad_delta.sum())
                merged.loc[
                    bad_delta,
                    [
                        "wind_obs",
                        "wdir_obs",
                        "u_obs",
                        "v_obs",
                        "delta_w_par",
                        "delta_w_perp",
                    ],
                ] = np.nan

        # Row filter: keep rows with at least one non-NaN target
        target_cols_present = [c for c in _OBS_TARGET_COLS if c in merged.columns]
        if not target_cols_present:
            continue
        has_any = merged[target_cols_present].notna().any(axis=1)
        n_qc["no_obs"] += int((~has_any).sum())
        merged = merged[has_any]
        if merged.empty:
            continue

        # ---- Stability proxy ----
        if "tmp_hrrr" in merged.columns and "dpt_hrrr" in merged.columns:
            merged["tmp_dpt_diff"] = merged["tmp_hrrr"] - merged["dpt_hrrr"]

        # ---- Temporal features ----
        doy = merged.index.dayofyear
        merged["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        merged["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

        # ---- Static features ----
        static_row = station_static.loc[fid]

        for col in sx_cols:
            merged[col] = float(static_row[col]) if col in static_row.index else np.nan
        if "terrain_openness" in static_row.index:
            merged["terrain_openness"] = float(static_row["terrain_openness"])
        if "terrain_directionality" in static_row.index:
            merged["terrain_directionality"] = float(
                static_row["terrain_directionality"]
            )

        for col in [
            "elevation",
            "slope",
            "aspect_sin",
            "aspect_cos",
            "tpi_4",
            "tpi_10",
        ]:
            if col in static_row.index:
                merged[col] = float(static_row[col])

        merged["latitude"] = float(static_row.get("latitude", np.nan))
        merged["longitude"] = float(static_row.get("longitude", np.nan))

        # ---- Flow-terrain interaction (uses HRRR wind basis, always available) ----
        # Recompute wind basis after row filter — pre-filter arrays have stale length.
        if "ugrd_hrrr" in merged.columns and "vgrd_hrrr" in merged.columns:
            _spd = np.sqrt(
                merged["ugrd_hrrr"].values ** 2 + merged["vgrd_hrrr"].values ** 2
            )
            e_par_x = merged["ugrd_hrrr"].values / (_spd + EPS)
            e_par_y = merged["vgrd_hrrr"].values / (_spd + EPS)
        else:
            e_par_x = np.ones(len(merged))
            e_par_y = np.zeros(len(merged))
        if "aspect_sin" in static_row.index and "aspect_cos" in static_row.index:
            asp_sin = float(static_row["aspect_sin"])
            asp_cos = float(static_row["aspect_cos"])
            e_up_x = -asp_sin
            e_up_y = -asp_cos
            e_cross_x = e_up_y
            e_cross_y = -e_up_x
            merged["flow_upslope"] = e_par_x * e_up_x + e_par_y * e_up_y
            merged["flow_cross"] = e_par_x * e_cross_x + e_par_y * e_cross_y
        else:
            merged["flow_upslope"] = 0.0
            merged["flow_cross"] = 0.0

        # ---- Wind-aligned Sx ----
        if (
            len(sx_cols_10k) > 0
            and fid in sx_df.index
            and "ugrd_hrrr" in merged.columns
        ):
            sx_10k_vals = sx_df.loc[fid, sx_cols_10k].values.astype("float64")
            wdir_from = (
                np.degrees(
                    np.arctan2(-merged["ugrd_hrrr"].values, -merged["vgrd_hrrr"].values)
                )
                + 360.0
            ) % 360.0
            merged["wind_aligned_sx"] = [
                _lerp_sx(sx_10k_vals, sx_azimuths_10k, wd) for wd in wdir_from
            ]
        else:
            merged["wind_aligned_sx"] = 0.0

        merged["fid"] = fid
        merged.index.name = "day"
        merged = merged.reset_index()
        all_rows.append(merged)
        processed += 1

        if processed % 1000 == 0:
            print(f"  {processed} stations processed")

    print(f"Processed {processed} stations")
    print(f"QC counts: {n_qc}")

    if not all_rows:
        raise RuntimeError("No rows produced — check input paths")

    out = pd.concat(all_rows, axis=0, ignore_index=True)
    out = out.set_index(["fid", "day"]).sort_index()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_parquet(out_path)
    print(f"Written: {out_path}  ({len(out):,} rows, {len(out.columns)} columns)")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build multivariable HRRR station-day table for GNN training."
    )
    p.add_argument(
        "--obsmet-source",
        default="madis",
        help="obsmet source identifier (default: madis).",
    )
    p.add_argument(
        "--obsmet-channel",
        default="prod",
        help="obsmet release channel (default: prod).",
    )
    p.add_argument(
        "--por-path",
        default=None,
        help="Direct path to permissive obsmet station_por dir (bypasses channel).",
    )
    p.add_argument("--hrrr-dir", default=f"{MVP_ROOT}/hrrr_daily")
    p.add_argument("--sx-path", default=f"{MVP_ROOT}/station_sx_pnw.parquet")
    p.add_argument("--terrain-tif", default=f"{MVP_ROOT}/terrain_pnw_rtma.tif")
    p.add_argument("--stations-csv", default="artifacts/madis_pnw.csv")
    p.add_argument("--out", default=f"{MVP_ROOT}/station_day_hrrr_pnw.parquet")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--n-stations",
        type=int,
        default=None,
        help="Limit to first N stations (for testing).",
    )
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    build_hrrr_station_day_table(
        obsmet_source=a.obsmet_source,
        hrrr_dir=a.hrrr_dir,
        sx_path=a.sx_path,
        terrain_tif=a.terrain_tif,
        stations_csv=a.stations_csv,
        out_path=a.out,
        overwrite=a.overwrite,
        obsmet_channel=a.obsmet_channel,
        por_path=a.por_path,
        n_stations=a.n_stations,
    )


if __name__ == "__main__":
    main()
