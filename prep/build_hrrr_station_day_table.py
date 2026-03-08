"""
Build a flat station-day table for HRRR bias correction GNN.

Joins HRRR daily baselines + obs (via obsmet adapter) + terrain + Sx into
one Parquet keyed by (fid, day).  Computes parallel/perpendicular wind
residual targets and flow-terrain interaction features.

Output schema matches what ``models/hrrr_da/hrrr_dataset.py`` expects:
  Targets:   delta_w_par, delta_w_perp
  Obs:       u_obs, v_obs, wind_obs, wdir_obs
  HRRR weather: ugrd_hrrr, vgrd_hrrr, wind_hrrr, tmp_hrrr, dpt_hrrr, ...
  Derived:   tmp_dpt_diff
  Terrain:   elevation, slope, aspect_sin, aspect_cos, tpi_4, tpi_10
  Sx:        sx_*_2k, sx_*_10k, terrain_openness, terrain_directionality
  Flow-terrain: flow_upslope, flow_cross, wind_aligned_sx
  Temporal:  doy_sin, doy_cos
  Location:  latitude, longitude
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
) -> str:
    if os.path.exists(out_path) and not overwrite:
        print(f"Output exists: {out_path}")
        return out_path

    # Load station inventory
    stations = pd.read_csv(stations_csv)
    id_col = "station_id" if "station_id" in stations.columns else "fid"
    fids = set(stations[id_col].astype(str))
    print(f"Stations: {len(fids)}")

    # Load wind obs via obsmet adapter
    obs_all = load_station_daily(
        obsmet_source,
        channel=obsmet_channel,
        variables=["wind", "wind_dir"],
        fids=fids,
    )

    # Decompose wind/wind_dir into u/v
    wdir_rad = np.deg2rad(obs_all["wind_dir"])
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
    n_dropped = {"wind_range": 0, "delta_range": 0, "nan_wind": 0}
    processed = 0

    for fid in sorted(fids):
        hrrr_path = os.path.join(hrrr_dir, f"{fid}.parquet")
        if not os.path.exists(hrrr_path):
            continue
        if fid not in obs_fids:
            continue
        if fid not in station_static.index:
            continue

        obs = obs_all.loc[[fid]].droplevel(0)
        hrrr = _read_station_daily(hrrr_path)

        if "u" not in obs.columns or "v" not in obs.columns:
            continue

        # Join obs + hrrr on day
        merged = obs[["u", "v", "wind", "wind_dir"]].join(hrrr, how="inner")
        if merged.empty:
            continue

        merged = merged.rename(
            columns={
                "u": "u_obs",
                "v": "v_obs",
                "wind": "wind_obs",
                "wind_dir": "wdir_obs",
            }
        )

        # QC: drop NaN wind
        mask_nan = (
            merged["wind_obs"].isna() | merged["u_obs"].isna() | merged["v_obs"].isna()
        )
        n_dropped["nan_wind"] += mask_nan.sum()
        merged = merged[~mask_nan]

        # Wind range
        mask_range = (merged["wind_obs"] < 0) | (merged["wind_obs"] > 50)
        n_dropped["wind_range"] += mask_range.sum()
        merged = merged[~mask_range]

        # Require HRRR u/v
        if "ugrd_hrrr" not in merged.columns or "vgrd_hrrr" not in merged.columns:
            continue

        # Raw residuals
        merged["delta_u"] = merged["u_obs"] - merged["ugrd_hrrr"]
        merged["delta_v"] = merged["v_obs"] - merged["vgrd_hrrr"]

        # Delta range filter
        mask_delta = (merged["delta_u"].abs() > 20) | (merged["delta_v"].abs() > 20)
        n_dropped["delta_range"] += mask_delta.sum()
        merged = merged[~mask_delta]

        if merged.empty:
            continue

        # Parallel/perpendicular targets
        wind_hrrr = np.sqrt(
            merged["ugrd_hrrr"].values ** 2 + merged["vgrd_hrrr"].values ** 2
        )
        e_par_x = merged["ugrd_hrrr"].values / (wind_hrrr + EPS)
        e_par_y = merged["vgrd_hrrr"].values / (wind_hrrr + EPS)
        e_perp_x = -e_par_y
        e_perp_y = e_par_x

        merged["delta_w_par"] = (
            merged["delta_u"].values * e_par_x + merged["delta_v"].values * e_par_y
        )
        merged["delta_w_perp"] = (
            merged["delta_u"].values * e_perp_x + merged["delta_v"].values * e_perp_y
        )

        # Stability proxy
        if "tmp_hrrr" in merged.columns and "dpt_hrrr" in merged.columns:
            merged["tmp_dpt_diff"] = merged["tmp_hrrr"] - merged["dpt_hrrr"]

        # Temporal features
        doy = merged.index.dayofyear
        merged["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        merged["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

        # Static features
        static_row = station_static.loc[fid]

        # Sx columns
        for col in sx_cols:
            merged[col] = float(static_row[col]) if col in static_row.index else np.nan
        if "terrain_openness" in static_row.index:
            merged["terrain_openness"] = float(static_row["terrain_openness"])
        if "terrain_directionality" in static_row.index:
            merged["terrain_directionality"] = float(
                static_row["terrain_directionality"]
            )

        # Terrain
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

        # Location
        merged["latitude"] = float(static_row.get("latitude", np.nan))
        merged["longitude"] = float(static_row.get("longitude", np.nan))

        # Flow-terrain interaction features
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

        # Wind-aligned Sx
        if len(sx_cols_10k) > 0 and fid in sx_df.index:
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
    print(f"Dropped rows: {n_dropped}")

    if not all_rows:
        raise RuntimeError("No rows produced — check input paths")

    out = pd.concat(all_rows, axis=0, ignore_index=True)
    out = out.set_index(["fid", "day"]).sort_index()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_parquet(out_path)
    print(f"Written: {out_path}  ({len(out)} rows, {len(out.columns)} columns)")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build HRRR station-day table for GNN training."
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
    p.add_argument("--hrrr-dir", default=f"{MVP_ROOT}/hrrr_daily")
    p.add_argument("--sx-path", default=f"{MVP_ROOT}/station_sx_pnw.parquet")
    p.add_argument("--terrain-tif", default=f"{MVP_ROOT}/terrain_pnw_rtma.tif")
    p.add_argument("--stations-csv", default="artifacts/madis_pnw.csv")
    p.add_argument("--out", default=f"{MVP_ROOT}/station_day_hrrr_pnw.parquet")
    p.add_argument("--overwrite", action="store_true")
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
    )


if __name__ == "__main__":
    main()
