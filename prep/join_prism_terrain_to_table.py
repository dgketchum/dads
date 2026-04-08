"""
Sample PRISM terrain rasters at station locations and join to station-day parquet.

Adds 6 columns:
  effective_terrain_height, terrain_i3d,
  facet_sin_12km, facet_cos_12km, facet_sin_36km, facet_cos_36km

Station lon/lat are reprojected to the raster CRS (EPSG:5070) for sampling.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from prep.paths import MVP_ROOT

# Facet orientation class → geographic degrees (0=N, CW)
_ORIENT_DEG = {
    0: np.nan,  # flat → sin=0, cos=0
    1: 0.0,
    2: 45.0,
    3: 90.0,
    4: 135.0,
    5: 180.0,
    6: 225.0,
    7: 270.0,
    8: 315.0,
}


def _sample_raster_at_stations(
    tif_path: str,
    xs_proj: np.ndarray,
    ys_proj: np.ndarray,
    bands: list[int] | None = None,
) -> np.ndarray:
    """Sample raster band(s) at projected coordinates.

    Returns array of shape (n_stations,) for single band or
    (n_bands, n_stations) for multiple bands.
    """
    with rasterio.open(tif_path) as src:
        inv = ~src.transform
        cols_f, rows_f = inv * (xs_proj, ys_proj)
        rows = np.clip(np.round(rows_f).astype(int), 0, src.height - 1)
        cols = np.clip(np.round(cols_f).astype(int), 0, src.width - 1)
        if bands is None:
            bands = list(range(1, src.count + 1))
        data = src.read(bands)
    if len(bands) == 1:
        return data[0, rows, cols].astype("float32")
    return data[:, rows, cols].astype("float32")


def _orient_to_sincos(
    orient_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert orientation class (0-8) to sin/cos. Flat (0) → 0, 0."""
    deg = np.array([_ORIENT_DEG.get(int(v), np.nan) for v in orient_vals])
    rad = np.radians(deg)
    s = np.where(np.isfinite(rad), np.sin(rad), 0.0).astype("float32")
    c = np.where(np.isfinite(rad), np.cos(rad), 0.0).astype("float32")
    return s, c


def join_prism_terrain(
    table_path: str,
    stations_csv: str,
    out_path: str,
    eth_tif: str | None = None,
    i3d_tif: str | None = None,
    orient_tif: str | None = None,
    overwrite: bool = False,
) -> str:
    if os.path.exists(out_path) and not overwrite:
        print(f"Output exists: {out_path}")
        return out_path

    # Defaults
    eth_tif = eth_tif or os.path.join(
        MVP_ROOT, "prism_effective_terrain_height_pnw_1km.tif"
    )
    i3d_tif = i3d_tif or os.path.join(
        MVP_ROOT, "prism_effective_terrain_i3d_pnw_1km.tif"
    )
    orient_tif = orient_tif or os.path.join(
        MVP_ROOT, "prism_facet_orientation_pnw_1km.tif"
    )

    # Load stations
    stations = pd.read_csv(stations_csv)
    id_col = "station_id" if "station_id" in stations.columns else "fid"
    fids = stations[id_col].astype(str).values
    lons = stations["longitude"].values
    lats = stations["latitude"].values
    print(f"Stations: {len(fids)}")

    # Reproject lon/lat → EPSG:5070
    with rasterio.open(eth_tif) as src:
        dst_crs = str(src.crs)
    to_proj = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    xs, ys = to_proj.transform(lons, lats)

    # Sample rasters
    print("Sampling effective terrain height …")
    eth_vals = _sample_raster_at_stations(eth_tif, xs, ys)

    print("Sampling I3d …")
    i3d_vals = _sample_raster_at_stations(i3d_tif, xs, ys)

    print("Sampling facet orientations (bands 2 and 4 = 12 km and 36 km) …")
    orient_12 = _sample_raster_at_stations(orient_tif, xs, ys, bands=[2])
    orient_36 = _sample_raster_at_stations(orient_tif, xs, ys, bands=[4])

    sin_12, cos_12 = _orient_to_sincos(orient_12)
    sin_36, cos_36 = _orient_to_sincos(orient_36)

    # Build station-level lookup
    static = pd.DataFrame(
        {
            "fid": fids,
            "effective_terrain_height": eth_vals,
            "terrain_i3d": i3d_vals,
            "facet_sin_12km": sin_12,
            "facet_cos_12km": cos_12,
            "facet_sin_36km": sin_36,
            "facet_cos_36km": cos_36,
        }
    ).set_index("fid")

    print(f"Station static sample:\n{static.describe()}")

    # Load and join to station-day table
    print(f"Loading table: {table_path}")
    df = pd.read_parquet(table_path)
    n_before = len(df)
    fid_col = "station_id" if "station_id" in df.columns else "fid"
    df[fid_col] = df[fid_col].astype(str)

    # Drop existing columns if present (for re-runs)
    new_cols = list(static.columns)
    for c in new_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    df = df.join(static, on=fid_col)
    assert len(df) == n_before, "Row count changed after join"

    # Check for NaNs
    for c in new_cols:
        n_nan = df[c].isna().sum()
        if n_nan > 0:
            print(f"  WARNING: {c} has {n_nan} NaN values ({n_nan / len(df):.1%})")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Written: {out_path}  ({len(df)} rows, {len(df.columns)} cols)")
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Join PRISM terrain features to station-day parquet."
    )
    p.add_argument(
        "--table-path",
        default=os.path.join(MVP_ROOT, "station_day_hrrr_daily_cdr_pnw.parquet"),
    )
    p.add_argument(
        "--stations-csv",
        default="/home/dgketchum/code/dads/artifacts/madis_pnw.csv",
    )
    p.add_argument(
        "--out-path", default=None, help="Output parquet (default: overwrite input)"
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    out = a.out_path or a.table_path
    join_prism_terrain(
        table_path=a.table_path,
        stations_csv=a.stations_csv,
        out_path=out,
        overwrite=a.overwrite,
    )


if __name__ == "__main__":
    main()
