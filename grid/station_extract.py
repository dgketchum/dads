"""Shared utilities for extracting gridded data at station locations.

Provides:
- ``load_station_csv`` — read a station inventory CSV with flexible column names
- ``project_to_lcc_pixels`` — project lon/lat to Lambert Conformal pixel coords
- ``build_haversine_tree`` / ``query_nearest`` — BallTree nearest-neighbor lookup
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── station CSV loading ───────────────────────────────────────────────────────

_ID_CANDIDATES = ("fid", "station_id", "STAID")
_LAT_CANDIDATES = ("latitude", "lat", "LAT", "Latitude")
_LON_CANDIDATES = ("longitude", "lon", "LON", "Longitude")


def load_station_csv(
    csv_path: str,
    bounds: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    """Load a station inventory CSV, returning a DataFrame with fid/latitude/longitude.

    Parameters
    ----------
    csv_path : str
        Path to a CSV containing station id, latitude, and longitude columns.
        Recognized column names are tried in priority order (see _ID_CANDIDATES etc.).
    bounds : (west, south, east, north) or None
        Optional lon/lat bounding box to clip stations.

    Returns
    -------
    DataFrame with columns ``fid`` (str), ``latitude`` (float), ``longitude`` (float).
    """
    df = pd.read_csv(csv_path)

    def _pick(cols, candidates, label):
        for c in candidates:
            if c in cols:
                return c
        raise KeyError(f"No {label} column found; tried {candidates}")

    id_col = _pick(df.columns, _ID_CANDIDATES, "station-ID")
    lat_col = _pick(df.columns, _LAT_CANDIDATES, "latitude")
    lon_col = _pick(df.columns, _LON_CANDIDATES, "longitude")

    out = df[[id_col, lat_col, lon_col]].copy()
    out.columns = ["fid", "latitude", "longitude"]
    out["fid"] = out["fid"].astype(str)
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.dropna(subset=["fid", "latitude", "longitude"])

    if bounds is not None:
        w, s, e, n = bounds
        out = out[
            (out["longitude"] >= w)
            & (out["longitude"] <= e)
            & (out["latitude"] >= s)
            & (out["latitude"] <= n)
        ]

    return out[["fid", "latitude", "longitude"]].reset_index(drop=True)


# ── Lambert Conformal pixel projection ────────────────────────────────────────


def project_to_lcc_pixels(
    grib_path: str,
    lons: np.ndarray,
    lats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Project lon/lat coordinates to pixel (row, col) in a GRIB's Lambert Conformal grid.

    Parameters
    ----------
    grib_path : str
        Path to a GRIB2 file whose CRS and transform define the grid.
    lons, lats : array-like
        Station coordinates in EPSG:4326.

    Returns
    -------
    rows, cols : int arrays
        Pixel row/col for each station (rounded to nearest).
    valid : bool array
        True where the station falls inside the grid.
    grid_height, grid_width : int
        Dimensions of the grid.
    """
    import rasterio
    from pyproj import Transformer

    with rasterio.open(grib_path) as src:
        crs = src.crs
        transform = src.transform
        h, w = src.height, src.width

    xformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = xformer.transform(lons, lats)

    inv = ~transform
    cols_f, rows_f = inv * (xs, ys)
    rows = np.round(rows_f).astype(int)
    cols = np.round(cols_f).astype(int)

    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    return rows, cols, valid, h, w


# ── BallTree nearest-neighbor ─────────────────────────────────────────────────


def build_haversine_tree(
    lats: np.ndarray,
    lons: np.ndarray,
) -> "BallTree":  # noqa: F821
    """Build a haversine BallTree from grid-point lat/lon (degrees).

    Works with any grid whose points are given as 1-D arrays of lat/lon,
    including HRRR grids where lons are in [0, 360).
    """
    from sklearn.neighbors import BallTree

    coords = np.deg2rad(np.column_stack([lats, lons]))
    return BallTree(coords, metric="haversine")


def query_nearest(
    tree: "BallTree",  # noqa: F821
    sta_lats: np.ndarray,
    sta_lons: np.ndarray,
) -> np.ndarray:
    """Find the nearest grid index for each station.

    Parameters
    ----------
    tree : BallTree
        Built by ``build_haversine_tree``.
    sta_lats, sta_lons : array-like
        Station coordinates in degrees.  Longitudes may be in [-180, 180]
        or [0, 360) — the caller must ensure they match the tree's convention.

    Returns
    -------
    1-D int array of grid indices, one per station.
    """
    coords = np.deg2rad(np.column_stack([sta_lats, sta_lons]))
    _, indices = tree.query(coords, k=1)
    return indices.ravel()
