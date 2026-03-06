"""Build a station-day KNN pair index for spatial-gradient training."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing in degrees from point 1 to point 2."""
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dlon = np.deg2rad(lon2 - lon1)
    y = np.sin(dlon) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlon)
    b = np.degrees(np.arctan2(y, x))
    return float((b + 360.0) % 360.0)


def build_pair_index(
    patch_index: str,
    out_parquet: str,
    target_col: str = "delta_tmax",
    k: int = 8,
    max_distance_km: float = 35.0,
    day_col: str = "day",
    fid_col: str = "fid",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    elev_col: str | None = None,
) -> str:
    use_cols = [fid_col, day_col, lat_col, lon_col]
    optional_cols = [target_col, "MGRS_TILE"]
    if elev_col:
        optional_cols.append(elev_col)

    df = pd.read_parquet(patch_index)
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(f"patch_index missing required columns: {missing}")

    keep = use_cols + [c for c in optional_cols if c in df.columns]
    df = df[keep].copy()
    df[fid_col] = df[fid_col].astype(str)
    df[day_col] = pd.to_datetime(df[day_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=[day_col, lat_col, lon_col])

    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    rows: list[dict] = []
    total_days = int(df[day_col].nunique())
    done = 0

    for day, g in df.groupby(day_col, sort=True):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            done += 1
            continue

        coords = np.deg2rad(g[[lat_col, lon_col]].to_numpy(dtype="float64"))
        tree = BallTree(coords, metric="haversine")
        qk = min(int(k) + 1, len(g))
        dists, nbrs = tree.query(coords, k=qk)

        seen: set[tuple[int, int]] = set()
        for i in range(len(g)):
            for d_rad, j in zip(dists[i, 1:], nbrs[i, 1:]):
                j = int(j)
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in seen:
                    continue
                dist_km = float(d_rad * 6371.0)
                if dist_km > float(max_distance_km):
                    continue
                seen.add((a, b))

                ri = g.iloc[a]
                rj = g.iloc[b]
                out = {
                    "day": pd.Timestamp(day),
                    "fid_i": str(ri[fid_col]),
                    "fid_j": str(rj[fid_col]),
                    "lat_i": float(ri[lat_col]),
                    "lon_i": float(ri[lon_col]),
                    "lat_j": float(rj[lat_col]),
                    "lon_j": float(rj[lon_col]),
                    "dist_km": dist_km,
                    "bearing_deg": _bearing_deg(
                        float(ri[lat_col]),
                        float(ri[lon_col]),
                        float(rj[lat_col]),
                        float(rj[lon_col]),
                    ),
                }
                if "MGRS_TILE" in g.columns:
                    out["MGRS_TILE_i"] = ri.get("MGRS_TILE")
                    out["MGRS_TILE_j"] = rj.get("MGRS_TILE")
                if elev_col and elev_col in g.columns:
                    zi = pd.to_numeric(ri.get(elev_col), errors="coerce")
                    zj = pd.to_numeric(rj.get(elev_col), errors="coerce")
                    out["dz"] = (
                        float(zi - zj) if pd.notna(zi) and pd.notna(zj) else np.nan
                    )
                if target_col in g.columns:
                    yi = pd.to_numeric(ri.get(target_col), errors="coerce")
                    yj = pd.to_numeric(rj.get(target_col), errors="coerce")
                    out["y_pair"] = (
                        float(yi - yj) if pd.notna(yi) and pd.notna(yj) else np.nan
                    )
                rows.append(out)

        done += 1
        if done % 50 == 0 or done == total_days:
            print(f"  processed {done}/{total_days} days", flush=True)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No pairs generated; check input data and constraints.")

    os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
    out.to_parquet(out_parquet, index=False)
    print(f"  pair index: {len(out):,} rows written to {out_parquet}")
    return out_parquet


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a station-day KNN pair index for pairwise spatial losses."
    )
    p.add_argument("--patch-index", required=True, help="Input patch-index parquet")
    p.add_argument("--out", required=True, help="Output pair-index parquet")
    p.add_argument(
        "--target-col",
        default="delta_tmax",
        help="Target column used to precompute y_pair (default: delta_tmax)",
    )
    p.add_argument("--k", type=int, default=8, help="K nearest neighbors")
    p.add_argument(
        "--max-distance-km",
        type=float,
        default=35.0,
        help="Max neighbor distance in km",
    )
    p.add_argument("--day-col", default="day")
    p.add_argument("--fid-col", default="fid")
    p.add_argument("--lat-col", default="latitude")
    p.add_argument("--lon-col", default="longitude")
    p.add_argument(
        "--elev-col",
        default=None,
        help="Optional elevation column for dz computation",
    )
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    build_pair_index(
        patch_index=a.patch_index,
        out_parquet=a.out,
        target_col=a.target_col,
        k=int(a.k),
        max_distance_km=float(a.max_distance_km),
        day_col=a.day_col,
        fid_col=a.fid_col,
        lat_col=a.lat_col,
        lon_col=a.lon_col,
        elev_col=a.elev_col,
    )


if __name__ == "__main__":
    main()
