"""
Clip a stations CSV by geographic bounds (and optionally downsample).

This is a convenience tool for the RTMA bias MVP to keep sampling/training runs
bounded to the PNW (or any region) without editing upstream station inventories.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def clip_stations(
    in_csv: str,
    out_csv: str,
    bounds: tuple[float, float, float, float],
    id_col: str = "fid",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    limit: int | None = None,
    seed: int = 0,
) -> str:
    w, s, e, n = map(float, bounds)
    df = pd.read_csv(in_csv)
    if id_col not in df.columns:
        raise KeyError(f"id_col not found: {id_col}")
    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError(f"lat/lon columns not found: {lat_col}, {lon_col}")

    out = df.copy()
    out[id_col] = out[id_col].astype(str)
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out = out.dropna(subset=[lat_col, lon_col])

    out = out[
        (out[lat_col] >= s)
        & (out[lat_col] <= n)
        & (out[lon_col] >= w)
        & (out[lon_col] <= e)
    ].copy()
    out = out.drop_duplicates(subset=[id_col]).reset_index(drop=True)

    if limit is not None:
        k = int(limit)
        if k > 0 and len(out) > k:
            rng = np.random.default_rng(int(seed))
            keep = rng.choice(len(out), size=k, replace=False)
            out = out.iloc[np.sort(keep)].reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out_csv


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clip station inventory CSV by bounds.")
    p.add_argument("--in", dest="in_csv", required=True, help="Input stations CSV.")
    p.add_argument(
        "--out", dest="out_csv", required=True, help="Output clipped stations CSV."
    )
    p.add_argument(
        "--bounds", nargs=4, type=float, metavar=("W", "S", "E", "N"), required=True
    )
    p.add_argument("--id-col", default="fid")
    p.add_argument("--lat-col", default="latitude")
    p.add_argument("--lon-col", default="longitude")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max stations to keep (random sample).",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    clip_stations(
        in_csv=a.in_csv,
        out_csv=a.out_csv,
        bounds=tuple(a.bounds),
        id_col=a.id_col,
        lat_col=a.lat_col,
        lon_col=a.lon_col,
        limit=a.limit,
        seed=int(a.seed),
    )


if __name__ == "__main__":
    main()
