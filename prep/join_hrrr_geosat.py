"""
Join 3-hourly geostationary satellite features onto the HRRR+CDR station-day parquet.

Usage:
    uv run python -m prep.join_hrrr_geosat
"""

from __future__ import annotations

import os

import pandas as pd

HRRR_CDR_TABLE = "/nas/dads/mvp/station_day_hrrr_cdr_pnw.parquet"
GEOSAT_DIR = "/nas/dads/mvp/geosat_3h_pnw_2018_2024"
OUT_TABLE = "/nas/dads/mvp/station_day_hrrr_cdr_geosat_pnw.parquet"


def main() -> None:
    df = pd.read_parquet(HRRR_CDR_TABLE)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["fid"] = df["fid"].astype(str)
    df["day"] = pd.to_datetime(df["day"]).dt.normalize()
    print(f"HRRR+CDR table: {len(df)} rows, {len(df.columns)} cols")

    geosat_frames = []
    fids = df["fid"].unique()
    n_found = 0
    for fid in fids:
        path = os.path.join(GEOSAT_DIR, f"{fid}.parquet")
        if not os.path.exists(path):
            continue
        gs = pd.read_parquet(path)
        gs.index = pd.to_datetime(gs.index).normalize()
        gs = gs.reset_index().rename(columns={gs.index.name or "index": "day"})
        gs["day"] = pd.to_datetime(gs["day"]).dt.normalize()
        gs["fid"] = fid
        geosat_frames.append(gs)
        n_found += 1

    print(f"Geosat parquets found: {n_found}/{len(fids)} stations")
    geosat_all = pd.concat(geosat_frames, ignore_index=True)
    print(f"Geosat rows: {len(geosat_all)}, cols: {list(geosat_all.columns)[:5]}...")

    merged = df.merge(geosat_all, on=["fid", "day"], how="left")
    gs_cols = [c for c in geosat_all.columns if c not in ("fid", "day")]
    n_nonnull = merged[gs_cols[0]].notna().sum() if gs_cols else 0
    print(
        f"Merged: {len(merged)} rows, {len(merged.columns)} cols, geosat non-null: {n_nonnull}"
    )

    merged.to_parquet(OUT_TABLE, index=False)
    print(f"Wrote {OUT_TABLE}")


if __name__ == "__main__":
    main()
