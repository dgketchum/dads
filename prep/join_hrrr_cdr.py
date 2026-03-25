"""
Join VIIRS CDR daily features onto the HRRR station-day parquet.

Reads per-station CDR parquets and left-joins (fid, day) onto the HRRR table.

Usage:
    uv run python prep/join_hrrr_cdr.py
"""

from __future__ import annotations

import os

import pandas as pd

HRRR_TABLE = "/nas/dads/mvp/station_day_hrrr_pnw.parquet"
CDR_DIR = "/nas/dads/mvp/cdr_daily_pnw_2018_2024"
OUT_TABLE = "/nas/dads/mvp/station_day_hrrr_cdr_pnw.parquet"


def main() -> None:
    df = pd.read_parquet(HRRR_TABLE)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["fid"] = df["fid"].astype(str)
    df["day"] = pd.to_datetime(df["day"]).dt.normalize()
    print(f"HRRR table: {len(df)} rows, {len(df.columns)} cols")

    cdr_frames = []
    fids = df["fid"].unique()
    n_found = 0
    for fid in fids:
        path = os.path.join(CDR_DIR, f"{fid}.parquet")
        if not os.path.exists(path):
            continue
        cdr = pd.read_parquet(path)
        cdr.index = pd.to_datetime(cdr.index).normalize()
        cdr = cdr.reset_index().rename(columns={cdr.index.name or "index": "day"})
        cdr["day"] = pd.to_datetime(cdr["day"]).dt.normalize()
        cdr["fid"] = fid
        cdr_frames.append(cdr)
        n_found += 1

    print(f"CDR parquets found: {n_found}/{len(fids)} stations")
    cdr_all = pd.concat(cdr_frames, ignore_index=True)
    print(f"CDR rows: {len(cdr_all)}, cols: {list(cdr_all.columns)}")

    merged = df.merge(cdr_all, on=["fid", "day"], how="left")
    cdr_cols = [c for c in cdr_all.columns if c not in ("fid", "day")]
    n_nonnull = merged[cdr_cols[0]].notna().sum() if cdr_cols else 0
    print(
        f"Merged: {len(merged)} rows, {len(merged.columns)} cols, CDR non-null: {n_nonnull}"
    )

    merged.to_parquet(OUT_TABLE, index=False)
    print(f"Wrote {OUT_TABLE}")


if __name__ == "__main__":
    main()
