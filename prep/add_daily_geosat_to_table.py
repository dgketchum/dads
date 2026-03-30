"""Compute daily geosat aggregates from 3-hourly columns and add to the merged table."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input", required=True, help="Input parquet with 3-hourly geosat columns"
    )
    p.add_argument(
        "--output", required=True, help="Output parquet with daily aggregates added"
    )
    args = p.parse_args()

    print(f"Reading {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  Shape: {df.shape}")

    # Match only the 3-hourly timestamped columns (00, 03, ..., 21),
    # not any derived daily aggregates that may already exist in the table.
    _3h_stamps = {"00", "03", "06", "09", "12", "15", "18", "21"}
    irwin_cols = sorted(
        c
        for c in df.columns
        if c.startswith("irwin_cdr_")
        and c.endswith("_geosat")
        and c.split("_")[2] in _3h_stamps
    )
    irwvp_cols = sorted(
        c
        for c in df.columns
        if c.startswith("irwvp_")
        and c.endswith("_geosat")
        and c.split("_")[1] in _3h_stamps
    )
    print(f"  IR window cols: {len(irwin_cols)}")
    print(f"  Water vapor cols: {len(irwvp_cols)}")

    irwin = df[irwin_cols].to_numpy(dtype="float32")
    df["irwin_cdr_mean_geosat"] = np.nanmean(irwin, axis=1)
    df["irwin_cdr_min_geosat"] = np.nanmin(irwin, axis=1)
    df["irwin_cdr_range_geosat"] = np.nanmax(irwin, axis=1) - np.nanmin(irwin, axis=1)

    irwvp = df[irwvp_cols].to_numpy(dtype="float32")
    df["irwvp_mean_geosat"] = np.nanmean(irwvp, axis=1)

    new_cols = [
        "irwin_cdr_mean_geosat",
        "irwin_cdr_min_geosat",
        "irwin_cdr_range_geosat",
        "irwvp_mean_geosat",
    ]
    for c in new_cols:
        n_valid = df[c].notna().sum()
        print(f"  {c}: {n_valid}/{len(df)} valid ({100 * n_valid / len(df):.1f}%)")

    print(f"Writing {args.output}...")
    df.to_parquet(args.output, index=False)
    print(f"  Done. Shape: {df.shape}")


if __name__ == "__main__":
    main()
