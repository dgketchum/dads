"""
Validate daily 1 km HRRR COGs produced by build_hrrr_1km.py.

Three checks are performed:

1. Metadata: each .tif has the expected shape, CRS, transform, band count,
   and band descriptions matching OUTPUT_BANDS from build_hrrr_1km.py.

2. Band sanity: spot-check physical value ranges for a sample of files.

3. Station cross-check: sample each raster at station lat/lon positions and
   compare against the corresponding per-station HRRR daily parquets.

Usage:
    uv run python -m prep.validate_hrrr_1km \\
        --background-dir /nas/dads/mvp/hrrr_1km_pnw \\
        --station-table /nas/dads/mvp/station_day_hrrr_pnw.parquet \\
        --hrrr-daily-dir /nas/dads/mvp/hrrr_daily \\
        --n-sample 20
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol

from prep.build_hrrr_1km import OUTPUT_BANDS
from prep.pnw_1km_grid import PNW_1KM_SHAPE, PNW_1KM_TRANSFORM

# Plausible physical ranges for PNW daily values
_SANITY_RANGES: dict[str, tuple[float, float]] = {
    "tmp_hrrr": (-60.0, 55.0),
    "tmax_hrrr": (-60.0, 60.0),
    "tmin_hrrr": (-70.0, 55.0),
    "dpt_hrrr": (-70.0, 35.0),
    "ea_hrrr": (0.0, 10.0),
    "pres_hrrr": (50.0, 110.0),
    "ugrd_hrrr": (-80.0, 80.0),
    "vgrd_hrrr": (-80.0, 80.0),
    "wind_hrrr": (0.0, 100.0),
    "wdir_hrrr": (0.0, 360.0),
    "dswrf_hrrr": (0.0, 1200.0),
    "spfh_hrrr": (0.0, 0.05),
    "tcdc_hrrr": (0.0, 100.0),
    "hpbl_hrrr": (0.0, 10000.0),
    "n_hours": (0.0, 24.0),
}


# ── metadata check ────────────────────────────────────────────────────────────


def check_metadata(path: str) -> list[str]:
    """Return a list of error strings; empty = OK."""
    errors: list[str] = []
    try:
        with rasterio.open(path) as src:
            if (src.height, src.width) != PNW_1KM_SHAPE:
                errors.append(
                    f"shape {(src.height, src.width)} != expected {PNW_1KM_SHAPE}"
                )
            if not src.crs.to_epsg() == 5070:
                errors.append(f"CRS {src.crs} != EPSG:5070")
            if src.transform != PNW_1KM_TRANSFORM:
                errors.append(f"transform mismatch: {src.transform}")
            if src.count != len(OUTPUT_BANDS):
                errors.append(f"band count {src.count} != {len(OUTPUT_BANDS)}")
            descs = list(src.descriptions)
            if tuple(d or "" for d in descs) != OUTPUT_BANDS:
                errors.append(f"band descriptions: {descs}")
    except Exception as exc:
        errors.append(f"open failed: {exc}")
    return errors


# ── band sanity check ─────────────────────────────────────────────────────────


def check_sanity(path: str) -> list[str]:
    """Return a list of out-of-range warnings; empty = OK."""
    warnings: list[str] = []
    with rasterio.open(path) as src:
        for i, name in enumerate(OUTPUT_BANDS, 1):
            data = src.read(i)
            valid = data[np.isfinite(data)]
            if len(valid) == 0:
                warnings.append(f"{name}: all NaN")
                continue
            lo, hi = _SANITY_RANGES.get(name, (-1e9, 1e9))
            if valid.min() < lo or valid.max() > hi:
                warnings.append(
                    f"{name}: [{valid.min():.3g}, {valid.max():.3g}] "
                    f"outside [{lo}, {hi}]"
                )
    return warnings


# ── station cross-check ───────────────────────────────────────────────────────

# Subset of bands also present in per-station HRRR daily parquets
_CROSSCHECK_BANDS = (
    "tmp_hrrr",
    "tmax_hrrr",
    "tmin_hrrr",
    "dpt_hrrr",
    "ea_hrrr",
    "pres_hrrr",
    "ugrd_hrrr",
    "vgrd_hrrr",
    "wind_hrrr",
    "dswrf_hrrr",
    "spfh_hrrr",
    "tcdc_hrrr",
    "hpbl_hrrr",
)


def _sample_raster_at_ll(
    path: str,
    lons: np.ndarray,
    lats: np.ndarray,
) -> dict[str, np.ndarray]:
    """Sample raster at (lon, lat) points; returns {band_name: values}."""
    with rasterio.open(path) as src:
        src_crs = src.crs
        tf = src.transform
        data = src.read().astype("float32")

    to_src = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
    xs, ys = to_src.transform(lons, lats)
    rows, cols = rowcol(tf, xs, ys)
    rows = np.array(rows, dtype=int)
    cols = np.array(cols, dtype=int)
    H, W = data.shape[1], data.shape[2]
    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)

    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(OUTPUT_BANDS):
        vals = np.full(len(lons), np.nan, dtype="float32")
        vals[valid] = data[i, rows[valid], cols[valid]]
        out[name] = vals
    return out


def cross_check_stations(
    background_dir: str,
    station_table: str,
    hrrr_daily_dir: str,
    n_sample: int = 20,
) -> pd.DataFrame:
    """Sample rasters at station locations and compare to per-station parquets.

    Returns a DataFrame of (fid, date, band, raster_val, parquet_val, diff).
    """
    df = pd.read_parquet(station_table)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["fid"] = df["fid"].astype(str)
    df["day"] = pd.to_datetime(df["day"]).dt.normalize()

    # Pick a random sample of fids that have parquet files
    fids_with_parquet = [
        fid
        for fid in df["fid"].unique()
        if os.path.exists(os.path.join(hrrr_daily_dir, f"{fid}.parquet"))
    ]
    if not fids_with_parquet:
        print("  No station parquets found — skipping cross-check.")
        return pd.DataFrame()

    rng = np.random.default_rng(seed=0)
    sampled_fids = list(
        rng.choice(
            fids_with_parquet, size=min(n_sample, len(fids_with_parquet)), replace=False
        )
    )

    # Build station metadata (lat/lon per fid)
    meta = (
        df[df["fid"].isin(sampled_fids)][["fid", "latitude", "longitude"]]
        .drop_duplicates("fid")
        .set_index("fid")
    )

    # Discover available rasters
    raster_files = sorted(
        f
        for f in os.listdir(background_dir)
        if f.startswith("HRRR_1km_") and f.endswith(".tif")
    )
    if not raster_files:
        print("  No raster files found — skipping cross-check.")
        return pd.DataFrame()

    sample_files = raster_files[:: max(1, len(raster_files) // 20)][:20]

    records: list[dict] = []
    for fname in sample_files:
        day_str = fname.replace("HRRR_1km_", "").replace(".tif", "")
        try:
            day = pd.Timestamp(day_str)
        except ValueError:
            continue
        tif_path = os.path.join(background_dir, fname)

        for fid in sampled_fids:
            if fid not in meta.index:
                continue
            lat = float(meta.loc[fid, "latitude"])
            lon = float(meta.loc[fid, "longitude"])

            raster_vals = _sample_raster_at_ll(
                tif_path,
                np.array([lon]),
                np.array([lat]),
            )

            parq_path = os.path.join(hrrr_daily_dir, f"{fid}.parquet")
            try:
                parq = pd.read_parquet(parq_path)
            except Exception:
                continue
            parq.index = pd.to_datetime(parq.index, errors="coerce").normalize()
            if day not in parq.index:
                continue

            for band in _CROSSCHECK_BANDS:
                rv = float(raster_vals[band][0]) if band in raster_vals else np.nan
                pv = float(parq.loc[day, band]) if band in parq.columns else np.nan
                records.append(
                    {
                        "fid": fid,
                        "date": day_str,
                        "band": band,
                        "raster_val": rv,
                        "parquet_val": pv,
                        "diff": rv - pv
                        if np.isfinite(rv) and np.isfinite(pv)
                        else np.nan,
                    }
                )

    return pd.DataFrame(records)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Validate HRRR 1 km COG outputs.")
    p.add_argument(
        "--background-dir", required=True, help="Directory of HRRR_1km_*.tif files."
    )
    p.add_argument(
        "--station-table",
        default=None,
        help="Station-day parquet table (for cross-check).",
    )
    p.add_argument(
        "--hrrr-daily-dir",
        default=None,
        help="Directory of per-station HRRR daily parquets ({fid}.parquet).",
    )
    p.add_argument(
        "--n-sample",
        type=int,
        default=20,
        help="Number of stations to sample in cross-check (default: 20).",
    )
    p.add_argument(
        "--sanity-sample",
        type=int,
        default=10,
        help="Number of raster files to sanity-check (default: 10).",
    )
    a = p.parse_args()

    raster_files = sorted(
        os.path.join(a.background_dir, f)
        for f in os.listdir(a.background_dir)
        if f.startswith("HRRR_1km_") and f.endswith(".tif")
    )
    print(f"Found {len(raster_files)} raster(s) in {a.background_dir}")

    # ── Check 2: metadata ──────────────────────────────────────────────────────
    print("\n=== Metadata check ===")
    meta_errors = 0
    for path in raster_files:
        errs = check_metadata(path)
        if errs:
            meta_errors += 1
            print(f"  FAIL {os.path.basename(path)}: {'; '.join(errs)}")
    if meta_errors == 0:
        print(f"  OK — all {len(raster_files)} files pass metadata check.")
    else:
        print(f"  {meta_errors}/{len(raster_files)} files have metadata errors.")

    # ── Check 3: band sanity ───────────────────────────────────────────────────
    print("\n=== Band sanity check ===")
    step = max(1, len(raster_files) // a.sanity_sample)
    sampled = raster_files[::step][: a.sanity_sample]
    sanity_warnings = 0
    for path in sampled:
        warns = check_sanity(path)
        if warns:
            sanity_warnings += len(warns)
            print(f"  WARN {os.path.basename(path)}:")
            for w in warns:
                print(f"    {w}")
    if sanity_warnings == 0:
        print(f"  OK — {len(sampled)} sampled files pass sanity check.")

    # ── Check 1: station cross-check ───────────────────────────────────────────
    if a.station_table and a.hrrr_daily_dir:
        print("\n=== Station cross-check ===")
        xdf = cross_check_stations(
            a.background_dir,
            a.station_table,
            a.hrrr_daily_dir,
            n_sample=a.n_sample,
        )
        if xdf.empty:
            print("  No cross-check records produced.")
        else:
            summary = (
                xdf.groupby("band")["diff"]
                .agg(["count", "mean", "std", lambda x: x.abs().mean()])
                .rename(columns={"<lambda_0>": "mae"})
                .round(4)
            )
            print(summary.to_string())
            n_large = int((xdf["diff"].abs() > 2.0).sum())
            print(f"  Pairs with |diff| > 2.0: {n_large}/{len(xdf)}")
    else:
        print(
            "\n(Skipping station cross-check: --station-table / --hrrr-daily-dir not provided.)"
        )

    sys.exit(0 if meta_errors == 0 else 1)


if __name__ == "__main__":
    main()
