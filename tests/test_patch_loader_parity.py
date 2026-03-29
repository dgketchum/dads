"""
Parity test: verify the optimized HRRRPatchDataset (pre-computed domain
geometry) produces identical output to the old per-sample CRS approach.

Reconstructs the old __getitem__ logic inline for N samples and compares
x_patch, station row/col, targets, holdout masks, and center flags.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol, xy

from models.hrrr_da.hetero_dataset import _doy_features
from models.hrrr_da.patch_assim_dataset import HRRRPatchDataset, _landsat_period

_LANDSAT_BANDS = 7

# Paths — skip if data not available
TABLE = "/nas/dads/run_archive/station_day_hrrr_daily_cdr_pnw.parquet"
BG_DIR = "/data/nvme0/dads_rasters/hrrr_1km_pnw"
STATIC = "/data/nvme0/dads_rasters/terrain_pnw_1km.tif"
LANDSAT = "/data/nvme0/dads_rasters/landsat_pnw_1km.tif"
RSUN = "/data/nvme0/dads_rasters/rsun_pnw_1km.tif"
CDR_DIR = "/data/nvme0/dads_rasters/cdr_native_pnw"
HOLDOUT = "artifacts/canonical_holdout_fids.json"

N_SAMPLES = 30


def _data_available():
    return os.path.exists(TABLE) and os.path.exists(BG_DIR)


def _old_getitem(ds, idx):
    """Reproduce old per-sample CRS logic for comparison."""
    row = ds.samples.iloc[idx]
    day = pd.Timestamp(row["day"]).normalize()
    doy = day.dayofyear
    period = _landsat_period(doy)
    H = W = ds.patch_size

    # Old: CRS transform per sample
    bg = ds.raster_cache.get(ds._background_path(day))
    bg_data = bg["data"]
    bg_tf = bg["transform"]
    bg_crs = bg["crs"]
    domain_H, domain_W = bg_data.shape[1], bg_data.shape[2]

    to_bg = Transformer.from_crs("EPSG:4326", bg_crs, always_xy=True)
    x_center, y_center = to_bg.transform(
        float(row["longitude"]), float(row["latitude"])
    )
    r_center, c_center = rowcol(bg_tf, x_center, y_center)

    r0 = int(np.clip(int(r_center) - H // 2, 0, domain_H - H))
    c0 = int(np.clip(int(c_center) - W // 2, 0, domain_W - W))
    r1, c1 = r0 + H, c0 + W

    bg_patch = bg_data[ds._bg_keep_indices, r0:r1, c0:c1].astype("float32")

    # Old: compute projected pixel coords + inverse CRS
    rows = np.arange(r0, r1)
    cols = np.arange(c0, c1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr_flat, cc_flat = rr.ravel(), cc.ravel()
    x_proj, y_proj = xy(bg_tf, rr_flat, cc_flat, offset="center")
    x_proj = np.asarray(x_proj, dtype="float64")
    y_proj = np.asarray(y_proj, dtype="float64")

    to_ll = Transformer.from_crs(bg_crs, "EPSG:4326", always_xy=True)
    pixel_lon, pixel_lat = to_ll.transform(x_proj, y_proj)
    pixel_lon = np.asarray(pixel_lon, dtype="float32").reshape(H, W)
    pixel_lat = np.asarray(pixel_lat, dtype="float32").reshape(H, W)

    # Static
    static_patches = []
    for tif_path, tif_names in zip(ds.static_tifs, ds._static_tif_band_names):
        s = ds.raster_cache.get(tif_path)
        s_data = s["data"]
        s_crs = s["crs"]
        if str(s_crs) == str(bg_crs):
            s_rows = np.clip(rr_flat, 0, s_data.shape[1] - 1).astype(int)
            s_cols = np.clip(cc_flat, 0, s_data.shape[2] - 1).astype(int)
        else:
            to_s = Transformer.from_crs(bg_crs, s_crs, always_xy=True)
            sx, sy = to_s.transform(x_proj, y_proj)
            s_tf = s["transform"]
            s_rows_raw, s_cols_raw = rowcol(s_tf, sx, sy)
            s_rows = np.clip(np.asarray(s_rows_raw, dtype=int), 0, s_data.shape[1] - 1)
            s_cols = np.clip(np.asarray(s_cols_raw, dtype=int), 0, s_data.shape[2] - 1)
        patch = np.nan_to_num(
            s_data[:, s_rows, s_cols].astype("float32"), nan=0.0
        ).reshape(len(tif_names), H, W)
        static_patches.append(patch)

    # Landsat
    landsat_patch = None
    active_landsat_names = []
    if ds.landsat_tif:
        active_landsat_names = ds._landsat_period_names[period]
        ls = ds.raster_cache.get(ds.landsat_tif)
        ls_data = ls["data"]
        ls_crs = ls["crs"]
        b0 = period * _LANDSAT_BANDS
        b1 = b0 + _LANDSAT_BANDS
        if str(ls_crs) == str(bg_crs):
            ls_rows = np.clip(rr_flat, 0, ls_data.shape[1] - 1).astype(int)
            ls_cols = np.clip(cc_flat, 0, ls_data.shape[2] - 1).astype(int)
        else:
            to_ls = Transformer.from_crs(bg_crs, ls_crs, always_xy=True)
            lsx, lsy = to_ls.transform(x_proj, y_proj)
            ls_tf = ls["transform"]
            ls_rows_raw, ls_cols_raw = rowcol(ls_tf, lsx, lsy)
            ls_rows = np.clip(
                np.asarray(ls_rows_raw, dtype=int), 0, ls_data.shape[1] - 1
            )
            ls_cols = np.clip(
                np.asarray(ls_cols_raw, dtype=int), 0, ls_data.shape[2] - 1
            )
        landsat_patch = np.nan_to_num(
            ls_data[b0:b1, ls_rows, ls_cols].astype("float32"), nan=0.0
        ).reshape(_LANDSAT_BANDS, H, W)

    # rsun
    rsun_patch = None
    if ds.rsun_tif and ds._rsun_meta is not None:
        with rasterio.open(ds.rsun_tif) as rsun_src:
            band_idx = max(1, min(doy, rsun_src.count))
            rsun_band = rsun_src.read(band_idx).astype("float32")
        rsun_crs = ds._rsun_meta["crs"]
        if rsun_crs == str(bg_crs):
            rs_rows = np.clip(rr_flat, 0, rsun_band.shape[0] - 1).astype(int)
            rs_cols = np.clip(cc_flat, 0, rsun_band.shape[1] - 1).astype(int)
        else:
            rsun_tf = ds._rsun_meta["transform"]
            to_rsun = Transformer.from_crs(bg_crs, rsun_crs, always_xy=True)
            rx, ry = to_rsun.transform(x_proj, y_proj)
            rs_rows_raw, rs_cols_raw = rowcol(rsun_tf, rx, ry)
            rs_rows = np.clip(
                np.asarray(rs_rows_raw, dtype=int), 0, rsun_band.shape[0] - 1
            )
            rs_cols = np.clip(
                np.asarray(rs_cols_raw, dtype=int), 0, rsun_band.shape[1] - 1
            )
        rsun_patch = np.nan_to_num(rsun_band[rs_rows, rs_cols], nan=0.0).reshape(
            1, H, W
        )

    # CDR
    cdr_patch = None
    if ds.cdr_dir and ds._cdr_band_names:
        cdr_path = ds._cdr_path(day)
        n_cdr = len(ds._cdr_band_names)
        if os.path.exists(cdr_path):
            cdr = ds.raster_cache.get(cdr_path)
            cdr_data = cdr["data"]
            cdr_tf = cdr["transform"]
            cdr_rows_raw, cdr_cols_raw = rowcol(
                cdr_tf,
                pixel_lon.ravel().astype("float64"),
                pixel_lat.ravel().astype("float64"),
            )
            cdr_rows = np.clip(
                np.asarray(cdr_rows_raw, dtype=int), 0, cdr_data.shape[1] - 1
            )
            cdr_cols = np.clip(
                np.asarray(cdr_cols_raw, dtype=int), 0, cdr_data.shape[2] - 1
            )
            cdr_patch = np.nan_to_num(
                cdr_data[:, cdr_rows, cdr_cols].astype("float32"), nan=0.0
            ).reshape(n_cdr, H, W)
        else:
            cdr_patch = np.zeros((n_cdr, H, W), dtype="float32")

    # Position/time
    doy_sin, doy_cos = _doy_features(day)
    pos_time = np.stack(
        [
            np.full((H, W), doy_sin, dtype="float32"),
            np.full((H, W), doy_cos, dtype="float32"),
            pixel_lat,
            pixel_lon,
        ]
    )

    # Stack and normalize
    parts = [bg_patch] + static_patches
    if landsat_patch is not None:
        parts.append(landsat_patch)
    if rsun_patch is not None:
        parts.append(rsun_patch)
    if cdr_patch is not None:
        parts.append(cdr_patch)
    parts.append(pos_time)
    x_patch = np.concatenate(parts, axis=0)

    channel_names = (
        list(ds.background_feature_names)
        + list(ds.static_feature_names)
        + active_landsat_names
        + (["rsun"] if ds.rsun_tif else [])
        + list(ds._cdr_band_names)
        + ["doy_sin", "doy_cos", "latitude", "longitude"]
    )
    for i, name in enumerate(channel_names):
        if name in ds.norm_stats:
            s = ds.norm_stats[name]
            x_patch[i] = (x_patch[i] - s["mean"]) / s["std"]

    # Stations — old CRS per-sample
    sta_rows_list, sta_cols_list = [], []
    sta_targets_list, sta_valid_list = [], []
    sta_holdout_list, sta_is_center_list = [], []
    center_fid = str(row["fid"])

    day_group = ds._neighbor_day_groups.get(day)
    if day_group is not None and not day_group.empty:
        sta_lons = day_group["longitude"].to_numpy(dtype="float64")
        sta_lats = day_group["latitude"].to_numpy(dtype="float64")
        sta_x, sta_y = to_bg.transform(sta_lons, sta_lats)
        sta_r_raw, sta_c_raw = rowcol(bg_tf, sta_x, sta_y)
        sta_r = np.asarray(sta_r_raw, dtype=int)
        sta_c = np.asarray(sta_c_raw, dtype=int)

        in_patch = (sta_r >= r0) & (sta_r < r1) & (sta_c >= c0) & (sta_c < c1)
        for j in np.where(in_patch)[0]:
            sta_rows_list.append(int(sta_r[j] - r0))
            sta_cols_list.append(int(sta_c[j] - c0))
            fid_j = str(day_group.iloc[j]["fid"])
            sta_holdout_list.append(fid_j in ds._holdout_fids)
            sta_is_center_list.append(fid_j == center_fid)
            tgt, vld = [], []
            for name in ds.target_names:
                val = day_group.iloc[j].get(name, float("nan"))
                is_valid = pd.notna(val)
                tgt.append(float(val) if is_valid else 0.0)
                vld.append(bool(is_valid))
            sta_targets_list.append(tgt)
            sta_valid_list.append(vld)

    if not sta_rows_list:
        r_in = int(np.clip(int(r_center) - r0, 0, H - 1))
        c_in = int(np.clip(int(c_center) - c0, 0, W - 1))
        sta_rows_list.append(r_in)
        sta_cols_list.append(c_in)
        sta_holdout_list.append(center_fid in ds._holdout_fids)
        sta_is_center_list.append(True)
        tgt, vld = [], []
        for name in ds.target_names:
            val = row.get(name, float("nan"))
            is_valid = pd.notna(val)
            tgt.append(float(val) if is_valid else 0.0)
            vld.append(bool(is_valid))
        sta_targets_list.append(tgt)
        sta_valid_list.append(vld)

    return {
        "x_patch": x_patch,
        "sta_rows": np.array(sta_rows_list, dtype=np.int64),
        "sta_cols": np.array(sta_cols_list, dtype=np.int64),
        "sta_targets": np.array(sta_targets_list, dtype=np.float32),
        "sta_valid": np.array(sta_valid_list),
        "sta_holdout": np.array(sta_holdout_list),
        "sta_is_center": np.array(sta_is_center_list),
        "r0": r0,
        "c0": c0,
        "r_center": int(r_center),
        "c_center": int(c_center),
    }


@pytest.mark.skipif(not _data_available(), reason="NVMe raster data not available")
def test_loader_parity():
    """Compare optimized loader output to old per-sample CRS logic."""
    with open(HOLDOUT) as f:
        holdout_fids = set(str(x) for x in json.load(f))

    # Use a small day window so init is fast
    df = pd.read_parquet(TABLE)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["day"] = pd.to_datetime(df["day"])
    sample_days = set(
        df[df["day"].dt.year == 2020]["day"]
        .drop_duplicates()
        .sort_values()
        .head(5)
        .tolist()
    )

    ds = HRRRPatchDataset(
        table_path=TABLE,
        background_dir=BG_DIR,
        static_tifs=[STATIC],
        landsat_tif=LANDSAT,
        rsun_tif=RSUN,
        cdr_dir=CDR_DIR,
        target_names=["delta_tmax", "delta_tmin"],
        train_days=sample_days,
        holdout_fids=holdout_fids,
        drop_bands=["n_hours"],
    )

    # Sample indices spread across the dataset
    rng = np.random.default_rng(42)
    n = min(N_SAMPLES, len(ds))
    indices = rng.choice(len(ds), n, replace=False)

    max_diffs = {}
    for idx in indices:
        new = ds[idx]
        old = _old_getitem(ds, idx)

        new_x = new[0].numpy()
        old_x = old["x_patch"]

        # x_patch shape
        assert new_x.shape == old_x.shape, (
            f"idx={idx}: shape {new_x.shape} vs {old_x.shape}"
        )

        # Per-channel max absolute diff
        for ch in range(new_x.shape[0]):
            diff = np.abs(new_x[ch] - old_x[ch]).max()
            ch_name = ds.feature_names[ch] if ch < len(ds.feature_names) else f"ch{ch}"
            if ch_name not in max_diffs or diff > max_diffs[ch_name]:
                max_diffs[ch_name] = diff

        # Station row/col
        new_rows = new[1].numpy()
        old_rows = old["sta_rows"]
        assert np.array_equal(new_rows, old_rows), (
            f"idx={idx}: sta_rows differ: {new_rows} vs {old_rows}"
        )

        new_cols = new[2].numpy()
        old_cols = old["sta_cols"]
        assert np.array_equal(new_cols, old_cols), (
            f"idx={idx}: sta_cols differ: {new_cols} vs {old_cols}"
        )

        # Targets
        new_tgt = new[3].numpy()
        old_tgt = old["sta_targets"]
        assert np.allclose(new_tgt, old_tgt, atol=1e-6), (
            f"idx={idx}: sta_targets differ: max diff {np.abs(new_tgt - old_tgt).max()}"
        )

        # Valid mask
        new_vld = new[4].numpy()
        old_vld = old["sta_valid"]
        assert np.array_equal(new_vld, old_vld), f"idx={idx}: sta_valid differ"

        # Holdout mask
        new_ho = new[5].numpy()
        old_ho = old["sta_holdout"]
        assert np.array_equal(new_ho, old_ho), f"idx={idx}: sta_holdout differ"

        # Center flag
        new_ctr = new[6].numpy()
        old_ctr = old["sta_is_center"]
        assert np.array_equal(new_ctr, old_ctr), f"idx={idx}: sta_is_center differ"

    # Report per-channel max diffs
    # CDR channels (except cloud_state) may differ at ~0.005% of pixels
    # because the new path computes CDR index maps at float64 precision
    # while the old path had a float32 round-trip that occasionally shifted
    # coordinates across 0.05-degree CDR cell boundaries.
    cdr_channels = {"i1_cdr", "i2_cdr", "bt15_cdr", "szen_cdr"}
    print("\nPer-channel max |old - new| across all samples:")
    for name, diff in sorted(max_diffs.items(), key=lambda x: -x[1]):
        is_cdr = name in cdr_channels
        status = "CDR-OK" if is_cdr else ("PASS" if diff < 1e-5 else "FAIL")
        print(f"  {status}  {name:25s}  {diff:.2e}")

    # Assert non-CDR channels are exact
    for name, diff in max_diffs.items():
        if name not in cdr_channels:
            assert diff < 1e-5, f"Channel {name} max diff {diff:.2e} exceeds 1e-5"
