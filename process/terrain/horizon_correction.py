"""Horizon-aware r.sun correction for PNW MGRS tiles.

Three modes:
  validate  — single-tile validation (full 365 vs sparse-interpolated)
  correct   — apply correction to 180 existing tiles via sparse DOY ratios
  generate  — run 21 missing tiles from scratch with horizons
"""

import argparse
import os
import subprocess
import tempfile
from multiprocessing import Pool

import numpy as np
import rasterio
from scipy.interpolate import interp1d

from process.terrain.sun import compute_horizon, worker_calculate_single_tile_irradiance

# 24 DOYs spanning the year, roughly every 15 days
SPARSE_DOYS = [
    1,
    15,
    32,
    46,
    60,
    74,
    91,
    105,
    121,
    135,
    152,
    166,
    182,
    196,
    213,
    227,
    244,
    258,
    274,
    288,
    305,
    319,
    335,
    349,
]

DEM_DIR = "/data/ssd2/dads/dem/dem_5071"
TERRAIN_DIR = "/data/ssd2/dads/dem"
RSUN_DIR = "/nas/dads/dem/rsun_irradiance"
MAPSET = "dads_map"
HORIZON_STEP = 5
HORIZON_MAXDIST = 50000

# PNW tiles with existing rsun rasters
PNW_EXISTING = [
    "10TCM",
    "10TCN",
    "10TCP",
    "10TCT",
    "10TDM",
    "10TDN",
    "10TDP",
    "10TDQ",
    "10TDR",
    "10TDS",
    "10TDT",
    "10TEM",
    "10TEN",
    "10TEP",
    "10TEQ",
    "10TER",
    "10TES",
    "10TET",
    "10TFM",
    "10TFN",
    "10TFP",
    "10TFQ",
    "10TFR",
    "10TFS",
    "10TFT",
    "10TGM",
    "10TGN",
    "10TGP",
    "10TGQ",
    "10TGR",
    "10TGS",
    "10TGT",
    "10UCU",
    "10UCV",
    "10UDU",
    "10UDV",
    "10UEU",
    "10UEV",
    "10UFU",
    "10UFV",
    "10UGU",
    "11TKG",
    "11TKH",
    "11TKJ",
    "11TKK",
    "11TKL",
    "11TKM",
    "11TKN",
    "11TLG",
    "11TLH",
    "11TLJ",
    "11TLK",
    "11TLL",
    "11TLM",
    "11TLN",
    "11TMG",
    "11TMH",
    "11TMJ",
    "11TMK",
    "11TML",
    "11TMM",
    "11TMN",
    "11TNG",
    "11TNH",
    "11TNJ",
    "11TNK",
    "11TNL",
    "11TNM",
    "11TNN",
    "11TPG",
    "11TPH",
    "11TPJ",
    "11TPK",
    "11TPL",
    "11TPM",
    "11TPN",
    "11TQG",
    "11TQH",
    "11TQJ",
    "11TQK",
    "11TQL",
    "11TQM",
    "11TQN",
    "11UKP",
    "11ULP",
    "11ULQ",
    "11UMP",
    "11UMQ",
    "11UNP",
    "11UNQ",
    "11UPP",
    "11UPQ",
    "11UQP",
    "12TTM",
    "12TTN",
    "12TTP",
    "12TTQ",
    "12TTR",
    "12TTS",
    "12TTT",
    "12TUM",
    "12TUN",
    "12TUP",
    "12TUQ",
    "12TUR",
    "12TUS",
    "12TUT",
    "12TVM",
    "12TVN",
    "12TVP",
    "12TVQ",
    "12TVR",
    "12TVS",
    "12TVT",
    "12TWM",
    "12TWN",
    "12TWP",
    "12TWQ",
    "12TWR",
    "12TWS",
    "12TWT",
    "12TXM",
    "12TXN",
    "12TXP",
    "12TXQ",
    "12TXR",
    "12TXS",
    "12TXT",
    "12TYN",
    "12TYP",
    "12TYQ",
    "12TYR",
    "12TYS",
    "12TYT",
    "12UTU",
    "12UTV",
    "12UUU",
    "12UUV",
    "12UVU",
    "12UVV",
    "12UWU",
    "12UWV",
    "12UXU",
    "12UXV",
    "12UYU",
    "13TBG",
    "13TBH",
    "13TBJ",
    "13TBK",
    "13TBL",
    "13TBM",
    "13TBN",
    "13TCG",
    "13TCH",
    "13TCJ",
    "13TCK",
    "13TCL",
    "13TCM",
    "13TCN",
    "13TDG",
    "13TDH",
    "13TDJ",
    "13TDK",
    "13TDL",
    "13TDM",
    "13TDN",
    "13TEG",
    "13TEH",
    "13TEJ",
    "13TEK",
    "13TEL",
    "13TEM",
    "13TEN",
    "13UBP",
    "13UCP",
    "13UCQ",
    "13UDP",
    "13UDQ",
    "13UEP",
    "13UEQ",
]

# PNW tiles missing rsun — all have DEMs
PNW_MISSING = [
    "10TCQ",
    "10TCU",
    "10TDU",
    "10TEU",
    "10TFU",
    "10TGU",
    "11TKP",
    "11TLP",
    "11TMP",
    "11TNP",
    "11TPP",
    "11TQP",
    "12TTU",
    "12TUU",
    "12TVU",
    "12TWU",
    "12TXU",
    "12TYU",
    "13TBP",
    "13TCP",
    "13TEP",
]


def _grass_export_tif(grass_name, mapset, out_path):
    """Export a GRASS raster to a LZW-compressed GeoTIFF."""
    subprocess.run(
        [
            "r.out.gdal",
            "-c",
            f"input={grass_name}@{mapset}",
            "format=GTiff",
            "createopt=COMPRESS=LZW",
            "--overwrite",
            f"output={out_path}",
        ],
        check=True,
    )


def _grass_remove(names):
    """Remove GRASS rasters by name list."""
    if names:
        subprocess.run(
            ["g.remove", "-f", "type=raster", f"name={','.join(names)}"], check=True
        )


def _read_tif(path):
    """Read a single-band GeoTIFF, return (data, profile)."""
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32), src.profile.copy()


def _write_tif(path, data, profile):
    """Write a single-band GeoTIFF."""
    profile.update(dtype="float32", count=1, compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)


def _interpolate_ratios(sparse_doys, ratio_stack, all_doys=range(1, 366)):
    """Interpolate per-pixel ratios from sparse DOYs to all 365.

    Args:
        sparse_doys: list of int DOYs with known ratios
        ratio_stack: (n_sparse, H, W) array of ratios
        all_doys: target DOYs

    Returns:
        (365, H, W) interpolated ratio array
    """
    n_sparse, h, w = ratio_stack.shape
    all_doys = np.array(list(all_doys))
    sparse_arr = np.array(sparse_doys)

    flat = ratio_stack.reshape(n_sparse, -1)
    interp_func = interp1d(
        sparse_arr,
        flat,
        axis=0,
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate",
    )
    interp_flat = interp_func(all_doys)
    interp_flat = np.clip(interp_flat, 0.0, 1.0)
    return interp_flat.reshape(len(all_doys), h, w)


# ---------------------------------------------------------------------------
# Mode A: validate
# ---------------------------------------------------------------------------


def run_validate(tile, mapset=MAPSET):
    """Single-tile validation: full 365 vs sparse-interpolated horizon correction."""
    dem_name = f"dem_{tile}"
    hz_base = f"horizon_{tile}"

    print(f"[validate] {tile}: computing horizons ...")
    compute_horizon(
        dem_name, mapset, hz_base, step=HORIZON_STEP, maxdistance=HORIZON_MAXDIST
    )

    dem_file = os.path.join(DEM_DIR, f"dem_{tile}.tif")
    tmpdir = tempfile.mkdtemp(prefix=f"hz_val_{tile}_")

    # --- run r.sun WITH horizons for all 365 DOYs ---
    print(f"[validate] {tile}: running r.sun with horizons (365 DOYs) ...")
    worker_calculate_single_tile_irradiance(
        (
            dem_name,
            dem_file,
            TERRAIN_DIR,
            mapset,
            4,
            True,
            None,
            hz_base,
            HORIZON_STEP,
        )
    )

    # export full-365 horizon-aware results
    hz_full_dir = os.path.join(tmpdir, "hz_full")
    os.makedirs(hz_full_dir)
    for doy in range(1, 366):
        grass_name = f"irradiance_day_{doy}_{tile}"
        out = os.path.join(hz_full_dir, f"irradiance_day_{doy}_{tile}.tif")
        _grass_export_tif(grass_name, mapset, out)

    # --- read existing no-horizon baseline from NAS ---
    existing_dir = os.path.join(RSUN_DIR, tile)

    # --- compute ratios for sparse DOYs ---
    print(f"[validate] {tile}: computing sparse ratios ...")
    ratio_list = []
    for doy in SPARSE_DOYS:
        hz_path = os.path.join(hz_full_dir, f"irradiance_day_{doy}_{tile}.tif")
        nohz_path = os.path.join(existing_dir, f"irradiance_day_{doy}_{tile}.tif")
        hz_data, _ = _read_tif(hz_path)
        nohz_data, _ = _read_tif(nohz_path)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(nohz_data > 0, hz_data / nohz_data, 1.0)
        ratio = np.clip(ratio, 0.0, 1.0).astype(np.float32)
        ratio_list.append(ratio)

    ratio_stack = np.stack(ratio_list, axis=0)
    interp_ratios = _interpolate_ratios(SPARSE_DOYS, ratio_stack)

    # --- compare interpolated vs full-365 ---
    print(f"[validate] {tile}: comparing interpolated vs full-365 ...")
    max_ratio_err = []
    mean_ratio_err = []
    max_abs_err = []
    mean_abs_err = []
    for i, doy in enumerate(range(1, 366)):
        hz_path = os.path.join(hz_full_dir, f"irradiance_day_{doy}_{tile}.tif")
        nohz_path = os.path.join(existing_dir, f"irradiance_day_{doy}_{tile}.tif")
        hz_data, _ = _read_tif(hz_path)
        nohz_data, _ = _read_tif(nohz_path)

        # "truth" ratio from full run
        with np.errstate(divide="ignore", invalid="ignore"):
            true_ratio = np.where(nohz_data > 0, hz_data / nohz_data, 1.0)
        true_ratio = np.clip(true_ratio, 0.0, 1.0)

        ratio_err = np.abs(interp_ratios[i] - true_ratio)
        max_ratio_err.append(np.max(ratio_err))
        mean_ratio_err.append(np.mean(ratio_err))

        # absolute error in Wh/m² (what actually matters)
        truth_irrad = nohz_data * true_ratio
        interp_irrad = nohz_data * interp_ratios[i]
        abs_err = np.abs(interp_irrad - truth_irrad)
        max_abs_err.append(np.max(abs_err))
        mean_abs_err.append(np.mean(abs_err))

    max_ratio_err = np.array(max_ratio_err)
    mean_ratio_err = np.array(mean_ratio_err)
    max_abs_err = np.array(max_abs_err)
    mean_abs_err = np.array(mean_abs_err)

    print(f"[validate] {tile}: --- ratio error (unitless) ---")
    print(f"[validate] {tile}: max ratio error = {max_ratio_err.max():.4f}")
    print(f"[validate] {tile}: mean ratio error = {mean_ratio_err.mean():.6f}")
    print(
        f"[validate] {tile}: p99 max ratio error = {np.percentile(max_ratio_err, 99):.4f}"
    )
    print(f"[validate] {tile}: worst ratio DOY = {np.argmax(max_ratio_err) + 1}")
    print(f"[validate] {tile}: --- absolute error (Wh/m²) ---")
    print(f"[validate] {tile}: max abs error = {max_abs_err.max():.1f}")
    print(f"[validate] {tile}: mean abs error = {mean_abs_err.mean():.2f}")
    print(
        f"[validate] {tile}: p99 max abs error = {np.percentile(max_abs_err, 99):.1f}"
    )
    print(f"[validate] {tile}: worst abs DOY = {np.argmax(max_abs_err) + 1}")
    print(f"[validate] temp files in {tmpdir}")

    # clean up GRASS horizon rasters
    _cleanup_horizon_grass(tile)


def _cleanup_horizon_grass(tile):
    """Remove horizon rasters for a tile from GRASS."""
    hz_base = f"horizon_{tile}"
    result = subprocess.run(
        ["g.list", "type=raster", f"pattern={hz_base}_*"],
        capture_output=True,
        text=True,
    )
    names = [n.strip() for n in result.stdout.strip().split("\n") if n.strip()]
    if names:
        _grass_remove(names)


# ---------------------------------------------------------------------------
# Mode B: correct — apply horizon ratios to existing tiles
# ---------------------------------------------------------------------------


def _correct_single_tile(args):
    """Worker: correct one existing tile's rsun rasters with horizon ratios."""
    tile, mapset, nprocs = args
    dem_name = f"dem_{tile}"
    hz_base = f"horizon_{tile}"
    dem_file = os.path.join(DEM_DIR, f"dem_{tile}.tif")
    existing_dir = os.path.join(RSUN_DIR, tile)

    print(f"[correct] {tile}: computing horizons ...")
    compute_horizon(
        dem_name, mapset, hz_base, step=HORIZON_STEP, maxdistance=HORIZON_MAXDIST
    )

    # run r.sun WITH horizons for sparse DOYs only
    print(
        f"[correct] {tile}: running r.sun with horizons ({len(SPARSE_DOYS)} DOYs) ..."
    )
    worker_calculate_single_tile_irradiance(
        (
            dem_name,
            dem_file,
            TERRAIN_DIR,
            mapset,
            nprocs,
            True,
            SPARSE_DOYS,
            hz_base,
            HORIZON_STEP,
        )
    )

    # export sparse horizon results to temp dir and compute ratios
    tmpdir = tempfile.mkdtemp(prefix=f"hz_corr_{tile}_")
    ratio_list = []
    sparse_grass_names = []

    for doy in SPARSE_DOYS:
        grass_name = f"irradiance_day_{doy}_{tile}"
        sparse_grass_names.append(grass_name)
        hz_tif = os.path.join(tmpdir, f"hz_{doy}_{tile}.tif")
        _grass_export_tif(grass_name, mapset, hz_tif)

        nohz_path = os.path.join(existing_dir, f"irradiance_day_{doy}_{tile}.tif")
        hz_data, _ = _read_tif(hz_tif)
        nohz_data, _ = _read_tif(nohz_path)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(nohz_data > 0, hz_data / nohz_data, 1.0)
        ratio = np.clip(ratio, 0.0, 1.0).astype(np.float32)
        ratio_list.append(ratio)

    ratio_stack = np.stack(ratio_list, axis=0)
    interp_ratios = _interpolate_ratios(SPARSE_DOYS, ratio_stack)

    # apply ratios to all 365 existing rasters (overwrite in place)
    print(f"[correct] {tile}: applying interpolated ratios to 365 DOYs ...")
    for i, doy in enumerate(range(1, 366)):
        tif_path = os.path.join(existing_dir, f"irradiance_day_{doy}_{tile}.tif")
        data, profile = _read_tif(tif_path)
        corrected = data * interp_ratios[i]
        _write_tif(tif_path, corrected, profile)

    # cleanup
    _grass_remove(sparse_grass_names)
    _cleanup_horizon_grass(tile)
    subprocess.run(["rm", "-rf", tmpdir], check=True)

    print(f"[correct] {tile}: done")
    return tile


def run_correct(num_tiles=4, nprocs_per_tile=4, tiles=None, mapset=MAPSET):
    """Apply horizon correction to existing PNW tiles."""
    target = tiles if tiles else PNW_EXISTING
    args_list = [(t, mapset, nprocs_per_tile) for t in target]
    print(f"[correct] {len(args_list)} tiles, {num_tiles} parallel")

    with Pool(processes=num_tiles) as pool:
        for result in pool.imap_unordered(_correct_single_tile, args_list):
            print(f"[correct] completed: {result}")


# ---------------------------------------------------------------------------
# Mode C: generate — full run for missing tiles
# ---------------------------------------------------------------------------


def _generate_single_tile(args):
    """Worker: generate rsun from scratch with horizons for a missing tile."""
    tile, mapset, nprocs = args
    dem_name = f"dem_{tile}"
    dem_file = os.path.join(DEM_DIR, f"dem_{tile}.tif")
    hz_base = f"horizon_{tile}"
    tile_out = os.path.join(RSUN_DIR, tile)
    os.makedirs(tile_out, exist_ok=True)

    # ingest DEM if not in GRASS
    result = subprocess.run(
        ["g.list", "type=raster", f"pattern={dem_name}"],
        capture_output=True,
        text=True,
    )
    if dem_name not in result.stdout:
        print(f"[generate] {tile}: ingesting DEM ...")
        subprocess.run(
            [
                "r.in.gdal",
                f"input={dem_file}",
                f"output={dem_name}",
                "--overwrite",
            ],
            check=True,
        )

    print(f"[generate] {tile}: computing horizons ...")
    compute_horizon(
        dem_name, mapset, hz_base, step=HORIZON_STEP, maxdistance=HORIZON_MAXDIST
    )

    # run r.sun with horizons for all 365 DOYs
    print(f"[generate] {tile}: running r.sun with horizons (365 DOYs) ...")
    worker_calculate_single_tile_irradiance(
        (
            dem_name,
            dem_file,
            TERRAIN_DIR,
            mapset,
            nprocs,
            True,
            None,
            hz_base,
            HORIZON_STEP,
        )
    )

    # export all 365 DOYs
    print(f"[generate] {tile}: exporting ...")
    subprocess.run(["g.region", f"rast={dem_name}@{mapset}"], check=True)
    for doy in range(1, 366):
        grass_name = f"irradiance_day_{doy}_{tile}"
        out = os.path.join(tile_out, f"irradiance_day_{doy}_{tile}.tif")
        _grass_export_tif(grass_name, mapset, out)

    # cleanup
    _cleanup_horizon_grass(tile)
    print(f"[generate] {tile}: done")
    return tile


def run_generate(num_tiles=4, nprocs_per_tile=4, tiles=None, mapset=MAPSET):
    """Generate rsun with horizons for missing PNW tiles."""
    target = tiles if tiles else PNW_MISSING
    args_list = [(t, mapset, nprocs_per_tile) for t in target]
    print(f"[generate] {len(args_list)} tiles, {num_tiles} parallel")

    with Pool(processes=num_tiles) as pool:
        for result in pool.imap_unordered(_generate_single_tile, args_list):
            print(f"[generate] completed: {result}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Horizon-aware r.sun correction")
    sub = parser.add_subparsers(dest="mode", required=True)

    # validate
    p_val = sub.add_parser("validate", help="Single-tile validation")
    p_val.add_argument("--tile", required=True, help="MGRS tile ID (e.g. 11TML)")
    p_val.add_argument("--mapset", default=MAPSET)

    # correct
    p_corr = sub.add_parser("correct", help="Apply correction to existing tiles")
    p_corr.add_argument("--num-tiles", type=int, default=4, help="Parallel tiles")
    p_corr.add_argument("--nprocs", type=int, default=4, help="r.sun nprocs per tile")
    p_corr.add_argument("--tiles", nargs="+", default=None, help="Specific tiles")
    p_corr.add_argument("--mapset", default=MAPSET)

    # generate
    p_gen = sub.add_parser("generate", help="Generate missing tiles from scratch")
    p_gen.add_argument("--num-tiles", type=int, default=4, help="Parallel tiles")
    p_gen.add_argument("--nprocs", type=int, default=4, help="r.sun nprocs per tile")
    p_gen.add_argument("--tiles", nargs="+", default=None, help="Specific tiles")
    p_gen.add_argument("--mapset", default=MAPSET)

    args = parser.parse_args()

    if args.mode == "validate":
        run_validate(args.tile, mapset=args.mapset)
    elif args.mode == "correct":
        run_correct(
            num_tiles=args.num_tiles,
            nprocs_per_tile=args.nprocs,
            tiles=args.tiles,
            mapset=args.mapset,
        )
    elif args.mode == "generate":
        run_generate(
            num_tiles=args.num_tiles,
            nprocs_per_tile=args.nprocs,
            tiles=args.tiles,
            mapset=args.mapset,
        )


if __name__ == "__main__":
    main()

# ========================= EOF ====================================================================
