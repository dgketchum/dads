"""
Build the full PRISM terrain raster stack for the PNW 1 km domain.

Reads the source DEM, applies the PRISM preprocessing pipeline from
``terrain.prism_topography``, and writes all output GeoTIFFs plus a
metadata JSON sidecar to the output directory.

Usage
-----
::

    uv run python -m prep.build_prism_terrain [options]

See ``--help`` for all options.

References
----------
Daly, C. et al. 2008. Physiographically sensitive mapping of climatological
temperature and precipitation across the conterminous United States.
Int. J. Climatol. 28, 2031–2064.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os

import numpy as np
import rasterio

from prep.paths import MVP_ROOT
from prep.pnw_1km_grid import PNW_1KM_CRS, PNW_1KM_SHAPE, PNW_1KM_TRANSFORM
from terrain.prism_topography import (
    FACET_MIN_CELLS,
    FACET_SCALES_KM,
    ETH_MIN_RADIUS_M,
    ETH_BASE_SMOOTH_RADIUS_M,
    ETH_FINAL_SMOOTH_RADIUS_M,
    H2_DEFAULT,
    H3_DEFAULT,
    I3A_RADIUS_M,
    PRECIPITATION_DEM_RADIUS_M,
    TEMPERATURE_DEM_RADIUS_M,
    build_effective_terrain_height,
    build_facet_hierarchy,
    build_i3a,
    build_i3c,
    build_i3d,
    build_precipitation_dem,
    build_temperature_dem,
    facet_counts_per_scale,
)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_DEFAULT_DEM_TIF = "/nas/dads/mvp/dem_pnw_1km.tif"

# ---------------------------------------------------------------------------
# GeoTIFF I/O helpers (pattern from terrain/grid.py)
# ---------------------------------------------------------------------------

_FLOAT_PROFILE = {
    "driver": "GTiff",
    "dtype": "float32",
    "crs": PNW_1KM_CRS,
    "transform": PNW_1KM_TRANSFORM,
    "compress": "lzw",
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
    "nodata": float("nan"),
}

_INT32_PROFILE = {
    "driver": "GTiff",
    "dtype": "int32",
    "crs": PNW_1KM_CRS,
    "transform": PNW_1KM_TRANSFORM,
    "compress": "lzw",
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
    "nodata": 0,
}

_UINT8_PROFILE = {
    "driver": "GTiff",
    "dtype": "uint8",
    "crs": PNW_1KM_CRS,
    "transform": PNW_1KM_TRANSFORM,
    "compress": "lzw",
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
    "nodata": 0,
}


def _write_single(
    path: str,
    arr: np.ndarray,
    profile_base: dict,
    description: str,
    overwrite: bool,
) -> None:
    """Write a single-band GeoTIFF."""
    if os.path.exists(path) and not overwrite:
        print(f"  skip (exists): {path}")
        return
    H, W = arr.shape
    profile = dict(profile_base, count=1, height=H, width=W)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, description)
    print(f"  written: {path}")


def _write_multiband(
    path: str,
    stack: np.ndarray,
    profile_base: dict,
    descriptions: list[str],
    overwrite: bool,
) -> None:
    """Write a multi-band GeoTIFF from a (N, H, W) stack."""
    if os.path.exists(path) and not overwrite:
        print(f"  skip (exists): {path}")
        return
    N, H, W = stack.shape
    profile = dict(profile_base, count=N, height=H, width=W)
    with rasterio.open(path, "w", **profile) as dst:
        for i, (band, desc) in enumerate(zip(stack, descriptions), 1):
            dst.write(band, i)
            dst.set_band_description(i, desc)
    print(f"  written: {path}  ({N} bands)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_prism_terrain(
    dem_tif: str = _DEFAULT_DEM_TIF,
    out_dir: str = MVP_ROOT,
    flat_slope_deg: float = 2.0,
    h2: float = H2_DEFAULT,
    h3: float = H3_DEFAULT,
    debug: bool = False,
    overwrite: bool = False,
) -> None:
    """Build the full PRISM terrain raster stack and write outputs.

    Parameters
    ----------
    dem_tif : str
        Path to source DEM GeoTIFF (single-band Int16, on PNW 1 km grid).
    out_dir : str
        Output directory for all GeoTIFFs and the metadata JSON.
    flat_slope_deg : float
        Flat threshold in degrees for aspect classification.
    h2 : float
        2D effective terrain height threshold (metres).
    h3 : float
        3D effective terrain height threshold (metres).
    debug : bool
        Write intermediate QA rasters (smoothed DEMs, raw aspect classes).
    overwrite : bool
        Overwrite existing output files.
    """
    os.makedirs(out_dir, exist_ok=True)
    cell_size_m = 1_000.0

    # ------------------------------------------------------------------
    # Step 1 — Read source DEM
    # ------------------------------------------------------------------
    print(f"[1/7] Reading source DEM: {dem_tif}")
    with rasterio.open(dem_tif) as src:
        raw = src.read(1)
        nodata = src.nodata

    dem = raw.astype("float32")
    if nodata is not None:
        dem[raw == nodata] = np.nan
    H, W = dem.shape
    valid_px = int(np.isfinite(dem).sum())
    print(f"  shape={H}x{W}, valid pixels={valid_px:,}")

    # ------------------------------------------------------------------
    # Step 2 — Temperature DEM  (1.3 km Gaussian)
    # ------------------------------------------------------------------
    print(
        f"[2/7] Building temperature DEM (radius={TEMPERATURE_DEM_RADIUS_M / 1e3} km) …"
    )
    temp_dem = build_temperature_dem(dem, cell_size_m=cell_size_m)

    out_temp_dem = os.path.join(out_dir, "prism_temperature_dem_pnw_1km.tif")
    _write_single(
        out_temp_dem,
        temp_dem,
        _FLOAT_PROFILE,
        f"temperature_dem_r{TEMPERATURE_DEM_RADIUS_M / 1e3:.1f}km",
        overwrite,
    )

    if debug:
        _write_single(
            os.path.join(out_dir, "debug_temperature_dem_raw_pnw_1km.tif"),
            dem,
            _FLOAT_PROFILE,
            "raw_dem",
            overwrite,
        )

    # ------------------------------------------------------------------
    # Step 3 — Precipitation DEM  (additional 7 km Gaussian)
    # ------------------------------------------------------------------
    print(
        f"[3/7] Building precipitation DEM (radius={PRECIPITATION_DEM_RADIUS_M / 1e3} km) …"
    )
    precip_dem = build_precipitation_dem(temp_dem, cell_size_m=cell_size_m)

    out_precip_dem = os.path.join(out_dir, "prism_precipitation_dem_pnw_1km.tif")
    _write_single(
        out_precip_dem,
        precip_dem,
        _FLOAT_PROFILE,
        f"precipitation_dem_r{PRECIPITATION_DEM_RADIUS_M / 1e3:.1f}km",
        overwrite,
    )

    # ------------------------------------------------------------------
    # Step 4 — Effective terrain height + I3c / I3a / I3d
    # ------------------------------------------------------------------
    print("[4/7] Building effective terrain height and I3 indices …")
    eth = build_effective_terrain_height(
        precip_dem,
        cell_size_m=cell_size_m,
        min_radius_m=ETH_MIN_RADIUS_M,
        base_smooth_radius_m=ETH_BASE_SMOOTH_RADIUS_M,
        final_smooth_radius_m=ETH_FINAL_SMOOTH_RADIUS_M,
        verbose=True,
    )

    print("  building I3c …")
    i3c = build_i3c(eth, h2=h2, h3=h3)
    print("  building I3a (100 km IDW — may take a moment) …")
    i3a = build_i3a(eth, cell_size_m=cell_size_m, h2=h2, h3=h3)
    print("  building I3d = max(I3c, I3a) …")
    i3d = build_i3d(i3c, i3a)

    _write_single(
        os.path.join(out_dir, "prism_effective_terrain_height_pnw_1km.tif"),
        eth,
        _FLOAT_PROFILE,
        "effective_terrain_height_m",
        overwrite,
    )
    _write_single(
        os.path.join(out_dir, "prism_effective_terrain_i3c_pnw_1km.tif"),
        i3c,
        _FLOAT_PROFILE,
        "I3c_cell_local",
        overwrite,
    )
    _write_single(
        os.path.join(out_dir, "prism_effective_terrain_i3a_pnw_1km.tif"),
        i3a,
        _FLOAT_PROFILE,
        "I3a_areal_support",
        overwrite,
    )
    _write_single(
        os.path.join(out_dir, "prism_effective_terrain_i3d_pnw_1km.tif"),
        i3d,
        _FLOAT_PROFILE,
        "I3d_final",
        overwrite,
    )

    # ------------------------------------------------------------------
    # Step 5 — Facet hierarchy (6 scales)
    # ------------------------------------------------------------------
    print("[5/7] Building 6-scale facet hierarchy …")
    facet_ids, facet_orients = build_facet_hierarchy(
        precip_dem,
        flat_slope_deg=flat_slope_deg,
        scales_km=FACET_SCALES_KM,
        min_facet_cells=FACET_MIN_CELLS,
        cell_size_m=cell_size_m,
    )

    # ------------------------------------------------------------------
    # Step 6 — Write GeoTIFFs
    # ------------------------------------------------------------------
    print("[6/7] Writing facet rasters …")
    facet_id_descs = [f"facet_id_r{km}km" for km in FACET_SCALES_KM]
    facet_ori_descs = [f"facet_orient_r{km}km" for km in FACET_SCALES_KM]

    _write_multiband(
        os.path.join(out_dir, "prism_facet_id_pnw_1km.tif"),
        facet_ids,
        _INT32_PROFILE,
        facet_id_descs,
        overwrite,
    )
    _write_multiband(
        os.path.join(out_dir, "prism_facet_orientation_pnw_1km.tif"),
        facet_orients,
        _UINT8_PROFILE,
        facet_ori_descs,
        overwrite,
    )

    if debug:
        # Write per-scale smoothed DEMs and raw aspect classes
        from terrain.prism_topography import (
            classify_aspect,
            gaussian_smooth,
            horns_slope_aspect,
        )

        for i, scale_km in enumerate(FACET_SCALES_KM):
            radius_m = scale_km * 1_000.0
            smoothed = gaussian_smooth(precip_dem, radius_m, cell_size_m)
            slope_deg, aspect_deg = horns_slope_aspect(smoothed, cell_size_m)
            orient_raw = classify_aspect(aspect_deg, slope_deg, flat_slope_deg)

            _write_single(
                os.path.join(out_dir, f"debug_smoothed_dem_r{scale_km}km_pnw_1km.tif"),
                smoothed,
                _FLOAT_PROFILE,
                f"smoothed_dem_r{scale_km}km",
                overwrite,
            )
            _write_single(
                os.path.join(out_dir, f"debug_aspect_class_r{scale_km}km_pnw_1km.tif"),
                orient_raw,
                _UINT8_PROFILE,
                f"aspect_class_r{scale_km}km",
                overwrite,
            )

    # ------------------------------------------------------------------
    # Step 7 — Write metadata JSON
    # ------------------------------------------------------------------
    print("[7/7] Writing metadata JSON …")
    n_facets_per_scale = facet_counts_per_scale(facet_ids)

    meta = {
        "build_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "source_dem": dem_tif,
        "output_crs": str(PNW_1KM_CRS),
        "output_transform": list(PNW_1KM_TRANSFORM),
        "output_shape_hw": list(PNW_1KM_SHAPE),
        "cell_size_m": cell_size_m,
        "temperature_dem_radius_m": TEMPERATURE_DEM_RADIUS_M,
        "precipitation_dem_radius_m": PRECIPITATION_DEM_RADIUS_M,
        "facet_scales_km": FACET_SCALES_KM,
        "facet_min_cells": FACET_MIN_CELLS,
        "facet_flat_slope_deg": flat_slope_deg,
        "n_facets_per_scale": dict(
            zip([str(km) for km in FACET_SCALES_KM], n_facets_per_scale)
        ),
        "eth_min_filter_radius_m": ETH_MIN_RADIUS_M,
        "eth_base_smooth_radius_m": ETH_BASE_SMOOTH_RADIUS_M,
        "eth_final_smooth_radius_m": ETH_FINAL_SMOOTH_RADIUS_M,
        "h2_threshold_m": h2,
        "h3_threshold_m": h3,
        "i3a_support_radius_m": I3A_RADIUS_M,
    }

    meta_path = os.path.join(out_dir, "prism_terrain_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  written: {meta_path}")

    print("\nPRISM terrain stack complete.")
    print(f"  outputs in: {out_dir}")
    print(f"  facets per scale: {dict(zip(FACET_SCALES_KM, n_facets_per_scale))}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build PRISM terrain raster stack for the PNW 1 km domain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dem-tif",
        default=_DEFAULT_DEM_TIF,
        help="Source DEM GeoTIFF (single-band Int16, PNW 1 km grid)",
    )
    p.add_argument(
        "--out-dir",
        default=MVP_ROOT,
        help="Output directory for all GeoTIFFs and metadata JSON",
    )
    p.add_argument(
        "--flat-slope",
        type=float,
        default=2.0,
        dest="flat_slope",
        help="Flat threshold in degrees for aspect classification",
    )
    p.add_argument(
        "--h2",
        type=float,
        default=H2_DEFAULT,
        help="2D effective terrain height threshold (metres)",
    )
    p.add_argument(
        "--h3",
        type=float,
        default=H3_DEFAULT,
        help="3D effective terrain height threshold (metres)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Write intermediate QA rasters (smoothed DEMs, raw aspect classes)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    build_prism_terrain(
        dem_tif=a.dem_tif,
        out_dir=a.out_dir,
        flat_slope_deg=a.flat_slope,
        h2=a.h2,
        h3=a.h3,
        debug=a.debug,
        overwrite=a.overwrite,
    )


if __name__ == "__main__":
    main()
