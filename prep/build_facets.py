"""
Build PRISM-style topographic facets for the PNW domain.

Uses GRASS GIS i.segment to region-grow on aspect_sin and aspect_cos,
producing contiguous terrain facets with coherent slope orientation.
Flat areas (slope < threshold) are zeroed so they cluster together.

Expects the pnw_1km GRASS location to already contain dem_pnw.

Outputs
-------
facets_pnw_1km.tif             Integer facet IDs (1-based)
facet_orientation_pnw_1km.tif  Dominant orientation class per facet
    0=flat  1=N  2=NE  3=E  4=SE  5=S  6=SW  7=W  8=NW
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import numpy as np
import rasterio

from prep.paths import MVP_ROOT

GRASS_DB = "/data/ssd2/dads/dem/grassdata"
GRASS_LOC = "pnw_1km"
GRASS_MAPSET = "PERMANENT"
GRASS_SESSION = f"{GRASS_DB}/{GRASS_LOC}/{GRASS_MAPSET}"

ORIENT_NAMES = {
    0: "flat",
    1: "N",
    2: "NE",
    3: "E",
    4: "SE",
    5: "S",
    6: "SW",
    7: "W",
    8: "NW",
}


def _grass(*args: str) -> str:
    """Run a GRASS command in the pnw_1km location."""
    cmd = ["grass", GRASS_SESSION, "--exec"] + [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"GRASS failed: {' '.join(str(a) for a in args)}\n{r.stderr}"
        )
    return r.stdout


def _cleanup_grass() -> None:
    """Remove temporary GRASS rasters created by this script."""
    names = [
        "_facet_slope",
        "_facet_aspect",
        "_facet_asp_sin",
        "_facet_asp_cos",
        "_facet_segs",
    ]
    for name in names:
        try:
            _grass("g.remove", "-f", "type=raster", f"name={name}")
        except RuntimeError:
            pass
    try:
        _grass("g.remove", "-f", "type=group", "name=_facet_grp")
    except RuntimeError:
        pass


def build_facets(
    threshold: float = 0.3,
    minsize: int = 20,
    flat_slope_deg: float = 2.0,
    out_dir: str = MVP_ROOT,
    overwrite: bool = False,
) -> str:
    """Build topographic facets via GRASS i.segment and export as GeoTIFF.

    Parameters
    ----------
    threshold : float
        i.segment similarity threshold (0–1). Lower → more / smaller facets.
    minsize : int
        Minimum facet size in cells (= km² at 1 km resolution).
    flat_slope_deg : float
        Slope below which cells are treated as flat (aspect ignored).
    out_dir : str
        Directory for output GeoTIFFs.
    overwrite : bool
        Re-run even if outputs exist.

    Returns
    -------
    str  Path to the facet-ID raster.
    """
    out_facets = os.path.join(out_dir, "facets_pnw_1km.tif")
    out_orient = os.path.join(out_dir, "facet_orientation_pnw_1km.tif")

    if os.path.exists(out_facets) and not overwrite:
        print(f"Output exists: {out_facets}  (use --overwrite)")
        return out_facets

    # ── GRASS: slope, aspect, sin/cos ──────────────────────────────────

    print("Computing slope/aspect from dem_pnw …")
    _grass(
        "r.slope.aspect",
        "--overwrite",
        "elevation=dem_pnw",
        "slope=_facet_slope",
        "aspect=_facet_aspect",
    )

    # GRASS aspect is degrees CCW from east.  sin/cos are convention-
    # agnostic for clustering; we convert to geographic in post-processing.
    # Flat cells get sin=cos=0 so they cluster together.
    print(f"Aspect sin/cos  (flat < {flat_slope_deg}°) …")
    _grass(
        "r.mapcalc",
        "--overwrite",
        f"expression=_facet_asp_sin = if(_facet_slope < {flat_slope_deg}, "
        f"0, sin(_facet_aspect))",
    )
    _grass(
        "r.mapcalc",
        "--overwrite",
        f"expression=_facet_asp_cos = if(_facet_slope < {flat_slope_deg}, "
        f"0, cos(_facet_aspect))",
    )

    # ── GRASS: imagery group + i.segment ──────────────────────────────

    print("Creating imagery group …")
    _grass(
        "i.group",
        "group=_facet_grp",
        "input=_facet_asp_sin,_facet_asp_cos",
    )

    print(
        f"Running i.segment  (threshold={threshold}, minsize={minsize}, 8-connected) …"
    )
    _grass(
        "i.segment",
        "-d",
        "--overwrite",
        "group=_facet_grp",
        "output=_facet_segs",
        f"threshold={threshold}",
        f"minsize={minsize}",
        "memory=2048",
    )

    # ── Export to temp TIFFs ──────────────────────────────────────────

    print("Exporting intermediate rasters …")
    with tempfile.TemporaryDirectory() as tmpdir:
        exports = {
            "_facet_segs": ("segs.tif", "Int32"),
            "_facet_asp_sin": ("sin.tif", "Float32"),
            "_facet_asp_cos": ("cos.tif", "Float32"),
            "_facet_slope": ("slope.tif", "Float32"),
        }
        paths = {}
        for grass_name, (fname, dtype) in exports.items():
            p = os.path.join(tmpdir, fname)
            _grass(
                "r.out.gdal",
                "-f",
                "--overwrite",
                f"input={grass_name}",
                f"output={p}",
                "format=GTiff",
                f"type={dtype}",
                "createopt=COMPRESS=LZW",
            )
            paths[grass_name] = p

        with rasterio.open(paths["_facet_segs"]) as src:
            segs = src.read(1)
            profile = src.profile.copy()
        with rasterio.open(paths["_facet_asp_sin"]) as src:
            asp_sin = src.read(1)
        with rasterio.open(paths["_facet_asp_cos"]) as src:
            asp_cos = src.read(1)
        with rasterio.open(paths["_facet_slope"]) as src:
            slope = src.read(1)

    # ── Python post-processing: per-facet orientation ─────────────────

    print("Computing per-facet orientation …")

    # GRASS nulls export as Int32 min; clamp to 0 (nodata)
    segs[segs < 0] = 0

    flat_ids = segs.ravel()
    valid = flat_ids > 0
    ids = flat_ids[valid]
    n = int(ids.max()) + 1

    sum_sin = np.bincount(ids, weights=asp_sin.ravel()[valid], minlength=n)
    sum_cos = np.bincount(ids, weights=asp_cos.ravel()[valid], minlength=n)
    sum_slp = np.bincount(ids, weights=slope.ravel()[valid], minlength=n)
    counts = np.bincount(ids, minlength=n).astype(np.float64)
    counts[counts == 0] = 1  # avoid div-by-zero for unused IDs

    mean_sin = sum_sin / counts
    mean_cos = sum_cos / counts
    mean_slp = sum_slp / counts

    # Circular mean aspect: GRASS (CCW from E) → geographic (CW from N)
    grass_deg = np.degrees(np.arctan2(mean_sin, mean_cos))
    grass_deg[grass_deg < 0] += 360
    geo_deg = (450 - grass_deg) % 360

    # Bin into 8 cardinal directions (1-8), 0 = flat
    orient_lut = (((geo_deg + 22.5) % 360) / 45).astype(np.int16) + 1
    orient_lut[mean_slp < flat_slope_deg] = 0
    orient_lut[0] = 0  # nodata / segment-0

    orientation = orient_lut[segs]

    # ── Write output GeoTIFFs ─────────────────────────────────────────

    profile.update(dtype="int32", nodata=0, compress="lzw")
    with rasterio.open(out_facets, "w", **profile) as dst:
        dst.write(segs.astype(np.int32), 1)
    print(f"Written: {out_facets}")

    profile.update(dtype="int16")
    with rasterio.open(out_orient, "w", **profile) as dst:
        dst.write(orientation, 1)
    print(f"Written: {out_orient}")

    # ── Summary stats ─────────────────────────────────────────────────

    unique_ids = np.unique(ids)
    n_facets = len(unique_ids)
    sizes = np.bincount(ids, minlength=n)[1:]
    sizes = sizes[sizes > 0]

    orient_dist = {
        ORIENT_NAMES[k]: int((orient_lut[unique_ids] == k).sum()) for k in range(9)
    }

    print(f"\nFacets:  {n_facets}")
    print(f"Per-orientation: {orient_dist}")
    print(
        f"Size (km²):  min={sizes.min()}  median={int(np.median(sizes))}  "
        f"mean={int(sizes.mean())}  max={sizes.max()}"
    )

    # ── Cleanup GRASS temp layers ─────────────────────────────────────

    _cleanup_grass()

    return out_facets


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build PRISM-style topographic facets (GRASS i.segment)."
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="i.segment similarity threshold 0–1 (default 0.3)",
    )
    p.add_argument(
        "--minsize",
        type=int,
        default=20,
        help="Minimum facet size in cells / km² (default 20)",
    )
    p.add_argument(
        "--flat-slope",
        type=float,
        default=2.0,
        help="Slope below which a cell is flat (degrees, default 2.0)",
    )
    p.add_argument("--out-dir", default=MVP_ROOT)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    build_facets(
        threshold=a.threshold,
        minsize=a.minsize,
        flat_slope_deg=a.flat_slope,
        out_dir=a.out_dir,
        overwrite=a.overwrite,
    )


if __name__ == "__main__":
    main()
