"""
Compute Winstral Sx (maximum upwind slope) at station locations.

For each station and each of N azimuth bins, compute the maximum terrain
slope angle along rays from the station at 2 search distances (local and
outlying).  Positive Sx = sheltered (upwind obstacle), negative = exposed.

Expects a projected DEM (e.g. EPSG:5071, 250 m cells).  Station lon/lat
coordinates are reprojected into the DEM CRS for windowed reads.

Output
------
Parquet with columns:
  fid, latitude, longitude,
  sx_{azimuth:03d}_{dist}k  (16 azimuths x 2 distances = 32),
  terrain_openness, terrain_directionality
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds


def _build_vrt(dem_dir: str, bounds_proj: tuple[float, float, float, float]) -> str:
    """Build a GDAL VRT mosaic from DEM tiles, buffered around projected bounds."""
    tiles = sorted(
        os.path.join(dem_dir, f) for f in os.listdir(dem_dir) if f.endswith(".tif")
    )
    if not tiles:
        raise FileNotFoundError(f"No .tif files in {dem_dir}")

    w, s, e, n = bounds_proj
    buf = 15_000  # 15 km buffer in metres
    tmp = tempfile.NamedTemporaryFile(suffix=".vrt", delete=False)
    vrt_path = tmp.name
    tmp.close()

    subprocess.run(
        [
            "gdalbuildvrt",
            "-te",
            str(w - buf),
            str(s - buf),
            str(e + buf),
            str(n + buf),
            vrt_path,
        ]
        + tiles,
        check=True,
        capture_output=True,
    )
    return vrt_path


def _compute_sx_one_station(
    x_proj: float,
    y_proj: float,
    dem_path: str,
    azimuths_deg: np.ndarray,
    distances_m: list[int],
    sector_half_deg: float = 15.0,
    n_sector_rays: int = 3,
) -> dict[str, float]:
    """Compute Sx for one station across all azimuths and distances.

    Coordinates must be in the DEM's projected CRS (metres).
    Returns a dict of sx_{azimuth:03d}_{dist/1000}k -> float.
    """
    max_dist_m = max(distances_m)
    buf_m = max_dist_m * 1.1

    with rasterio.open(dem_path) as src:
        win = from_bounds(
            x_proj - buf_m,
            y_proj - buf_m,
            x_proj + buf_m,
            y_proj + buf_m,
            src.transform,
        )
        # Clamp to raster bounds
        row_off = max(0, int(win.row_off))
        col_off = max(0, int(win.col_off))
        row_end = min(src.height, int(win.row_off + win.height))
        col_end = min(src.width, int(win.col_off + win.width))
        if row_end <= row_off or col_end <= col_off:
            return {}

        win_clamped = rasterio.windows.Window(
            col_off, row_off, col_end - col_off, row_end - row_off
        )
        dem = src.read(1, window=win_clamped).astype("float32")
        win_transform = src.window_transform(win_clamped)

    # Station position in pixel coords
    inv_tf = ~win_transform
    st_col, st_row = inv_tf * (x_proj, y_proj)
    st_row, st_col = int(round(st_row)), int(round(st_col))

    if st_row < 0 or st_row >= dem.shape[0] or st_col < 0 or st_col >= dem.shape[1]:
        return {}

    z_station = float(dem[st_row, st_col])
    if not np.isfinite(z_station):
        return {}

    # Cell size in metres (constant for projected CRS)
    cell_m = abs(win_transform.a)  # square pixels

    results: dict[str, float] = {}

    for az_deg in azimuths_deg:
        # Build sector rays
        if n_sector_rays > 1 and sector_half_deg > 0:
            ray_angles = np.linspace(
                az_deg - sector_half_deg, az_deg + sector_half_deg, n_sector_rays
            )
        else:
            ray_angles = np.array([az_deg])

        for dist_m in distances_m:
            sx_best = -np.inf
            d0 = cell_m  # skip station's own pixel

            for ray_az in ray_angles:
                az_rad = np.radians(ray_az)
                # Azimuth 0=N, 90=E.  Projected CRS: x=east, y=north.
                # In pixel coords: row decreases with increasing y (northward),
                # col increases with increasing x (eastward).
                dx_m = np.sin(az_rad)  # eastward
                dy_m = -np.cos(az_rad)  # northward → row decreases

                n_steps = int((dist_m - d0) / cell_m) + 1

                for i in range(n_steps):
                    d = d0 + i * cell_m
                    r = st_row + (dy_m * d) / cell_m
                    c = st_col + (dx_m * d) / cell_m
                    ri, ci = int(round(r)), int(round(c))

                    if ri < 0 or ri >= dem.shape[0] or ci < 0 or ci >= dem.shape[1]:
                        break

                    z = dem[ri, ci]
                    if not np.isfinite(z):
                        continue

                    slope = (z - z_station) / d
                    if slope > sx_best:
                        sx_best = slope

            az_label = f"{int(az_deg) % 360:03d}"
            dist_label = f"{dist_m // 1000}k"
            key = f"sx_{az_label}_{dist_label}"
            results[key] = float(sx_best) if np.isfinite(sx_best) else 0.0

    return results


def build_station_sx(
    dem_dir: str,
    stations_csv: str,
    out_path: str,
    n_azimuths: int = 16,
    distances: list[int] | None = None,
    max_workers: int = 8,
    overwrite: bool = False,
) -> str:
    if os.path.exists(out_path) and not overwrite:
        print(f"Output exists: {out_path}")
        return out_path

    if distances is None:
        distances = [2000, 10000]

    stations = pd.read_csv(stations_csv)
    id_col = "station_id" if "station_id" in stations.columns else "fid"
    lat_col = "latitude"
    lon_col = "longitude"

    print(f"Stations: {len(stations)}")
    print(f"Azimuths: {n_azimuths}, distances: {distances}")

    azimuths_deg = np.linspace(0, 360, n_azimuths, endpoint=False)

    # Detect DEM CRS from a sample tile
    sample_tile = next(
        os.path.join(dem_dir, f)
        for f in sorted(os.listdir(dem_dir))
        if f.endswith(".tif")
    )
    with rasterio.open(sample_tile) as src:
        dem_crs = src.crs
    print(f"DEM CRS: {dem_crs}")

    # Reproject station coordinates to DEM CRS
    to_proj = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    lons = stations[lon_col].values
    lats = stations[lat_col].values
    xs_proj, ys_proj = to_proj.transform(lons, lats)

    # Compute projected bounds for VRT
    w = float(np.min(xs_proj))
    e = float(np.max(xs_proj))
    s = float(np.min(ys_proj))
    n = float(np.max(ys_proj))

    print("Building VRT from DEM tiles...")
    vrt_path = _build_vrt(dem_dir, (w, s, e, n))
    print(f"VRT: {vrt_path}")

    rows = []
    errors = 0

    def _process(fid, x_proj, y_proj, lon, lat):
        try:
            sx = _compute_sx_one_station(
                x_proj, y_proj, vrt_path, azimuths_deg, distances
            )
            if not sx:
                return None
            sx["fid"] = fid
            sx["latitude"] = lat
            sx["longitude"] = lon
            return sx
        except Exception as exc:
            return f"ERROR:{fid}:{exc}"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for i, row in stations.iterrows():
            fid = str(row[id_col])
            lon = float(row[lon_col])
            lat = float(row[lat_col])
            fut = pool.submit(_process, fid, xs_proj[i], ys_proj[i], lon, lat)
            futures[fut] = fid

        done = 0
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            if result is None:
                errors += 1
            elif isinstance(result, str) and result.startswith("ERROR:"):
                errors += 1
                if errors <= 5:
                    print(f"  {result}")
            else:
                rows.append(result)
            if done % 1000 == 0:
                print(f"  {done}/{len(stations)} stations processed")

    print(f"Processed: {len(rows)} ok, {errors} errors")

    # Clean up VRT
    try:
        os.unlink(vrt_path)
    except OSError:
        pass

    df = pd.DataFrame(rows)

    # Derived metrics
    sx_cols = [c for c in df.columns if c.startswith("sx_")]
    sx_10k = [c for c in sx_cols if c.endswith("_10k")]
    all_sx = df[sx_cols].values
    df["terrain_openness"] = np.nanmean(all_sx, axis=1)
    df["terrain_directionality"] = np.nanstd(df[sx_10k].values, axis=1)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Written: {out_path}  ({len(df)} stations, {len(df.columns)} columns)")
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Winstral Sx at station locations.")
    p.add_argument("--dem-dir", default="/data/ssd2/dads/dem/dem_5071")
    p.add_argument("--stations-csv", default="artifacts/madis_pnw.csv")
    p.add_argument("--out", default="/nas/dads/mvp/station_sx_pnw.parquet")
    p.add_argument("--azimuths", type=int, default=16)
    p.add_argument(
        "--distances",
        default="2000,10000",
        help="Comma-separated search distances in metres",
    )
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    distances = [int(d) for d in a.distances.split(",")]
    build_station_sx(
        dem_dir=a.dem_dir,
        stations_csv=a.stations_csv,
        out_path=a.out,
        n_azimuths=a.azimuths,
        distances=distances,
        max_workers=a.max_workers,
        overwrite=a.overwrite,
    )


if __name__ == "__main__":
    main()
