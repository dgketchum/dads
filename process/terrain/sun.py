import os
import subprocess
from multiprocessing import Pool

import pandas as pd

from utils.station_parameters import station_par_map


def compute_horizon(
    dem_name_grass, mapset, horizon_basename, step=5, maxdistance=50000
):
    """Run r.horizon for a tile. Produces N rasters named {horizon_basename}_{azimuth}.

    Args:
        dem_name_grass: e.g. 'dem_11TML'
        mapset: GRASS mapset name
        horizon_basename: output basename, e.g. 'horizon_11TML'
        step: azimuth step in degrees (5 = 72 rasters)
        maxdistance: search radius in meters (50 km)
    """
    subprocess.run(["g.region", f"rast={dem_name_grass}@{mapset}"], check=True)
    cmd = [
        "r.horizon",
        f"elevation={dem_name_grass}@{mapset}",
        f"step={step}",
        f"output={horizon_basename}",
        f"maxdistance={maxdistance}",
        "--overwrite",
    ]
    subprocess.run(cmd, check=True)


def ingest_rasters(in_dir, tiles, mapset, overwrite=False):
    file_names = sorted(os.listdir(in_dir))
    dem_files = [os.path.join(in_dir, f) for f in file_names if f.endswith(".tif")]

    for in_dem in dem_files:
        tile = in_dem.split(".")[0][-5:]

        if tiles and tile not in tiles:
            continue

        dem_name = f"dem_{tile}"
        cmd = ["r.in.gdal", f"input={in_dem}", f"output={dem_name}@{mapset}"]

        if overwrite:
            cmd += ["--overwrite"]

        subprocess.call(cmd)


def worker_calculate_single_tile_irradiance(args):
    """"""
    if len(args) == 9:
        (
            dem_name_grass,
            dem_file_path,
            terrain_dir,
            mapset,
            r_sun_nprocs,
            overwrite,
            doys,
            horizon_basename,
            horizon_step,
        ) = args
    else:
        (
            dem_name_grass,
            dem_file_path,
            terrain_dir,
            mapset,
            r_sun_nprocs,
            overwrite,
            doys,
        ) = args
        horizon_basename, horizon_step = None, None

    tile = dem_name_grass.split("_")[-1]

    slp_dir = os.path.join(terrain_dir, "slope")
    asp_dir = os.path.join(terrain_dir, "aspect")

    slope_output_tif = os.path.join(slp_dir, f"slope_{tile}.tif")
    aspect_output_tif = os.path.join(asp_dir, f"aspect_{tile}.tif")

    slope_grass_name = f"slope_{tile}"
    aspect_grass_name = f"aspect_{tile}"

    slope_cmd = ["gdaldem", "slope", dem_file_path, slope_output_tif]
    aspect_cmd = ["gdaldem", "aspect", dem_file_path, aspect_output_tif]

    subprocess.run(slope_cmd, check=True)
    subprocess.run(aspect_cmd, check=True)

    slope_ingest_cmd = [
        "r.in.gdal",
        f"input={slope_output_tif}",
        f"output={slope_grass_name}",
        "--overwrite",
    ]
    subprocess.run(slope_ingest_cmd, check=True)

    aspect_ingest_cmd = [
        "r.in.gdal",
        f"input={aspect_output_tif}",
        f"output={aspect_grass_name}",
        "--overwrite",
    ]
    subprocess.run(aspect_ingest_cmd, check=True)

    subprocess.run(["g.region", f"rast={dem_name_grass}@{mapset}"], check=True)

    for day in range(1, 366):
        if doys and day not in doys:
            continue

        irradiance_grass_name = f"irradiance_day_{day}_{tile}"

        rsun_command_list = [
            "r.sun",
            f"elevation={dem_name_grass}@{mapset}",
            f"slope={slope_grass_name}@{mapset}",
            f"aspect={aspect_grass_name}@{mapset}",
            f"day={day}",
            f"glob_rad={irradiance_grass_name}",
            f"nprocs={r_sun_nprocs}",
            "--overwrite",
        ]

        if horizon_basename:
            rsun_command_list.insert(-1, f"horizon_basename={horizon_basename}")
            rsun_command_list.insert(-1, f"horizon_step={horizon_step}")

        try:
            subprocess.run(rsun_command_list, check=True)
            print(f"......................................{irradiance_grass_name}")
        except subprocess.CalledProcessError:
            print(f"{irradiance_grass_name} failed")
            continue

    return f"Tile {tile}: Done"


def calculate_terrain_irradiance_parallel(
    terrain_dir,
    terrain_source_path,
    mapset="PERMANENT",
    tiles_filter=None,
    overwrite=False,
    specific_bad_files=None,
    num_parallel_tiles=None,
    r_sun_nprocs_per_tile=1,
):
    """"""
    source_dem_tif_filenames = sorted(
        [f for f in os.listdir(terrain_source_path) if f.endswith(".tif")]
    )

    task_args_list = []
    doys = None
    for dem_tif_filename in source_dem_tif_filenames:
        dem_name_grass = dem_tif_filename.split(".")[0]
        tile_id = dem_name_grass.split("_")[-1]

        if tiles_filter and tile_id not in tiles_filter:
            continue

        if specific_bad_files:
            doys = specific_bad_files[tile_id]

        elif not overwrite:
            try:
                glist_result = subprocess.run(
                    ["g.list", "type=raster", f"pattern=irradiance_day_*_{tile_id}"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                existing_irradiance_files = [
                    r for r in glist_result.stdout.strip().split("\n") if r
                ]

                if len(existing_irradiance_files) >= 365:
                    print(f"{tile_id} is complete, skipping")
                    continue

                else:
                    print(
                        f"{len(existing_irradiance_files)} files complete: adding {tile_id}"
                    )

            except subprocess.CalledProcessError as e:
                print(f"error {e} on {tile_id}")

        dem_file_full_path = os.path.join(terrain_source_path, dem_tif_filename)
        task_args_list.append(
            (
                dem_name_grass,
                dem_file_full_path,
                terrain_dir,
                mapset,
                r_sun_nprocs_per_tile,
                overwrite,
                doys,
            )
        )

    if num_parallel_tiles is None:
        num_parallel_tiles = 1

    collected_results = []
    if task_args_list and num_parallel_tiles > 0:
        with Pool(processes=num_parallel_tiles) as p:
            for worker_result in p.imap_unordered(
                worker_calculate_single_tile_irradiance, task_args_list
            ):
                collected_results.append(worker_result)

    return collected_results


def export_rasters(
    terrain_dir, rsun_out, mapset="PERMANENT", overwrite=False, mgrs_list=None
):
    """"""
    dem_files = sorted(os.listdir(terrain_dir))
    dem_names = sorted([f.split(".")[0] for f in dem_files if f.endswith(".tif")])
    dem_files = [os.path.join(terrain_dir, f) for f in dem_files if f.endswith(".tif")]
    print(f"{len(dem_files)} dem files to export")

    for dem_name, dem_file in zip(dem_names, dem_files):
        tile = dem_name.split("_")[-1]
        if mgrs_list and tile not in mgrs_list:
            continue

        # print('\n', tile)

        tile_dir = os.path.join(rsun_out, tile)
        if not os.path.isdir(tile_dir):
            os.mkdir(tile_dir)

        tile_contents = [f for f in os.listdir(tile_dir) if ".tif" in f]
        if len(tile_contents) >= 365 and not overwrite:
            print(f"tile {tile} already exists, skipping")
            continue

        subprocess.call(["g.region", f"rast=dem_{tile}@{mapset}"])

        for day in range(1, 366):
            irradiance_output_tif = os.path.join(
                tile_dir, "irradiance_day_{0}_{1}.tif".format(day, tile)
            )
            irradiance_input = "irradiance_day_{0}_{1}".format(day, tile)

            subprocess.call(
                [
                    "r.out.gdal",
                    "-c",
                    "input={0}@{1}".format(irradiance_input, mapset),
                    "format=GTiff",
                    "createopt=COMPRESS=LZW",
                    "--overwrite",
                    "output={0}".format(irradiance_output_tif),
                ]
            )

            # print(tile, day)


def compute_rsun_seamless(
    dem_tif,
    grass_db,
    location,
    out_dir,
    horizon_step=5,
    horizon_maxdist=50000,
    nprocs=16,
):
    """Run r.horizon + r.sun on a seamless full-domain DEM.

    Creates a GRASS location from the DEM, imports it, computes horizons once,
    then runs r.sun for DOY 1-365, exporting each DOY as a single-band TIF.

    Args:
        dem_tif: path to seamless DEM GeoTIFF (e.g. dem_pnw_1km.tif)
        grass_db: GRASS database directory (e.g. /data/ssd2/dads/dem/grassdata)
        location: GRASS location name (e.g. pnw_1km)
        out_dir: output directory for rsun_doy_NNN.tif files
        horizon_step: azimuth step in degrees (default 5 → 72 rasters)
        horizon_maxdist: horizon search radius in meters (default 50 km)
        nprocs: r.sun internal threading
    """
    os.makedirs(out_dir, exist_ok=True)
    loc_path = os.path.join(grass_db, location, "PERMANENT")

    # Create GRASS location from DEM if it doesn't exist
    loc_dir = os.path.join(grass_db, location)
    if not os.path.isdir(loc_path):
        print(f"Creating GRASS location at {loc_dir} ...")
        subprocess.run(
            ["grass", "-c", dem_tif, loc_dir, "-e"],
            check=True,
        )

    dem_name = "dem_pnw"
    slope_name = "slope_pnw"
    aspect_name = "aspect_pnw"
    hz_base = "horizon_pnw"
    mapset = "PERMANENT"

    # Helper to run GRASS commands within the location
    def _grass_cmd(cmd_list):
        env = os.environ.copy()
        env["GISBASE"] = subprocess.run(
            ["grass", "--config", "path"], capture_output=True, text=True, check=True
        ).stdout.strip()
        full = ["grass", os.path.join(grass_db, location, mapset), "--exec"] + cmd_list
        subprocess.run(full, check=True)

    # 1. Import DEM
    print("Importing DEM into GRASS ...")
    _grass_cmd(["r.in.gdal", f"input={dem_tif}", f"output={dem_name}", "--overwrite"])

    # 2. Set region to DEM
    _grass_cmd(["g.region", f"rast={dem_name}@{mapset}"])

    # 3. Compute slope/aspect via gdaldem then import
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="rsun_seamless_")
    slope_tif = os.path.join(tmpdir, "slope.tif")
    aspect_tif = os.path.join(tmpdir, "aspect.tif")

    print("Computing slope/aspect via gdaldem ...")
    subprocess.run(["gdaldem", "slope", dem_tif, slope_tif], check=True)
    subprocess.run(["gdaldem", "aspect", dem_tif, aspect_tif], check=True)

    print("Importing slope/aspect into GRASS ...")
    _grass_cmd(
        ["r.in.gdal", f"input={slope_tif}", f"output={slope_name}", "--overwrite"]
    )
    _grass_cmd(
        ["r.in.gdal", f"input={aspect_tif}", f"output={aspect_name}", "--overwrite"]
    )

    # 4. Run r.horizon for full domain
    print(f"Running r.horizon (step={horizon_step}°, maxdist={horizon_maxdist}m) ...")
    _grass_cmd(
        [
            "r.horizon",
            f"elevation={dem_name}@{mapset}",
            f"step={horizon_step}",
            f"output={hz_base}",
            f"maxdistance={horizon_maxdist}",
            "--overwrite",
        ]
    )

    # 5. Loop DOY 1-365: r.sun with horizons
    print("Running r.sun for 365 DOYs ...")
    for doy in range(1, 366):
        rsun_name = f"rsun_doy_{doy:03d}"
        _grass_cmd(
            [
                "r.sun",
                f"elevation={dem_name}@{mapset}",
                f"slope={slope_name}@{mapset}",
                f"aspect={aspect_name}@{mapset}",
                f"horizon_basename={hz_base}",
                f"horizon_step={horizon_step}",
                f"day={doy}",
                f"glob_rad={rsun_name}",
                f"nprocs={nprocs}",
                "--overwrite",
            ]
        )

        # Export
        out_tif = os.path.join(out_dir, f"rsun_doy_{doy:03d}.tif")
        _grass_cmd(
            [
                "r.out.gdal",
                "-c",
                f"input={rsun_name}@{mapset}",
                "format=GTiff",
                "createopt=COMPRESS=LZW",
                "--overwrite",
                f"output={out_tif}",
            ]
        )

        if doy % 50 == 0 or doy == 365:
            print(f"  rsun: {doy}/365 exported", flush=True)

    # Cleanup temp files
    subprocess.run(["rm", "-rf", tmpdir])
    print(f"Done. 365 RSUN TIFs written to {out_dir}")


def _parse_args_seamless():
    import argparse

    p = argparse.ArgumentParser(description="Seamless full-domain r.sun pipeline")
    p.add_argument(
        "--dem-tif",
        required=True,
        help="Seamless DEM GeoTIFF (e.g. /nas/dads/mvp/dem_pnw_1km.tif)",
    )
    p.add_argument(
        "--grass-db",
        default="/data/ssd2/dads/dem/grassdata",
        help="GRASS database directory",
    )
    p.add_argument(
        "--location",
        default="pnw_1km",
        help="GRASS location name",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for rsun_doy_NNN.tif files",
    )
    p.add_argument("--horizon-step", type=int, default=5)
    p.add_argument("--horizon-maxdist", type=int, default=50000)
    p.add_argument("--nprocs", type=int, default=16)
    p.add_argument(
        "--seamless",
        action="store_true",
        help="Run seamless full-domain pipeline (required)",
    )
    return p.parse_args()


if __name__ == "__main__":
    import sys

    if "--seamless" in sys.argv:
        _a = _parse_args_seamless()
        compute_rsun_seamless(
            dem_tif=_a.dem_tif,
            grass_db=_a.grass_db,
            location=_a.location,
            out_dir=_a.out_dir,
            horizon_step=_a.horizon_step,
            horizon_maxdist=_a.horizon_maxdist,
            nprocs=_a.nprocs,
        )
        sys.exit(0)

    root = "/nas"
    dem_d = "/data/ssd2/dads/dem"

    _bucket = "gs://wudr"
    station_set = "ndbc"
    zone = "conus"

    if station_set == "madis":
        stations = "madis_17MAY2025_gap_mgrs"
        sites = os.path.join(root, "dads", "met", "stations", f"{stations}.csv")
        chk = os.path.join(root, "dads", "rs", "landsat", stations)

    elif station_set == "ghcn":
        stations = "ghcn_CANUSA_stations_mgrs"
        sites = os.path.join(
            root, "climate", "ghcn", "stations", "ghcn_stations_mgrs_country.csv"
        )
        chk = os.path.join(root, "dads", "rs", "ghcn_stations", "landsat", "tiles")

    elif station_set == "ndbc":
        stations = "ndbc_stations"
        sites = os.path.join(root, "climate", "ndbc", "ndbc_meta", "ndbc_stations.csv")
        chk = None

    else:
        raise ValueError

    if zone == "canada":
        bounds = (-141.0, 49.0, -60.0, 85.0)
        epsg = "3978"
        tif_dem = os.path.join(dem_d, f"dem_{epsg}")
        mapset_ = "dads_map_canada"

    elif zone == "conus":
        bounds = (-180.0, 23.0, -60.0, 49.0)
        epsg = "5071"
        tif_dem = os.path.join(dem_d, f"dem_{epsg}")
        mapset_ = "dads_map"

    elif zone == "alaska":
        bounds = (-180.0, 49.0, -60.0, 85.0)
        epsg = "6393"
        tif_dem = os.path.join(dem_d, f"dem_{epsg}")
        mapset_ = "dads_map_alaska"

    else:
        raise ValueError

    sites_df = pd.read_csv(sites)

    kw = station_par_map(station_set)

    sites_df = sites_df[
        (sites_df[kw["lat"]] < bounds[3]) & (sites_df[kw["lat"]] >= bounds[1])
    ]
    sites_df = sites_df[
        (sites_df[kw["lon"]] < bounds[2]) & (sites_df[kw["lon"]] >= bounds[0])
    ]

    if zone == "canada":
        sites_df = sites_df[sites_df["AFF_ISO"] == "CA"]

    zone_tiles = [d[4:9] for d in os.listdir(tif_dem)]

    bad_tiles_file = os.path.join(root, "dads", "dem", "bad_tiles.txt")
    with open(bad_tiles_file, "r") as f:
        lines = f.readlines()

    bad_files, total_files = {}, 0
    for line in lines:
        splt = line.split(os.path.sep)
        try:
            tile = splt[8]
        except IndexError:
            continue

        if tile not in zone_tiles:
            continue

        doy = int(splt[9].split("_")[2])
        if tile not in bad_files:
            bad_files[tile] = [doy]
        else:
            bad_files[tile].append(doy)
        total_files += 1

    tiles = list(bad_files.keys())

    ingest_rasters(tif_dem, tiles, mapset=mapset_, overwrite=True)

    rsun_out_ = os.path.join(root, "dads", "dem", "rsun_irradiance")

    calculate_terrain_irradiance_parallel(
        dem_d,
        tif_dem,
        mapset=mapset_,
        tiles_filter=tiles,
        overwrite=True,
        specific_bad_files=bad_files,
        num_parallel_tiles=4,
        r_sun_nprocs_per_tile=4,
    )

    export_rasters(tif_dem, rsun_out_, mapset=mapset_, mgrs_list=tiles, overwrite=True)

# ========================= EOF ====================================================================
