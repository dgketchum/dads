import os
import glob
import subprocess

import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from shapely.geometry import Point, box

from prep.station_parameters import station_par_map


def get_mgrs_code_from_filename(filename, known_mgrs_codes):
    basename = os.path.basename(filename)
    for code in known_mgrs_codes:
        if isinstance(code, str) and code in basename:
            return code
    return None


def validate_raster_point_intersections(
    raster_directory,
    points_dataframe,
    latitude_col,
    longitude_col,
    mgrs_col,
    expected_raster_epsg_str,
    unique_mgrs_codes_list,
):
    source_points_crs = CRS.from_epsg(4326)
    expected_raster_crs = CRS.from_epsg(int(expected_raster_epsg_str))

    suspect_raster_files = []
    raster_filepaths = glob.glob(os.path.join(raster_directory, "*.tif"))

    for raster_path in raster_filepaths:
        raster_filename = os.path.basename(raster_path)
        raster_mgrs_code = get_mgrs_code_from_filename(
            raster_filename, unique_mgrs_codes_list
        )

        if not raster_mgrs_code:
            continue

        relevant_points_df = points_dataframe[
            (points_dataframe[mgrs_col] == raster_mgrs_code)
            & pd.notna(points_dataframe[longitude_col])
            & pd.notna(points_dataframe[latitude_col])
        ].copy()

        if relevant_points_df.empty:
            continue

        with rasterio.open(raster_path) as src:
            if not src.crs:
                if raster_filename not in suspect_raster_files:
                    suspect_raster_files.append(raster_filename)
                continue

            if src.crs != expected_raster_crs:
                if raster_filename not in suspect_raster_files:
                    suspect_raster_files.append(raster_filename)
                continue

            raster_bounds_polygon = box(*src.bounds)
            intersection_found = False

            for _, point_row in relevant_points_df.iterrows():
                point_geometry_orig_crs = Point(
                    point_row[longitude_col], point_row[latitude_col]
                )
                transformed_point_dict = transform_geom(
                    source_points_crs,
                    src.crs,
                    point_geometry_orig_crs.__geo_interface__,
                )
                point_geometry_raster_crs = Point(transformed_point_dict["coordinates"])

                if raster_bounds_polygon.intersects(point_geometry_raster_crs):
                    intersection_found = True
                    break

            if not intersection_found:
                if raster_filename not in suspect_raster_files:
                    suspect_raster_files.append(raster_filename)
                    print(f"removing {raster_filename}")
                    os.remove(raster_path)

    return suspect_raster_files


def _internal_parse_grass_coords(output_str: str) -> dict:
    coords = {}
    for line in output_str.strip().split("\n"):
        if "=" not in line:
            continue
        key, value_str = line.split("=", 1)
        try:
            coords[key.strip()] = float(value_str)
        except ValueError:
            pass
    return coords


def remove_rasters(dem_dir, tiles):

    dem_files = [os.path.join(dem_dir, t) for t in tiles]

    for dem_name, dem_file in zip(tiles, dem_files):
        tile = dem_name.split("_")[-1]
        subprocess.call(["g.remove", "type=raster", f"name=dem_{tile}", "-f"])
        subprocess.call(["g.remove", "type=raster", f"name=slope_{tile}", "-f"])
        subprocess.call(["g.remove", "type=raster", f"name=aspect_{tile}", "-f"])
        subprocess.call(["g.remove", "type=raster", f"name=aspect_{tile}", "-f"])

        for day in range(1, 366):
            irradiance_output_path = "irradiance_day_{0}_{1}".format(day, tile)
            subprocess.call(
                ["g.remove", "type=raster", f"name={irradiance_output_path}", "-f"]
            )

        print("removed {}".format(tile))


def list_and_check_geotiff_crs(directory_path, expected_epsg=5071, dry_run=True):
    ct, rm = 0, 0
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if os.path.isfile(filepath) and filename.lower().endswith(".tif"):
            with rasterio.open(filepath) as dataset:
                ct += 1
                crs_object = int(dataset.crs.data["init"].split(":")[1])

                if crs_object != expected_epsg:
                    rm += 1
                    if dry_run:
                        print(f"{filename} has crs {crs_object}")
                    else:
                        base, ext = os.path.splitext(filename)
                        temp_filename = f"{base}_TEMP_CRS_FIX{ext}"
                        temp_filepath = os.path.join(directory_path, temp_filename)

                        os.rename(filepath, temp_filepath)

                        reproject_single_geotiff(
                            input_file_path=temp_filepath,
                            output_file_path=filepath,
                            source_epsg=crs_object,
                            target_epsg=expected_epsg,
                        )

                        os.remove(temp_filepath)

    print(f"reproject {rm} of {ct} files")


def reproject_single_geotiff(
    input_file_path, output_file_path, source_epsg, target_epsg
):
    subprocess.run(
        [
            "gdalwarp",
            "-s_srs",
            f"EPSG:{source_epsg}",
            "-overwrite",
            "-t_srs",
            f"EPSG:{target_epsg}",
            "-r",
            "bilinear",
            "-of",
            "GTiff",
            input_file_path,
            output_file_path,
        ]
    )


def reproject_dems(in_dir, tiles, output_dir, in_epsg, out_epsg):
    file_names = sorted(os.listdir(in_dir))
    dem_files = [os.path.join(in_dir, f) for f in file_names if f.endswith(".tif")]
    out_files = [os.path.join(output_dir, f) for f in file_names if f.endswith(".tif")]

    for in_dem, out_dem in zip(dem_files, out_files):
        tile = in_dem.split(".")[0][-5:]

        if tile not in tiles:
            continue
        subprocess.run(
            [
                "gdalwarp",
                "-s_srs",
                f"EPSG:{in_epsg}",
                "-overwrite",
                "-t_srs",
                f"EPSG:{out_epsg}",
                "-r",
                "bilinear",
                "-of",
                "GTiff",
                in_dem,
                out_dem,
            ]
        )
        print(tile, os.path.basename(in_dem), os.path.basename(out_dem))


if __name__ == "__main__":
    d = "/nas"
    dem_d = "/data/ssd2/dads/dem"

    _bucket = "gs://wudr"
    station_set = "madis"
    zone = "conus"

    if station_set == "madis":
        stations = "madis_17MAY2025_gap_mgrs"
        sites = os.path.join(d, "dads", "met", "stations", f"{stations}.csv")
    elif station_set == "ghcn":
        stations = "ghcn_CANUSA_stations_mgrs"
        sites = os.path.join(
            d, "climate", "ghcn", "stations", "ghcn_CANUSA_stations_mgrs.csv"
        )
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

    tiles = sites_df["MGRS_TILE"].unique().tolist()
    tiles = [m for m in tiles if isinstance(m, str)]
    mgrs_tiles = list(set(tiles))
    mgrs_tiles.sort()

    list_and_check_geotiff_crs(tif_dem, expected_epsg=5071, dry_run=False)


# ========================= EOF ====================================================================
