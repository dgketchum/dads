import os
import glob
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from shapely.geometry import Point, box


def get_mgrs_code_from_filename(filename, known_mgrs_codes):
    basename = os.path.basename(filename)
    for code in known_mgrs_codes:
        if isinstance(code, str) and code in basename:
            return code
    return None


def validate_raster_point_intersections(raster_directory, points_dataframe,
                                        latitude_col, longitude_col, mgrs_col,
                                        expected_raster_epsg_str, unique_mgrs_codes_list):
    source_points_crs = CRS.from_epsg(4326)
    expected_raster_crs = CRS.from_epsg(int(expected_raster_epsg_str))

    suspect_raster_files = []
    raster_filepaths = glob.glob(os.path.join(raster_directory, '*.tif'))

    for raster_path in raster_filepaths:
        raster_filename = os.path.basename(raster_path)
        raster_mgrs_code = get_mgrs_code_from_filename(raster_filename, unique_mgrs_codes_list)

        if not raster_mgrs_code:
            continue

        relevant_points_df = points_dataframe[
            (points_dataframe[mgrs_col] == raster_mgrs_code) & \
            pd.notna(points_dataframe[longitude_col]) & \
            pd.notna(points_dataframe[latitude_col])
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
                point_geometry_orig_crs = Point(point_row[longitude_col], point_row[latitude_col])
                transformed_point_dict = transform_geom(
                    source_points_crs,
                    src.crs,
                    point_geometry_orig_crs.__geo_interface__
                )
                point_geometry_raster_crs = Point(transformed_point_dict['coordinates'])

                if raster_bounds_polygon.intersects(point_geometry_raster_crs):
                    intersection_found = True
                    break

            if not intersection_found:
                if raster_filename not in suspect_raster_files:
                    suspect_raster_files.append(raster_filename)
                    print(f'removing {raster_filename}')
                    os.remove(raster_path)

    return suspect_raster_files


import subprocess


def _internal_parse_grass_coords(output_str: str) -> dict:
    coords = {}
    for line in output_str.strip().split('\n'):
        if '=' not in line:
            continue
        key, value_str = line.split('=', 1)
        try:
            coords[key.strip()] = float(value_str)
        except ValueError:
            pass
    return coords


def remove_rasters(dem_dir, tiles):

    dem_files = [os.path.join(dem_dir, t) for t in tiles]

    for dem_name, dem_file in zip(tiles, dem_files):
        tile = dem_name.split('_')[-1]
        subprocess.call(['g.remove', 'type=raster', f'name=dem_{tile}', '-f'])
        subprocess.call(['g.remove', 'type=raster', f'name=slope_{tile}', '-f'])
        subprocess.call(['g.remove', 'type=raster', f'name=aspect_{tile}', '-f'])
        subprocess.call(['g.remove', 'type=raster', f'name=aspect_{tile}', '-f'])

        for day in range(1, 366):
            irradiance_output_path = 'irradiance_day_{0}_{1}'.format(day, tile)
            subprocess.call(['g.remove', 'type=raster', f'name={irradiance_output_path}', '-f'])

        print('removed {}'.format(tile))


def reproject_dems(in_dir, tiles, output_dir, in_epsg, out_epsg):
    file_names = sorted(os.listdir(in_dir))
    dem_files = [os.path.join(in_dir, f) for f in file_names if f.endswith('.tif')]
    out_files = [os.path.join(output_dir, f) for f in file_names if f.endswith('.tif')]

    for in_dem, out_dem in zip(dem_files, out_files):
        tile = in_dem.split('.')[0][-5:]

        if tile not in tiles:
            continue
        subprocess.run([
            "gdalwarp", "-s_srs", f"EPSG:{in_epsg}", '-overwrite',
            "-t_srs", f"EPSG:{out_epsg}", "-r", "bilinear",
            "-of", "GTiff", in_dem, out_dem
        ])
        print(tile, os.path.basename(in_dem), os.path.basename(out_dem))


if __name__ == '__main__':

    d = '/home/dgketchum/data/IrrigationGIS'
    dem_d = '/data/ssd2/dads/dem'

    _bucket = 'gs://wudr'
    station_set = 'madis'
    zone = 'north'

    if station_set == 'madis':
        stations = 'madis_17MAY2025_gap_mgrs'
        sites = os.path.join(d, 'dads', 'met', 'stations', f'{stations}.csv')
    elif station_set == 'ghcn':
        stations = 'ghcn_CANUSA_stations_mgrs'
        sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    else:
        raise ValueError

    if zone == 'north':
        bounds = (-180., 49., -60., 85.)
        epsg = '3978'
        chk = os.path.join(dem_d, f'/dem_{epsg}')
        mapset_ = "dads_map_canada"


    elif zone == 'south':
        bounds = (-180., 23., -60., 49.)
        epsg = '5071'
        chk = os.path.join(dem_d, f'/dem_{epsg}')
        mapset_ = "dads_map_conus"

    else:
        raise ValueError

    sites_df = pd.read_csv(sites)

    sites_df = sites_df[(sites_df['latitude'] < bounds[3]) & (sites_df['latitude'] >= bounds[1])]
    sites_df = sites_df[(sites_df['longitude'] < bounds[2]) & (sites_df['longitude'] >= bounds[0])]

    tiles = sites_df['MGRS_TILE'].dropna().unique().tolist()
    mgrs_tiles_list = [m for m in tiles if isinstance(m, str)]
    mgrs_tiles_list.sort()
    ind = [t for t in mgrs_tiles_list]

# ========================= EOF ====================================================================
