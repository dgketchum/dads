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


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS'

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
        chk = f'/media/nvm/IrrigationGIS/dads/dem/dem_{epsg}'
    elif zone == 'south':
        bounds = (-180., 23., -60., 49.)
        epsg = '5071'
        chk = f'/media/nvm/IrrigationGIS/dads/dem/dem_{epsg}'
    else:
        raise ValueError

    sites_df = pd.read_csv(sites)

    sites_df = sites_df[(sites_df['latitude'] < bounds[3]) & (sites_df['latitude'] >= bounds[1])]
    sites_df = sites_df[(sites_df['longitude'] < bounds[2]) & (sites_df['longitude'] >= bounds[0])]

    tiles = sites_df['MGRS_TILE'].dropna().unique().tolist()
    mgrs_tiles_list = [m for m in tiles if isinstance(m, str)]
    mgrs_tiles_list.sort()

    suspect_files = validate_raster_point_intersections(
        raster_directory=chk,
        points_dataframe=sites_df,
        latitude_col='latitude',
        longitude_col='longitude',
        mgrs_col='MGRS_TILE',
        expected_raster_epsg_str=epsg,
        unique_mgrs_codes_list=mgrs_tiles_list
    )

    print(suspect_files)

# ========================= EOF ====================================================================
