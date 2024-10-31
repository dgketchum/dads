import os

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterstats import zonal_stats
from concurrent.futures import ProcessPoolExecutor


def process_tile(tile, points, raster_dir, table_out, index_col='fid', overwrite=False):
    write = False
    file_ = os.path.join(table_out, 'tile_{}.csv'.format(tile))
    if os.path.exists(file_) and not overwrite:
        return

    tile_points = points[points['MGRS_TILE'] == tile]
    tile_dir = os.path.join(raster_dir, tile)
    results = {k: {} for k, v in tile_points.iterrows()}

    first, found = True, True
    for day in range(1, 366):
        raster_file = os.path.join(tile_dir, f'irradiance_day_{day}_{tile}.tif')

        if not os.path.exists(raster_file):
            if found:
                print(f"Warning: Raster file not found for tile {tile}, day {day}. Skipping.")
            found = False
            write = False
            continue

        values = zonal_stats(tile_points, raster_file, stats="mean", geojson_out=True)

        for v in values:
            if v['properties']['mean'] is None:
                print('Empty value in {} for {}'.format(tile, v['properties'][index_col]))
                continue
            if first:
                results[v['id']] = {day: v['properties']['mean']}
            else:
                results[v['id']][day] = v['properties']['mean']

        first = False
        if results[v['id']]:
            write = True
        else:
            write = False

    if write:
        df = pd.DataFrame(results)
        df.to_csv(file_)
        print(os.path.basename(file_))


def extract_raster_values_by_tile(shapefile_path, raster_dir, table_out, shuffle=False,
                                  overwrite=False, num_workers=2, index_col='fid'):
    points = gpd.read_file(shapefile_path)
    points.index = points[index_col]
    if shuffle:
        points = points.sample(frac=1)

    tiles = points['MGRS_TILE'].unique().tolist()
    print(f'{len(tiles)} tiles, {points.shape[0]} points')

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_tile, tiles, [points] * len(tiles), [raster_dir] * len(tiles),
                     [table_out] * len(tiles), [index_col] * len(tiles), [overwrite] * len(tiles))


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    dem_d = os.path.join(root, 'dem')
    mgrs = os.path.join(dem_d, 'w17_tiles.csv')

    # this must be EPSG:5071 shapefile
    # shapefile_path_ = os.path.join(root, 'met', 'stations', 'dads_stations_res_elev_mgrs_5071.shp')
    # shapefile_path_ = os.path.join(root, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs_5071.shp')
    shapefile_path_ = os.path.join(root, 'dads', 'met', 'stations', 'madis_mgrs_28OCT2024_5071.shp')

    raster_dir_ = os.path.join(root, 'dads', 'dem', 'rsun')
    solrad_out = os.path.join(root, 'dads', 'dem', 'rsun_tables', 'madis_28OCT2024')
    extract_raster_values_by_tile(shapefile_path_, raster_dir_, solrad_out, num_workers=6,
                                  shuffle=True, overwrite=False, index_col='fid')

# ========================= EOF ====================================================================
