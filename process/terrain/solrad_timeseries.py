import os
import csv

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from concurrent.futures import ProcessPoolExecutor
import rasterio


def process_tile(tile, points, raster_dir, out_dir, index_col='fid', overwrite=False, bad_tiles=None):

    if bad_tiles:
        with open(bad_tiles, 'r') as f:
            tile_list = [line.strip() for line in f]
        if tile in tile_list:
            print(f'{tile} was already found to be bad')
            return

    tile_points = points[points['MGRS_TILE'] == tile]
    station_ct = tile_points.shape[0]
    if tile_points.empty:
        return

    complete_stations = [f.split('.')[0] for f in os.listdir(out_dir)]
    complete_tile_stations = [s for s in tile_points.index if s in complete_stations]

    if all([s in complete_tile_stations for s in tile_points.index]) and not overwrite:
        print(f'{tile} {len(tile_points)} tile points already processed, skipping')
        return
    elif overwrite:
        pass
    else:
        tile_points = tile_points.loc[[i for i in tile_points.index if i not in complete_tile_stations]]

    tile_dir = os.path.join(raster_dir, tile)
    results = {k: {} for k, v in tile_points.iterrows()}

    first_raster_file = None
    for day in range(1, 366):
        potential_raster = os.path.join(tile_dir, f'irradiance_day_{day}_{tile}.tif')
        if os.path.exists(potential_raster):
            first_raster_file = potential_raster
            break

    if not first_raster_file:
        print(f'did not find a raster file for {tile}')
        return

    with rasterio.open(first_raster_file) as src:
        raster_crs = src.crs

    points_crs = tile_points.crs

    if not points_crs.equals(raster_crs):
        tile_points = tile_points.to_crs(raster_crs)

    first, missing, bad_file_ct, values = True, False, False, None
    print(f'process {tile_points.shape[0]} of {station_ct} from {tile}')
    for day in range(1, 366):
        raster_file = os.path.join(tile_dir, f'irradiance_day_{day}_{tile}.tif')

        try:
            values = zonal_stats(tile_points, raster_file, stats="mean", geojson_out=True)
        except rasterio.errors.RasterioIOError:
            print(f'{raster_file}', flush=True)
            bad_file_ct += 1
            if bad_tiles:
                with open(bad_tiles, 'a') as f:
                    f.write(f'{raster_file}\n')
                continue

        for i, v in enumerate(values):
            point_index = tile_points.index[i]
            mean_val = v['properties']['mean']

            if mean_val is None:
                print(f'{raster_file}', flush=True)
                bad_file_ct += 1
                if bad_tiles:
                    os.remove(raster_file)
                    with open(bad_tiles, 'a') as f:
                        f.write(f'{raster_file}\n')
                    break

            if first:
                results[point_index] = {day: mean_val}
            else:
                results[point_index][day] = mean_val

        if first:
            first = False

    if bad_file_ct > 0:
        return

    df = pd.DataFrame.from_dict(results, orient='index')
    df.index.name = index_col
    df = df.reindex(sorted(df.columns), axis=1)

    ct = 0
    for i, r in df.iterrows():
        sf = os.path.join(out_dir, f'{i}.csv')

        if os.path.exists(sf) and not overwrite:
            continue

        r.to_csv(sf)
        ct += 1

    print(f'tile {tile}, {ct} new stations written of {station_ct}')


def extract_raster_values_by_tile(shapefile_path, raster_dir, table_out, shuffle=False, avoid_tiles=None,
                                  write_errors=None, target_tiles=None, overwrite=False, num_workers=2,
                                  index_col='fid', debug=False):

    points = gpd.read_file(shapefile_path)
    points.index = points[index_col]

    if shuffle:
        points = points.sample(frac=1)

    tiles = points['MGRS_TILE'].unique().tolist()
    print(f'{len(tiles)} tiles, {points.shape[0]} points')

    if avoid_tiles:
        tiles = [t for t in tiles if t not in avoid_tiles]
        print(f'{len(tiles)} tiles after {len(avoid_tiles)} excluded')

    if target_tiles:
        ln = len(tiles)
        tiles = [t for t in tiles if t in target_tiles]
        print(f'{len(tiles)} tiles from {ln} in file')

    if debug:
        for tile in tiles:
            process_tile(tile, points.copy(), raster_dir, table_out, index_col, overwrite, write_errors)

    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            executor.map(process_tile, tiles, [points] * len(tiles), [raster_dir] * len(tiles),
                         [table_out] * len(tiles), [index_col] * len(tiles), [overwrite] * len(tiles),
                         [write_errors] * len(tiles))


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    raster_dir_ = os.path.join(root, 'dads', 'dem', 'rsun_irradiance')
    solrad_out = os.path.join(root, 'dads', 'dem', 'rsun_stations')

    # bad_tiles_file = os.path.join(root, 'dads', 'dem', 'bad_tiles.txt')
    # if not os.path.isfile(bad_tiles_file):
    #     with open(bad_tiles_file, 'w'):
    #         pass

    bad_tiles_file = os.path.join(root, 'dads', 'dem', 'bad_tiles.txt')
    with open(bad_tiles_file, 'r') as f:
        lines = f.readlines()

    bad_files, total_files = {}, 0
    for line in lines:
        splt = line.split(os.path.sep)
        try:
            tile = splt[8]
        except IndexError:
            continue

        doy = int(splt[9].split('_')[2])
        if tile not in bad_files:
            bad_files[tile] = [doy]
        else:
            bad_files[tile].append(doy)
        total_files += 1

    tiles = list(bad_files.keys())

    # shapefile_path_ = os.path.join(root, 'dads', 'met', 'stations', 'madis_17MAY2025_mgrs.shp')
    shapefile_path_ = os.path.join(root, 'climate', 'ndbc', 'ndbc_meta', 'ndbc_stations.shp')
    extract_raster_values_by_tile(shapefile_path_, raster_dir_, solrad_out, num_workers=16, avoid_tiles=None,
                                  target_tiles=None, shuffle=True, overwrite=False, index_col='station_id',
                                  debug=False, write_errors=None)

    # shapefile_path_ = os.path.join(root, 'climate', 'ghcn', 'stations', 'ghcn_stations_mgrs_country.shp')
    # extract_raster_values_by_tile(shapefile_path_, raster_dir_, solrad_out, num_workers=16, avoid_tiles=None,
    #                               target_tiles=None, shuffle=True, overwrite=False, index_col='STAID',
    #                               debug=True)

# ========================= EOF ====================================================================
