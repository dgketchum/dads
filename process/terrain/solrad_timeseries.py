import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterstats


def extract_raster_values_by_tile(shapefile_path, raster_dir, table_out, shuffle=False, overwrite=False):
    """"""
    write = False
    points = gpd.read_file(shapefile_path)
    points.index = points['fid']
    if shuffle:
        points = points.sample(frac=1)

    for tile in points['MGRS_TILE'].unique():

        if tile is None:
            continue

        file_ = os.path.join(table_out, 'tile_{}.csv'.format(tile))
        if os.path.exists(file_) and not overwrite:
            df = pd.read_csv(file_, index_col=0)
            try:
                nans = np.any(np.isnan(df.values))
            except TypeError:
                os.remove(file_)
                continue
            if nans:
                os.remove(file_)
                print('remove rsun table {}, has nan'.format(tile))
            else:
                continue

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

            values = rasterstats.zonal_stats(tile_points, raster_file, stats="mean", geojson_out=True)

            for v in values:
                if v['properties']['mean'] is None:
                    print('Empty value in {} for {}'.format(tile, v['properties']['fid']))
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


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/dads'

    dem_d = os.path.join(root, 'dem')
    mgrs = os.path.join(dem_d, 'w17_tiles.csv')

    # this must be EPSG:5071 shapefile
    shapefile_path_ = os.path.join(root, 'met', 'stations', 'dads_stations_res_elev_mgrs_5071.shp')
    raster_dir_ = os.path.join(root, 'dem', 'rsun')
    solrad_out = os.path.join(root, 'dem', 'rsun_tables')
    extract_raster_values_by_tile(shapefile_path_, raster_dir_, solrad_out, shuffle=True, overwrite=False)

# ========================= EOF ====================================================================
