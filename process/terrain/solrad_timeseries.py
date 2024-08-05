import os

import geopandas as gpd
import pandas as pd
import rasterstats


def extract_raster_values_by_tile(shapefile_path, raster_dir, table_out, shuffle=False):
    """"""
    points = gpd.read_file(shapefile_path)
    points.index = points['fid']
    if shuffle:
        points = points.sample(frac=1)

    for tile in points['MGRS_TILE'].unique():

        tile_points = points[points['MGRS_TILE'] == tile]
        tile_dir = os.path.join(raster_dir, tile)
        results = {k: {} for k, v in tile_points.iterrows()}

        if tile != '11UQP':
            continue

        first = True
        for day in range(1, 366):
            raster_file = os.path.join(tile_dir, f'irradiance_day_{day}_{tile}.tif')

            if not os.path.exists(raster_file):
                print(f"Warning: Raster file not found for tile {tile}, day {day}. Skipping.")
                continue

            values = rasterstats.zonal_stats(tile_points, raster_file, stats="mean", geojson_out=True)

            for v in values:
                if first:
                    results[v['id']] = {day: v['properties']['mean']}
                else:
                    results[v['id']][day] = v['properties']['mean']

            first = False

        df = pd.DataFrame(results)
        file_ = os.path.join(table_out, 'tile_{}.csv'.format(tile))
        df.to_csv(file_)
        print(os.path.basename(file_))


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/dads'

    dem_d = os.path.join(root, 'dem')
    mgrs = os.path.join(root, 'training', 'w17_tiles.csv')

    shapefile_path_ = os.path.join(root, 'met', 'stations', 'dads_stations_WMT_mgrs.shp')
    raster_dir_ = os.path.join(root, 'dem', 'rsun')
    solrad_out = os.path.join(root, 'dem', 'rsun_tables')
    extract_raster_values_by_tile(shapefile_path_, raster_dir_, solrad_out, shuffle=True)

# ========================= EOF ====================================================================
