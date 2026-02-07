import os
import sys
import time

import ee
import pandas as pd
from utils.station_parameters import station_par_map


sys.path.insert(0, os.path.abspath('../..'))
from extract.rs.earth_engine.ee_utils import is_authorized

sys.setrecursionlimit(2000)

BOUNDARIES = 'users/dgketchum/boundaries'


def export_terrain_features(file_prefix, points_layer, buffer_, tiles, check_dir=None):
    """"""
    points = ee.FeatureCollection(points_layer)
    points = points.map(lambda x: x.buffer(buffer_))
    mgrs = ee.FeatureCollection('users/dgketchum/boundaries/MGRS_TILE')
    task = None

    for tile in tiles:

        clip = mgrs.filterMetadata('MGRS_TILE', 'equals', tile)

        desc = '{}_{}'.format(file_prefix, tile)
        if check_dir:
            outfile = os.path.join(check_dir, '{}.csv'.format(desc))
            if os.path.exists(outfile):
                print('{} exists'.format(os.path.basename(outfile)))
                continue

        try:
            stack = get_terrain_image()
            stack = stack.clip(clip.first().geometry().buffer(1000))
            tile_pts = points.filterMetadata('MGRS_TILE', 'equals', tile)

            data = stack.reduceRegions(collection=tile_pts,
                                       reducer=ee.Reducer.mean(),
                                       scale=90,
                                       tileScale=16)

            bucket_file = os.path.join('dads_terrain', desc)

            task = ee.batch.Export.table.toCloudStorage(
                collection=data,
                description=desc,
                fileNamePrefix=bucket_file,
                bucket='wudr',
                fileFormat='CSV')

            task.start()
            print(desc)

        except ee.ee_exception.EEException as e:
            print('{}, waiting on '.format(e), desc, '......')
            time.sleep(600)
            task.start()
            print(desc)


def get_terrain_image():
    dem_coll = ee.Image('CGIAR/SRTM90_V4')
    dem = dem_coll.select('elevation')
    terrain = ee.Terrain.products(dem).select(['elevation', 'slope', 'aspect'])
    elev = terrain.select(['elevation'])

    tpi_500 = elev.subtract(elev.focal_mean(500, 'circle', 'meters')).add(0.5).rename('tpi_500')
    tpi_2500 = elev.subtract(elev.focal_mean(2500, 'circle', 'meters')).add(0.5).rename('tpi_2500')
    tpi_10000 = elev.subtract(elev.focal_mean(10000, 'circle', 'meters')).add(0.5).rename('tpi_10000')
    tpi_22500 = elev.subtract(elev.focal_mean(22500, 'circle', 'meters')).add(0.5).rename('tpi_22500')

    img = terrain.addBands([
        tpi_500,
        tpi_2500,
        tpi_10000,
        tpi_22500,
    ])
    return img


if __name__ == '__main__':
    is_authorized()

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/nas'

    _bucket = 'gs://wudr'

    # sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    # stations = 'ghcn_CANUSA_stations_mgrs'
    # stype = 'ghcn'
    # check = os.path.join(d, 'dads', 'dem', 'terrain', 'ghcn_stations')

    # NDBC buoys
    sites = os.path.join(d, 'climate', 'ndbc', 'ndbc_meta', 'ndbc_stations.csv')
    stations = 'ndbc_stations'
    check = os.path.join(d, 'dads', 'dem', 'terrain', 'ndbc_stations')
    stype = 'ndbc'

    # sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_17MAY2025_gap_mgrs.csv')
    # stations = 'madis_17MAY2025_gap_mgrs'
    # check = os.path.join(d, 'dads', 'dem', 'terrain', 'madis_stations')
    # stype = 'madis'

    kw = station_par_map(stype)
    bounds = (-180., 25., -60., 85.)
    sites_df = pd.read_csv(sites)
    w, s, e, n = bounds
    sites_df = sites_df[(sites_df[kw['lat']] < n) & (sites_df[kw['lat']] >= s)]
    sites_df = sites_df[(sites_df[kw['lon']] < e) & (sites_df[kw['lon']] >= w)]

    tiles = sites_df['MGRS_TILE'].unique().tolist()
    tiles = [m for m in tiles if isinstance(m, str)]
    mgrs_tiles = list(set(tiles))
    mgrs_tiles.sort()

    pt_buffer = 100
    pts = 'projects/ee-dgketchum/assets/dads/{}'.format(stations)
    file_ = '{}_{}'.format(stations, pt_buffer)

    export_terrain_features(file_prefix=file_, points_layer=pts, buffer_=pt_buffer,
                            tiles=mgrs_tiles, check_dir=check)

# ========================= EOF ====================================================================
