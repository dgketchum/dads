import os
import sys
import time

import ee
import pandas as pd

sys.path.insert(0, os.path.abspath('../..'))
from extract.rs.earth_engine.ee_utils import is_authorized

sys.setrecursionlimit(2000)

BOUNDARIES = 'users/dgketchum/boundaries'


def export_terrain_features(file_prefix, points_layer, buffer_, tiles, check_dir=None):
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

            task = ee.batch.Export.table.toCloudStorage(
                collection=data,
                description=desc,
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


def export_dem(csv, check_dir=None):
    """"""
    df = pd.read_csv(csv)
    tiles = list(df['MGRS_TILE'])
    ned = ee.Image('USGS/SRTMGL1_003').select(['elevation'])
    elev = ee.Terrain.products(ned).select(['elevation'])
    mgrs = ee.FeatureCollection('users/dgketchum/boundaries/MGRS_TILE')

    for tile in tiles:

        desc = 'dem_{}'.format(tile)

        if check_dir:
            outfile = os.path.join(check_dir, '{}.tif'.format(desc))
            if os.path.exists(outfile):
                print('{} exists'.format(outfile))
                continue

        clip = mgrs.filterMetadata('MGRS_TILE', 'equals', tile)
        img = elev.clip(clip.first().geometry().buffer(1000))

        task = ee.batch.Export.image.toCloudStorage(
            image=img,
            description=desc,
            bucket='wudr',
            fileNamePrefix=desc,
            scale=250,
            crs='EPSG:5071',
            maxPixels=1e13)

        try:
            task.start()
            print(desc)
        except ee.ee_exception.EEException as e:
            print('{}, waiting on '.format(e), desc, '......')
            time.sleep(600)
            task.start()
            print(desc)


if __name__ == '__main__':
    is_authorized()

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    _bucket = 'gs://wudr'

    sites = os.path.join(d, 'dads', 'dem', 'w17_tiles.csv')
    sites = pd.read_csv(sites)['MGRS_TILE']
    mgrs_tiles = list(set(sites))
    mgrs_tiles.sort()

    chk = '/media/nvm/IrrigationGIS/dads/dem/dem_250'
    mgrs = '/media/research/IrrigationGIS/dads/training/w17_tiles.csv'
    # export_dem(mgrs, chk)

    pt_buffer = 100
    # stations = 'dads_stations_elev_mgrs'
    stations = 'ghcn_CANUSA_stations_mgrs'
    pts = 'projects/ee-dgketchum/assets/dads/{}'.format(stations)
    file_ = '{}_{}'.format(stations, pt_buffer)
    export_terrain_features(file_prefix=file_, points_layer=pts, buffer_=pt_buffer,
                            tiles=mgrs_tiles, check_dir=None)

# ========================= EOF ====================================================================
