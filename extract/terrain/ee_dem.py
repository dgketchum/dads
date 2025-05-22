import os
import sys
import time

import ee
import pandas as pd

sys.path.insert(0, os.path.abspath('../..'))
from extract.rs.earth_engine.ee_utils import is_authorized

sys.setrecursionlimit(2000)

BOUNDARIES = 'users/dgketchum/boundaries'


def export_dem(tiles, check_dir=None):
    """"""
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
    station_set = 'madis'

    if station_set == 'madis':
        stations = 'madis_17MAY2025_gap_mgrs'
        sites = os.path.join(d, 'dads', 'met', 'stations', f'{stations}.csv')
        chk = os.path.join(d, 'dads', 'rs', 'landsat', stations)


    elif station_set == 'ghcn':
        stations = 'ghcn_CANUSA_stations_mgrs'
        sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
        chk = os.path.join(d, 'dads', 'rs', 'ghcn_stations', 'landsat', 'tiles')

    else:
        raise NotImplementedError

    bounds = (-180., 25., -60., 85.)
    sites_df = pd.read_csv(sites)
    sites_df = sites_df[(sites_df['latitude'] < bounds[3]) & (sites_df['latitude'] >= bounds[1])]
    sites_df = sites_df[(sites_df['longitude'] < bounds[2]) & (sites_df['longitude'] >= bounds[0])]

    tiles = sites_df['MGRS_TILE'].unique().tolist()
    tiles = [m for m in tiles if isinstance(m, str)]
    mgrs_tiles = list(set(tiles))
    mgrs_tiles.sort()

    chk = '/media/nvm/IrrigationGIS/dads/dem/dem_250'
    export_dem(mgrs_tiles, chk)

# ========================= EOF ====================================================================
