import os
import sys
import time

import ee
import pandas as pd

sys.path.insert(0, os.path.abspath('../..'))
from extract.rs.earth_engine.ee_utils import is_authorized

sys.setrecursionlimit(2000)

BOUNDARIES = 'users/dgketchum/boundaries'


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
    export_dem(mgrs, chk)

# ========================= EOF ====================================================================
