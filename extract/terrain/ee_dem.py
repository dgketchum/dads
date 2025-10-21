import os
import sys
import time

import ee
import pandas as pd

sys.path.insert(0, os.path.abspath('../..'))
from extract.rs.earth_engine.ee_utils import is_authorized
from utils.station_parameters import station_par_map

sys.setrecursionlimit(2000)

BOUNDARIES = 'users/dgketchum/boundaries'


def export_dem(tiles, check_dir=None, crs_epsg='5071', dry_run=False):
    """"""

    mgrs = ee.FeatureCollection('users/dgketchum/boundaries/MGRS_TILE')

    for tile in tiles:

        desc = 'dem_{}'.format(tile)

        if check_dir:
            outfile = os.path.join(check_dir, '{}.tif'.format(desc))
            if os.path.exists(outfile):
                print('{} exists'.format(outfile))
                continue

            else:
                if dry_run:
                    print(f'export {outfile}')
                    continue

        clip = mgrs.filterMetadata('MGRS_TILE', 'equals', tile)
        coords = clip.getInfo()['features'][0]['geometry']['coordinates'][0]
        max_lat = max([abs(c[1]) for c in coords])

        if max_lat < 59.0:
            # Good to 59 degrees latitude
            ned = ee.Image('USGS/SRTMGL1_003').select(['elevation'])
            elev = ee.Terrain.products(ned).select(['elevation'])
        else:
            # Good in arctic regions
            dataset = ee.ImageCollection('JAXA/ALOS/AW3D30/V3_2')
            elev = dataset.select('DSM').mosaic()

        img = elev.clip(clip.first().geometry().buffer(1000))
        bucket_file = os.path.join(f'dem_{crs_epsg}', desc)

        task = ee.batch.Export.image.toCloudStorage(
            image=img,
            description=desc,
            bucket='wudr',
            fileNamePrefix=bucket_file,
            scale=250,
            crs=f'EPSG:{crs_epsg}',
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

    dem_d = '/data/ssd2/dads/dem'
    d = '/home/dgketchum/data/IrrigationGIS'

    _bucket = 'gs://wudr'
    station_set = 'ghcn'
    zone = 'north'

    if station_set == 'madis':
        stations = 'madis_17MAY2025_gap_mgrs'
        sites = os.path.join(d, 'dads', 'met', 'stations', f'{stations}.csv')

    elif station_set == 'ghcn':
        stations = 'ghcn_CANUSA_stations_mgrs'
        sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_stations_mgrs_country.csv')

    elif station_set == 'ndbc':
        stations = 'ndbc_stations'
        sites = os.path.join(d, 'climate', 'ndbc', 'ndbc_meta', 'ndbc_stations.csv')
    
    else:
        raise ValueError

    if zone == 'north':
        bounds = (-180., 49., -60., 85.)
        epsg = '3978'
        chk = os.path.join(dem_d, f'dem_{epsg}')

    elif zone == 'south':
        bounds = (-180., 23., -60., 49.)
        epsg = '5071'
        chk = os.path.join(dem_d, f'dem_{epsg}')

    else:
        raise ValueError

    sites_df = pd.read_csv(sites)

    kw = station_par_map(station_set)

    sites_df = sites_df[(sites_df[kw['lat']] < bounds[3]) & (sites_df[kw['lat']] >= bounds[1])]
    sites_df = sites_df[(sites_df[kw['lon']] < bounds[2]) & (sites_df[kw['lon']] >= bounds[0])]

    if zone == 'north':
        sites_df = sites_df[sites_df['AFF_ISO'] == 'CA']

    tiles = sites_df['MGRS_TILE'].unique().tolist()
    tiles = [m for m in tiles if isinstance(m, str)]
    mgrs_tiles = list(set(tiles))
    mgrs_tiles.sort()

    export_dem(mgrs_tiles, chk, epsg, dry_run=False)

# ========================= EOF ====================================================================
