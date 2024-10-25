import os
import sys
import time

import ee
import pandas as pd

sys.path.insert(0, os.path.abspath('../..'))
from extract.rs.earth_engine.cdl import get_cdl
from extract.rs.earth_engine.ee_utils import is_authorized, landsat_composites

sys.setrecursionlimit(2000)

BOUNDARIES = 'users/dgketchum/boundaries'


def request_band_extract(file_prefix, points_layer, region, years, buffer, tiles, check_dir=None):
    """

    """
    tasks = None

    # if there are tasks in process:
    # earthengine task list | grep -E '(READY|COMPLETED)' | awk '{print $3}' > processing.txt

    processing = os.path.join(os.path.dirname(__file__), 'processing.txt')
    if os.path.exists(processing):
        with open(processing, 'r') as f:
            tasks = f.read().splitlines()

        tasks = [os.path.join(check_dir, '{}.csv'.format(t)) for t in tasks]

    roi = ee.FeatureCollection(region)
    points = ee.FeatureCollection(points_layer)
    points = points.map(lambda x: x.buffer(buffer))

    failed = {}
    mgrs = ee.FeatureCollection('users/dgketchum/boundaries/MGRS_TILE')

    for tile in tiles:

        clip = mgrs.filterMetadata('MGRS_TILE', 'equals', tile)

        for yr in years:
            desc = '{}_{}_{}'.format(file_prefix, yr, tile)
            if check_dir:
                outfile = os.path.join(check_dir, '{}.csv'.format(desc))
                if os.path.exists(outfile):
                    print('{} exists'.format(os.path.basename(outfile)))
                    continue
                elif tasks is not None:
                    if desc in tasks:
                        print('{} is processing'.format(os.path.basename(outfile)))
                        continue
                else:
                    pass

            try:
                stack = stack_bands(yr, roi)
                stack = stack.clip(clip.first().geometry().buffer(1000))
                tile_pts = points.filterMetadata('MGRS_TILE', 'equals', tile)

                data = stack.reduceRegions(collection=tile_pts,
                                           reducer=ee.Reducer.mean(),
                                           scale=30,
                                           tileScale=16)

                task = ee.batch.Export.table.toCloudStorage(
                    collection=data,
                    description=desc,
                    bucket='wudr',
                    fileFormat='CSV')

                task.start()
                print(desc, yr)

            except ee.ee_exception.EEException as e:
                print('{}, waiting on '.format(e), desc, '......')
                time.sleep(600)
                task.start()
                print(desc)

            except Exception as e:
                print(tile, yr, e)
                if tile not in failed.keys():
                    failed[tile] = [str(yr)]
                else:
                    failed[tile].append(str(yr))
                continue


def stack_bands(yr, roi):
    """
    Create a stack of bands for the year and region of interest specified.
    :param yr:
    :param southern
    :param roi:
    :return:
    """

    winter_s, winter_e = '{}-01-01'.format(yr), '{}-03-01'.format(yr),
    spring_s, spring_e = '{}-03-01'.format(yr), '{}-05-01'.format(yr),
    late_spring_s, late_spring_e = '{}-05-01'.format(yr), '{}-07-15'.format(yr)
    summer_s, summer_e = '{}-07-15'.format(yr), '{}-09-30'.format(yr)
    fall_s, fall_e = '{}-09-30'.format(yr), '{}-12-31'.format(yr)

    prev_s, prev_e = '{}-05-01'.format(yr - 1), '{}-09-30'.format(yr - 1),
    p_summer_s, p_summer_e = '{}-07-15'.format(yr - 1), '{}-09-30'.format(yr - 1)

    pprev_s, pprev_e = '{}-05-01'.format(yr - 2), '{}-09-30'.format(yr - 2),
    pp_summer_s, pp_summer_e = '{}-07-15'.format(yr - 2), '{}-09-30'.format(yr - 2)

    periods = [('gs', spring_e, fall_s),
               ('0', winter_s, spring_s),
               ('1', spring_s, spring_e),
               ('2', late_spring_s, late_spring_e),
               ('3', summer_s, summer_e),
               ('4', fall_s, fall_e),
               ('m1', prev_s, prev_e),
               ('3_m1', p_summer_s, p_summer_e),
               ('m2', pprev_s, pprev_e),
               ('3_m2', pp_summer_s, pp_summer_e)]

    first = True
    for name, start, end in periods:
        prev = 'm' in name
        bands = landsat_composites(yr, start, end, roi, name, composites_only=prev)
        if first:
            input_bands = bands
            proj = bands.select('B2_gs').projection().getInfo()
            first = False
        else:
            input_bands = input_bands.addBands(bands)

    integrated_composite_bands = []

    for feat in ['nd', 'gi', 'nw', 'evi']:
        periods = [x for x in range(2, 5)]
        # periods = [x for x in range(2, 4)]
        c_bands = ['{}_{}'.format(feat, p) for p in periods]
        b = input_bands.select(c_bands).reduce(ee.Reducer.sum()).rename('{}_int'.format(feat))

        integrated_composite_bands.append(b)

    input_bands = input_bands.addBands(integrated_composite_bands)

    coords = ee.Image.pixelLonLat().rename(['lon', 'lat']).resample('bilinear').reproject(crs=proj['crs'], scale=30)

    input_bands = input_bands.addBands([coords])

    nlcd = ee.Image('USGS/NLCD/NLCD2011').select('landcover').reproject(crs=proj['crs'], scale=30).rename('nlcd')

    cdl_cult, cdl_crop, cdl_simple = get_cdl(yr)

    gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    occ_pos = gsw.select('occurrence').gt(0)
    water = occ_pos.unmask(0).rename('gsw')

    input_bands = input_bands.addBands([nlcd, cdl_cult, cdl_crop, cdl_simple, water])

    input_bands = input_bands.clip(roi)

    return input_bands


def extract_modis(glb, points_layer, years, check_dir=None):
    """
    """
    points = ee.FeatureCollection(points_layer)

    for yr in years:
        dt = pd.date_range(f'{yr}-01-01', f'{yr}-12-31', freq='D')

        for d in dt:

            d_str = d.strftime('%Y_%m_%d')
            desc = '{}_{}'.format(glb, d_str)

            if check_dir:
                outfile = os.path.join(check_dir, '{}.csv'.format(desc))
                if os.path.exists(outfile):
                    print('{} exists'.format(outfile))
                    continue

            stack = ee.Image('MODIS/061/MCD18A1/{}'.format(d_str)).select('DSR').rename('DSR')
            data = stack.sampleRegions(
                collection=points,
                scale=30,
                tileScale=16)

            task = ee.batch.Export.table.toCloudStorage(
                collection=data,
                description=desc,
                selectors=['DSR', 'fid'],
                bucket='wudr',
                fileFormat='CSV')

            try:
                task.start()
                print(desc)
            except ee.ee_exception.EEException as e:
                if 'Image.load' in e.args[0]:
                    print(e, desc, 'Image load failure')
                    continue
                else:
                    print('waiting on ', desc, '......')
                    time.sleep(600)
                    task.start()


def shapely_to_ee_polygon(shapely_geom):
    geojson = shapely_geom.__geo_interface__
    return ee.Geometry.Polygon(geojson)


if __name__ == '__main__':
    is_authorized()

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    _bucket = 'gs://wudr'

    w17 = os.path.join(d, 'dads', 'dem', 'w17_tiles.csv')
    w17 = pd.read_csv(w17)['MGRS_TILE'].astype(str)
    w17 = list(set(w17))

    # sites = os.path.join(d, 'dads', 'met', 'stations', 'dads_stations_res_elev_mgrs.csv')
    # sites = pd.read_csv(sites)['MGRS_TILE'].astype(str)
    # mgrs_tiles = list(set(sites))
    # mgrs_tiles = [t for t in mgrs_tiles if t not in w17 and t != 'nan']
    # mgrs_tiles.sort()
    # mgrs_tiles.reverse()

    stations = 'ghcn_CANUSA_stations_mgrs'
    pts = 'projects/ee-dgketchum/assets/dads/{}'.format(stations)

    geo = 'users/dgketchum/boundaries/western_states_expanded_union'
    years_ = list(range(1990, 2024))
    years_.reverse()

    failed = []
    chk = os.path.join(d, 'dads', 'rs', 'ghcn_stations', 'landsat', 'tiles')
    for buffer_ in [500]:
        file_ = '{}_{}'.format(stations, buffer_)
        request_band_extract(file_, pts, region=geo, years=years_[1:], buffer=buffer_, check_dir=chk, tiles=w17)

    # chk = os.path.join(d, 'dads', 'rs', 'dads_stations', 'modis')
    # extract_modis(stations, pts, years=years_, check_dir=chk)

# ========================= EOF ====================================================================
