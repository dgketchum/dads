import os
import sys
from datetime import datetime, date

from numpy import ceil, linspace
from pprint import pprint

import ee
from numpy import ceil, linspace

sys.path.insert(0, os.path.abspath('..'))
from ee_utils import get_world_climate, landsat_composites, landsat_masked
from cdl import get_cdl

sys.setrecursionlimit(2000)

BOUNDARIES = 'users/dgketchum/boundaries'
IRRIGATION_TABLE = 'users/dgketchum/western_states_irr/NV_agpoly'
RF_ASSET = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'

TARGET_STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
E_STATES = ['ND', 'SD', 'NE', 'KS', 'OK', 'TX']

# list of years we have verified irrigated fields
YEARS = [1986, 1987, 1988, 1989, 1993, 1994, 1995, 1996, 1997,
         1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,
         2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
         2016, 2017, 2018, 2019]

TEST_YEARS = [2005]
ALL_YEARS = [x for x in range(1986, 2021)]


def request_band_extract(file_prefix, points_layer, region, years, buffer, diagnose=False, properties=None):
    """

    """
    roi = ee.FeatureCollection(region)
    points = ee.FeatureCollection(points_layer)
    points = points.map(lambda x: x.buffer(buffer))

    for yr in years:
        stack = stack_bands(yr, roi)

        # if tables are coming out empty, use this to find missing bands
        if diagnose:
            filtered = ee.FeatureCollection([points.first()])
            bad_ = []
            bands = stack.bandNames().getInfo()
            for b in bands:
                stack_ = stack.select([b])

                def sample_regions(i, points):
                    red = ee.Reducer.toCollection(i.bandNames())
                    reduced = i.reduceRegions(points, red, 30, stack_.select(b).projection())
                    fc = reduced.map(lambda f: ee.FeatureCollection(f.get('features'))
                                     .map(lambda q: q.copyProperties(f, None, ['features'])))
                    return fc.flatten()

                data = sample_regions(stack_, filtered)
                try:
                    print(b, data.getInfo()['features'][0]['properties'][b])
                except Exception as e:
                    print(b, 'not there', e)
                    bad_.append(b)
            print(bad_)
            return None

        data = stack.reduceRegions(collection=points,
                                   reducer=ee.Reducer.mean(),
                                   scale=30,
                                   tileScale=16)

        desc = '{}_{}'.format(file_prefix, yr)
        task = ee.batch.Export.table.toCloudStorage(
            collection=data,
            description=desc,
            bucket='wudr',
            fileFormat='CSV')

        task.start()
        print(desc, yr)


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
    ned = ee.Image('USGS/3DEP/10m')
    terrain = ee.Terrain.products(ned).select(['elevation', 'slope', 'aspect']).reduceResolution(
        ee.Reducer.mean()).reproject(crs=proj['crs'], scale=30)

    elev = terrain.select(['elevation'])
    tpi_1250 = elev.subtract(elev.focal_mean(1250, 'circle', 'meters')).add(0.5).rename('tpi_1250')
    tpi_250 = elev.subtract(elev.focal_mean(250, 'circle', 'meters')).add(0.5).rename('tpi_250')
    tpi_150 = elev.subtract(elev.focal_mean(150, 'circle', 'meters')).add(0.5).rename('tpi_150')
    input_bands = input_bands.addBands([coords, terrain, tpi_1250, tpi_250, tpi_150])

    nlcd = ee.Image('USGS/NLCD/NLCD2011').select('landcover').reproject(crs=proj['crs'], scale=30).rename('nlcd')

    cdl_cult, cdl_crop, cdl_simple = get_cdl(yr)

    gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    occ_pos = gsw.select('occurrence').gt(0)
    water = occ_pos.unmask(0).rename('gsw')

    input_bands = input_bands.addBands([nlcd, cdl_cult, cdl_crop, cdl_simple, water])

    input_bands = input_bands.clip(roi)


    return input_bands


def is_authorized():
    try:
        ee.Initialize(project='ee-dgketchum')
        print('Authorized')
    except Exception as e:
        print('You are not authorized: {}'.format(e))
        exit(1)
    return None


if __name__ == '__main__':
    is_authorized()
    _bucket = 'gs://wudr'
    buffer_ = 2000
    file_ = 'bands_{}'.format(buffer_)
    pts = 'projects/ee-dgketchum/assets/dads/openet_gwx'
    geo = 'users/dgketchum/boundaries/western_states_expanded_union'
    years_ = list(range(2000, 2021))
    years_.reverse()
    request_band_extract(file_, pts, region=geo, years=years_[:1], buffer=buffer_, diagnose=False)

# ========================= EOF ====================================================================
