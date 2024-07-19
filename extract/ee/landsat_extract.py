import os
import sys
import time
from datetime import datetime
import ee
import pandas as pd
import geopandas as gpd

from extract.ee.ee_utils import landsat_masked, is_authorized

sys.path.insert(0, os.path.abspath('../..'))
sys.setrecursionlimit(5000)

IRR = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'

EC_POINTS = 'users/dgketchum/flux_ET_dataset/stations'

STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']

SELECT = ['B{}'.format(b) for b in [2, 3, 4, 5, 6, 7, 10]]


def get_flynn():
    return ee.FeatureCollection(ee.Feature(ee.Geometry.Polygon([[-106.63372199162623, 46.235698473362476],
                                                                [-106.49124304875514, 46.235698473362476],
                                                                [-106.49124304875514, 46.31472036075997],
                                                                [-106.63372199162623, 46.31472036075997],
                                                                [-106.63372199162623, 46.235698473362476]]),
                                           {'key': 'Flynn_Ex'}))


def multipoint_landsat(shapefile, bucket=None, debug=False, check_dir=None, index='FID'):
    df = pd.read_csv(shapefile, parse_dates=True)

    fmt = '%m/%d/%Y %H:%M'
    df['record_start'] = [datetime.strptime(v, fmt) for v in df['record_start']]
    df['record_end'] = [datetime.strptime(v, fmt) for v in df['record_end']]

    for i, (fid, row) in enumerate(df.iterrows(), start=1):

        site = row[index]

        s, e = row['record_start'], row['record_end']
        print('{} of {}: {}'.format(i, df.shape[0], fid))

        for year in range(s.year, e.year + 1):

            desc = 'bands_{}_{}'.format(site, year)
            if check_dir:
                f = os.path.join(check_dir, '{}.csv'.format(desc))
                if os.path.exists(f):
                    continue

            point = ee.Geometry.Point([row['LON'], row['LAT']])
            geo = point.buffer(2000.)
            fc = ee.FeatureCollection(ee.Feature(geo, {index: site}))

            coll = landsat_masked(year, fc).select(SELECT)
            scenes = coll.aggregate_histogram('system:index').getInfo()

            first, bands = True, None
            selectors = [site]
            for img_id in scenes:

                splt = img_id.split('_')
                _names = ['{}_{}'.format('_'.join(splt[-3:]), b) for b in SELECT]

                selectors.append(_names)

                nd_img = coll.filterMetadata('system:index', 'equals', img_id).first().rename(_names)

                nd_img = nd_img.clip(fc.geometry())

                if first:
                    bands = nd_img
                    first = False
                else:
                    bands = bands.addBands([nd_img])

                if debug:
                    data = nd_img.sample(point, 30).getInfo()
                    print(data['features'])

            data = bands.reduceRegions(collection=fc,
                                       reducer=ee.Reducer.mean(),
                                       scale=30)

            task = ee.batch.Export.table.toCloudStorage(
                data,
                description=desc,
                bucket=bucket,
                fileNamePrefix=desc,
                fileFormat='CSV',
                selectors=selectors)

            try:
                task.start()
                print(desc)
            except ee.ee_exception.EEException:
                print('waiting on ', desc, '......')
                time.sleep(600)
                task.start()
                print(desc)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    is_authorized()
    bucket_ = 'wudr'

    fields = os.path.join(d, 'dads', 'met', 'stations', 'openet_gridwxcomp_input.csv')
    chk = os.path.join(d, 'dads', 'rs', 'gwx_stations')

    multipoint_landsat(fields, bucket_, debug=False, check_dir=chk, index='STATION_ID')

# ========================= EOF ====================================================================
