import os
import time
import datetime

import ee


def is_authorized(project='ee-dgketchum'):
    try:
        ee.Initialize(project=project)
        print('Authorized')
    except Exception as e:
        print('You are not authorized: {}'.format(e))
        exit(1)
    return None


def export_rtma_to_gcs(bucket_name, start_date, end_date, params):
    is_authorized()

    rtma_geo = ee.Geometry.Polygon([[
        [-127.0, 19.0],
        [-69.0, 20.0],
        [-58., 55.],
        [-139., 53.],
        [-127., 19.],
    ]])

    rtma = ee.ImageCollection('NOAA/NWS/RTMA') \
        .filter(ee.Filter.date(start_date, end_date)) \
        .select(list(params.keys()))

    start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%SZ')
    end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%SZ')
    total_days = (end_datetime - start_datetime).days

    date_list = [start_datetime + datetime.timedelta(days=d) for d in range(total_days)]
    date_list.reverse()

    for i, date in enumerate(date_list):

        if i == 0:
            continue

        daily_rtma = rtma.filter(ee.Filter.date(date, date + datetime.timedelta(days=1)))

        bands = []

        for k, v in params.items():

            if k == 'PRES':
                band = daily_rtma.select('PRES').mean().divide(100).int()
                daily_image = band
                continue

            elif k == 'ACPC01':
                img = daily_rtma.select('ACPC01').sum().multiply(100).int()

            else:
                img = daily_rtma.select(k).mean().multiply(100).int()

            bands.append(img)

        daily_image = daily_image.addBands(bands)

        filename = f'RTMA_{date.strftime("%Y%m%d")}'

        task = ee.batch.Export.image.toCloudStorage(
            image=daily_image,
            description=filename,
            bucket=bucket_name,
            fileNamePrefix=filename,
            scale=2500,
            maxPixels=1e13,
            region=rtma_geo,
            formatOptions={
                'cloudOptimized': True
            }
        )

        try:
            task.start()
            print(f'Exporting {filename} ({i + 1}/{total_days})')

        except ee.ee_exception.EEException as e:
            print('{}, waiting on '.format(e), filename, '......')
            time.sleep(600)
            task.start()
            print(filename)


if __name__ == '__main__':
    start_date_ = '2012-01-01T00:00:00Z'
    end_date_ = '2025-01-10T00:00:00Z'
    bucket_name_ = 'wudr'
    params_ = {
        'PRES': 1000,
        'TMP': 100,
        'DPT': 100,
        'UGRD': 100,
        'VGRD': 100,
        'SPFH': 100,
        'WDIR': 100,
        'WIND': 100,
        'TCDC': 100,
        'ACPC01': 100
    }

    export_rtma_to_gcs(bucket_name_, start_date_, end_date_, params_)
# ========================= EOF ====================================================================
