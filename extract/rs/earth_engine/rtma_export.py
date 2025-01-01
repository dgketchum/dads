import datetime
import time

import ee

from ee_utils import is_authorized


def export_rtma_to_gcs(bucket_name, start_date, end_date, params):
    is_authorized()

    rtma = ee.ImageCollection('NOAA/NWS/RTMA') \
        .filter(ee.Filter.date(start_date, end_date)) \
        .select(params)

    start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%SZ')
    end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%SZ')
    total_hours = int((end_datetime - start_datetime).total_seconds() / 3600)

    timestamps = [start_datetime + datetime.timedelta(hours=i) for i in range(total_hours)]

    for i, timestamp in enumerate(timestamps):
        image = rtma.filter(ee.Filter.date(timestamp, timestamp + datetime.timedelta(hours=1))).first()

        filename = f'RTMA_{timestamp.strftime("%Y%m%d_%H%M%S")}'

        task = ee.batch.Export.image.toCloudStorage(
            image=image,
            description=filename,
            bucket=bucket_name,
            fileNamePrefix=filename,
            region=image.geometry(),
            scale=2500,
            formatOptions={
                'cloudOptimized': True
            }
        )

        try:
            task.start()
            print(f'Exporting {filename} ({i + 1}/{total_hours})')

        except ee.ee_exception.EEException as e:
            print('{}, waiting on '.format(e), filename, '......')
            time.sleep(600)
            task.start()
            print(filename)


if __name__ == '__main__':
    start_date_ = '2011-01-01T00:00:00Z'
    end_date_ = '2024-12-31T18:00:00Z'
    bucket_name_ = 'wudr'
    params_ = [
        'HGT',
        'PRES',
        'TMP',
        'DPT',
        'UGRD',
        'VGRD',
        'SPFH',
        'WDIR',
        'WIND',
        'GUST',
        'VIS',
        'TCDC',
        'ACPC01'
    ]

    export_rtma_to_gcs(bucket_name_, start_date_, end_date_, params_)
# ========================= EOF ====================================================================
