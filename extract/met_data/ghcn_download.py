import os

# import boto3
import pandas as pd

# s3 = boto3.resource('s3', region_name='us-east-1')
# bucket = s3.Bucket('noaa-ghcn-pds')


def get_station_data(inventory, out_dir, bounds=(-125., 25., -96., 49.), overwrite=False):

    with open(inventory) as fh:
        data = fh.readlines()

    df = pd.DataFrame([row.split() for row in data],
                      columns=['station', 'latitude', 'longitude', 'element', 'firstyear', 'lastyear'])

    df['latitude'] = pd.to_numeric(df['latitude'])
    df['longitude'] = pd.to_numeric(df['longitude'])
    df['firstyear'] = pd.to_numeric(df['firstyear'])
    df['lastyear'] = pd.to_numeric(df['lastyear'])

    df = df[(df['latitude'] < bounds[3]) & (df['latitude'] >= bounds[1])]
    df = df[(df['longitude'] < bounds[2]) & (df['longitude'] >= bounds[0])]

    rhav = df[df['element'] == 'RHAV']
    rhmn = df[df['element'] == 'RHMN']
    rhmx = df[df['element'] == 'RHMX']

    station_path = 'parquet/by_station/STATION=AUM00011343/'

    df = pd.read_parquet(f's3://noaa-ghcn-pds/{station_path}')

    result = df[df['ELEMENT'] == 'TMAX'][['ID', 'DATE', 'DATA_VALUE', 'ELEMENT']]

    print(result)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    inventroy_ = os.path.join(d, 'climate', 'ghcn', 'ghcnd-inventory.txt')
    rec_dir = os.path.join(d, 'dads', 'met', 'ghcn')
    get_station_data(inventroy_, rec_dir, overwrite=True)

# ========================= EOF ====================================================================
