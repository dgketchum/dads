import os
import json

import numpy as np
import pandas as pd
from pandas import read_csv


def download_ghcn(station_id, file_dst):
    url = 'https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{}.csv.gz'.format(station_id)

    df = read_csv(url, header=None, usecols=[1, 2, 3, 6])
    df.columns = ['DATE', 'PARAM', 'VALUE', 'FLAG']
    params = list(np.unique(df['PARAM']))
    target_cols = ['TMIN', 'TMAX', 'PRCP']

    if not any([p in params for p in target_cols]):
        print('records missing parameters')
        return []

    df = df.pivot_table(index='DATE', columns=['PARAM'],
                        values='VALUE', aggfunc='first').reindex()

    params_ = [c for c in target_cols if c in df.columns]
    if len(params_) == 0:
        return []

    df = df[params_]
    df.index = [str(dt) for dt in df.index]
    df.to_csv(file_dst)
    return params_


def get_station_data(inventory, out_dir, bounds=(-125., 25., -60., 49.), overwrite=False, tracker=None):
    with open(inventory) as fh:
        data = fh.readlines()

    if tracker:
        if os.path.exists(tracker):
            with open(tracker, 'r') as f:
                meta = json.load(f)
        else:
            meta = {'PRCP': [], 'TMIN': [], 'TMAX': []}

    stations = pd.DataFrame([row.split() for row in data],
                            columns=['station', 'latitude', 'longitude', 'element', 'firstyear', 'lastyear'])

    stations['latitude'] = pd.to_numeric(stations['latitude'])
    stations['longitude'] = pd.to_numeric(stations['longitude'])
    stations['firstyear'] = pd.to_numeric(stations['firstyear'])
    stations['lastyear'] = pd.to_numeric(stations['lastyear'])

    stations = stations[(stations['latitude'] < bounds[3]) & (stations['latitude'] >= bounds[1])]
    stations = stations[(stations['longitude'] < bounds[2]) & (stations['longitude'] >= bounds[0])]

    for sid in stations['station'].unique():
        out_file = os.path.join(out_dir, f'{sid}.csv')
        if os.path.exists(out_file) and not overwrite:
            print(sid, 'exists, skipping')
            continue

        try:
            params = download_ghcn(sid, out_file)
        except Exception as e:
            print(e)
            continue

        if tracker and len(params) > 0:
            [meta[p].append(sid) for p in params]

        print(sid)

    if tracker:
        with open(tracker, 'w') as fp:
            json.dump(meta, fp, indent=4)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    ghcn = os.path.join(d, 'climate', 'ghcn')
    inventroy_ = os.path.join(ghcn, 'ghcnd-inventory.txt')
    rec_dir = os.path.join(ghcn, 'station_data')
    tracking = os.path.join(ghcn, 'downloaded_ghcn.json')
    get_station_data(inventroy_, rec_dir, overwrite=False, tracker=tracking)

# ========================= EOF ====================================================================
