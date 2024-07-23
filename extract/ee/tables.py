import os

import numpy as np
from pandas import read_csv, concat, errors


def concatenate_band_extract(root, out_dir, glob='None', sample=None, nd_only=False):
    l = [os.path.join(root, x) for x in os.listdir(root) if glob in x]
    l.sort()
    first = True
    for csv in l:
        year = csv.split('.')[0][-4:]
        try:
            if first:
                df = read_csv(csv, index_col='STATION_ID')
                sites = np.unique(df.index)
                df.drop(columns=['.geo', 'system:index'], inplace=True)
                df.dropna(how='all', inplace=True, axis=0)
                df['year'] = year
                first = False
            else:
                c = read_csv(csv, index_col='STATION_ID')
                c.drop(columns=['.geo', 'system:index'], inplace=True)
                c.dropna(how='all', inplace=True, axis=0)
                c['year'] = year
                df = concat([df, c], sort=False)
        except errors.EmptyDataError:
            print('{} is empty'.format(csv))
            pass

    if nd_only:
        drop_cols = []
        for c in list(df.columns):
            if 'evi' in c:
                drop_cols.append(c)
            if 'gi' in c:
                drop_cols.append(c)
            if 'nw' in c:
                drop_cols.append(c)
        df = df.drop(columns=drop_cols)

    if sample:
        df = df.sample(frac=sample).reset_index(drop=True)

    out_file = os.path.join(out_dir, '{}.csv'.format(glob))

    print('size: {}'.format(df.shape))
    print('file: {}'.format(out_file))
    df.to_csv(out_file, index=False)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    fields = os.path.join(d, 'dads', 'met', 'stations', 'openet_gridwxcomp_input.csv')
    extracts = os.path.join(d, 'dads', 'rs', 'gwx_stations', 'ee_extracts')
    out = os.path.join(d, 'dads', 'rs', 'gwx_stations', 'concatenated')

    concatenate_band_extract(extracts, out, 'bands')

# ========================= EOF ====================================================================
