import os

import pandas as pd

COLS = ['latitude', 'longitude', 'aspect', 'elevation','slope', 'tpi_10000', 'tpi_22500',
        'tpi_2500', 'tpi_500']


def write_station_terrain(stations, tile_dir, station_out, shuffle=False, overwrite=False, index_col='fid',
                          glob_='dads_stations_elev_mgrs_100', coords=['latitude', 'longitude']):
    stations = pd.read_csv(stations)
    stations.index = stations[index_col]
    stations.sort_index(inplace=True)

    if shuffle:
        stations = stations.sample(frac=1)

    ts, ct, scaling, first, shape = None, 0, {}, True, None

    scaling['stations'] = []

    tiles = stations['MGRS_TILE'].unique().tolist()

    for i, tile in enumerate(tiles, start=1):

        if not isinstance(tile, str):
            continue

        f = os.path.join(tile_dir, f'{glob_}_{tile}.csv')

        if not os.path.exists(f):
            print(f'{tile} data not found')
            continue

        df = pd.read_csv(f)
        df = df.rename(columns={k: v for k, v in zip(coords, ['latitude', 'longitude'])})

        for i, r in df.iterrows():

            sf = os.path.join(station_out, f'{r['fid']}.csv')

            if os.path.exists(sf) and not overwrite:
                continue

            d = r[COLS + [index_col]]
            d.to_csv(sf)
        print(tile)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    out = os.path.join(d, 'dem', 'terrain', 'station_data')

    # fields = os.path.join(d, 'met', 'stations', 'dads_stations_res_elev_mgrs.csv')
    # ee_out = os.path.join(d, 'dem', 'terrain', 'madis_stations')
    # write_station_terrain(fields, ee_out, out, shuffle=True, overwrite=False, index_col='fid')
    #
    fields = os.path.join(d, 'met', 'stations', 'madis_mgrs_28OCT2024.csv')
    ee_out = os.path.join(d, 'dem', 'terrain', 'new_madis')
    glb = 'missing_madis_100'
    coords_ = ['lat', 'lon']
    write_station_terrain(fields, ee_out, out, shuffle=True, overwrite=False,
                          index_col='fid', glob_=glb, coords=coords_)

    fields = os.path.join(d, 'met', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    ee_out = os.path.join(d, 'dem', 'terrain', 'ghcn_stations')
    glb = 'ghcn_CANUSA_stations_mgrs_100'
    coords_ = ['LAT', 'LON']
    write_station_terrain(fields, ee_out, out, shuffle=True, overwrite=False,
                          index_col='STAID', coords=coords_)

# ========================= EOF ====================================================================
