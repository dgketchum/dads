import os

import pandas as pd


def write_station_solrad(stations, tile_dir, station_out, shuffle=False, overwrite=False):
    stations = pd.read_csv(stations)
    stations.sort_index(inplace=True)

    if shuffle:
        stations = stations.sample(frac=1)

    tiles = stations['MGRS_TILE'].unique().tolist()

    for i, tile in enumerate(tiles, start=1):

        if not isinstance(tile, str):
            continue

        f = os.path.join(tile_dir, f'tile_{tile}.csv')

        if not os.path.exists(f):
            print(f'{tile} data not found')
            continue

        df = pd.read_csv(f, index_col=0).T

        for i, r in df.iterrows():

            sf = os.path.join(station_out, f'{i}.csv')

            if os.path.exists(sf) and not overwrite:
                continue

            r.to_csv(sf)

        print(tile)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    out = os.path.join(d, 'dem', 'terrain', 'station_data')

    # shapefile_path_ = os.path.join(d, 'met', 'stations', 'dads_stations_res_elev_mgrs.csv')
    shapefile_path_ = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    # shapefile_path_ = os.path.join(d, 'met', 'stations', 'madis_mgrs_28OCT2024.csv')

    # solrad_tiled = os.path.join(d, 'dem', 'rsun_tables', 'madis')
    solrad_tiled = os.path.join(d, 'dem', 'rsun_tables', 'ghcn')
    # solrad_tiled = os.path.join(d, 'dem', 'rsun_tables', 'madis_27OCT2024')

    solrad_out = os.path.join(d, 'dem', 'rsun_tables', 'station_rsun')

    write_station_solrad(shapefile_path_, solrad_tiled, solrad_out, shuffle=True, overwrite=False)

# ========================= EOF ====================================================================
