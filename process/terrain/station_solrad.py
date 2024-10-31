import os

import pandas as pd
from concurrent.futures import ProcessPoolExecutor


def process_tile(tile, tile_dir, station_out, overwrite=False):
    if not isinstance(tile, str):
        return

    f = os.path.join(tile_dir, f'tile_{tile}.csv')

    if not os.path.exists(f):
        print(f'{tile} data not found')
        return

    df = pd.read_csv(f, index_col=0).T

    for i, r in df.iterrows():
        sf = os.path.join(station_out, f'{i}.csv')

        if os.path.exists(sf) and not overwrite:
            continue

        r.to_csv(sf)

    print(tile)


def write_station_solrad(stations, tile_dir, station_out, num_workers=2, shuffle=False, overwrite=False):

    stations = pd.read_csv(stations)
    stations.sort_index(inplace=True)

    if shuffle:
        stations = stations.sample(frac=1)

    tiles = stations['MGRS_TILE'].unique().tolist()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_tile, tiles, [tile_dir] * len(tiles),
                    [station_out] * len(tiles), [overwrite] * len(tiles))



if __name__ == '__main__':

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    out = os.path.join(d, 'dem', 'terrain', 'station_data')

    # shapefile_path_ = os.path.join(d, 'dads', 'met', 'stations', 'dads_stations_res_elev_mgrs.csv')
    shapefile_path_ = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    # shapefile_path_ = os.path.join(d,'dads',  'met', 'stations', 'madis_mgrs_28OCT2024.csv')

    # solrad_tiled = os.path.join(d, 'dads', 'dem', 'rsun_tables', 'madis')
    solrad_tiled = os.path.join(d, 'dads', 'dem', 'rsun_tables', 'ghcn')
    # solrad_tiled = os.path.join(d, 'dads', 'dem', 'rsun_tables', 'madis_27OCT2024')

    solrad_out = os.path.join(d, 'dem', 'rsun_tables', 'station_rsun')

    write_station_solrad(shapefile_path_, solrad_tiled, solrad_out, num_workers=1,
                         shuffle=True, overwrite=False)

# ========================= EOF ====================================================================
