import os
import re

import pandas as pd
from pandarallel import pandarallel

from utils.calc_eto import calc_asce_params
from utils.station_parameters import station_par_map

RENAME = {
    'date': 'date',
    'day': 'doy',
    'TAvg (C)': 'mean_temp',
    'TMax (C)': 'max_temp',
    'TMin (C)': 'min_temp',
    'Vapor Pres (kPa)': 'ea',
    'Rs (w/m2)': 'rsds',
    'Precip (mm)': 'prcp',
    'ws_2m (m/s)': 'wind'
}


def read_gwx(stations, target_sites, gwx_src, gwx_dst, overwrite=False):
    """"""
    network_sites = pd.read_csv(stations)
    dads_sites = pd.read_csv(target_sites, index_col='index')

    kw = station_par_map('openet')

    for i, row in network_sites.iterrows():

        # handle non-unique station names as in join_station_lists
        try:
            _ = int(row['original_network_id'])
            int_name = True
        except ValueError:
            int_name = False

        if int_name:
            base_name = re.sub(r'\W+', '_', row['original_station_name'])[:10].upper()
        else:
            base_name = row['original_network_id'].upper()

        assert base_name in dads_sites.index

        out_file = os.path.join(gwx_dst, '{}.csv'.format(base_name))

        if os.path.exists(out_file) and not overwrite:
            print(os.path.basename(out_file), 'exists, skipping')
            continue

        # anemometer height already adjusted
        lon, lat, elv, zw = row[kw['lon']], row[kw['lat']], row[kw['elev']], 2.0

        in_file = os.path.join(gwx_src, '{}_data.xlsx'.format(row['STATION_ID']))
        if not os.path.exists(in_file):
            print('{} does not exist, skipping'.format(in_file))
            continue

        df = pd.read_excel(in_file, index_col=0)
        df['date'] = df.index
        df.rename(columns=RENAME, inplace=True)
        df = df[[v for k, v in RENAME.items()]].copy()
        df['rsds'] *= 0.0864
        asce_params = df.parallel_apply(calc_asce_params, lat=lat, elev=elv, zw=zw, axis=1)
        # asce_params = df.apply(calc_asce_params, lat=lat, elev=elv, zw=zw, axis=1)

        try:
            df[['mean_temp', 'vpd', 'rn', 'u2', 'eto']] = pd.DataFrame(asce_params.tolist(),
                                                                       index=df.index)
        except ValueError as e:
            print(e)
            return None

        df.to_csv(out_file, index=False)
        print(os.path.basename(out_file))


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS')

    pandarallel.initialize(nb_workers=6)

    dads_stations = os.path.join(d, 'dads', 'met', 'stations', 'dads_stations_elev_mgrs.csv')
    gwx_list = os.path.join(d, 'dads', 'met', 'stations', 'openet_gridwxcomp_input.csv')

    gwx_src_ = os.path.join(d, 'climate', 'gridwxcomp', 'station_data')
    gwx_dst_ = os.path.join(d, 'dads', 'met', 'obs', 'gwx')

    read_gwx(gwx_list, dads_stations, gwx_src_, gwx_dst_, overwrite=False)

# ========================= EOF ====================================================================
