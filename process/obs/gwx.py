import os

import pandas as pd

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


def read_gwx(stations, gwx_src, gwx_dst, overwrite=False):
    """"""
    sites = pd.read_csv(stations)

    kw = station_par_map('openet')

    for i, row in sites.iterrows():

        out_file = os.path.join(gwx_dst, '{}.csv'.format(row['original_network_id']))

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
        df = df[[v for k, v in RENAME.items()]]
        df['rsds'] *= 0.0864
        # asce_params = daily_df.parallel_apply(calc_asce_params, lat=lat_, elev=elev_, zw=10, axis=1)
        asce_params = df.apply(calc_asce_params, lat=lat, elev=elv, zw=zw, axis=1)

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

    # pandarallel.initialize(nb_workers=6)

    fields = os.path.join(d, 'dads', 'met', 'stations', 'dads_stations.csv')
    gwx_list = os.path.join(d, 'dads', 'met', 'stations', 'openet_gridwxcomp_input.csv')

    gwx_src_ = os.path.join(d, 'climate', 'gridwxcomp', 'station_data')
    gwx_dst_ = os.path.join(d, 'dads', 'met', 'obs', 'gwx')

    read_gwx(gwx_list, gwx_src_, gwx_dst_, overwrite=True)

# ========================= EOF ====================================================================
