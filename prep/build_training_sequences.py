import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

TERRAIN_FEATURES = ['slope', 'aspect', 'elevation', 'tpi_1250', 'tpi_250', 'tpi_150']


def join_training(stations, ts_dir, landsat_dir, dem_dir, out_dir, scaling_json=None, var='rsds',
                  bounds=None, shuffle=False, overwrite=False, hourly=False, sample_frac=1.0):
    """"""

    stations = pd.read_csv(stations)
    stations.index = stations['fid']
    stations.sort_index(inplace=True)

    if bounds:
        w, s, e, n = bounds
        stations = stations[(stations['latitude'] < n) & (stations['latitude'] >= s)]
        stations = stations[(stations['longitude'] < e) & (stations['longitude'] >= w)]

    if shuffle:
        stations = stations.sample(frac=sample_frac)

    ts, ct, scaling, first, shape = None, 0, {}, True, None
    missing = {'sol_file': 0, 'station_file': 0, 'rs_file': 0, 'snotel': 0, 'rs_obs_time_misalign': 0,
               'sol_fid': 0}

    scaling['stations'] = []

    tiles = stations['MGRS_TILE'].unique()
    for tile in tqdm(tiles, total=len(tiles)):

        # if tile not in ['12TUP', '12TVP', '12TUN', '12TVN']:
        #     continue

        sol_file = os.path.join(dem_dir, 'tile_{}.csv'.format(tile))

        try:
            sol_df = pd.read_csv(sol_file, index_col=0)
        except FileNotFoundError:
            missing['sol_file'] += 1
            continue

        tile_sites = stations[stations['MGRS_TILE'] == tile]

        for i, (f, row) in enumerate(tile_sites.iterrows(), start=1):

            # if f != 'CAMINO':
            #     continue

            outfile = os.path.join(out_dir, '{}.csv'.format(f))
            if os.path.exists(outfile) and not overwrite:
                continue

            if f in stations['orig_netid'].tolist():
                fid = f
            else:
                fid = row['orig_netid']

            if row['source'] == 'snotel':
                missing['snotel'] += 1
                continue

            sta_file = os.path.join(ts_dir, '{}.csv'.format(f))
            if not os.path.exists(sta_file):
                missing['station_file'] += 1
                continue

            ts = pd.read_csv(sta_file, index_col='Unnamed: 0', parse_dates=True)

            # training depends on having the first three columns like so
            feats = [c for c in ts.columns if c.endswith('_nl') or c.endswith('_nl_hr')]
            feats = [c for c in feats if var not in c]
            ts = ts[[f'{var}_obs', f'{var}_gm', f'{var}_nl'] + feats]
            try:
                ts.loc[:, 'lat'], ts.loc[:, 'lon'] = row['latitude'], row['longitude']
            except ValueError:
                continue
            ts = ts.astype(float)

            # currently will find non-unique original FID file in case original FID is non-unique integer
            rs_file = os.path.join(landsat_dir, '{}.csv'.format(fid))
            if not os.path.exists(rs_file):
                missing['rs_file'] += 1
                continue

            rs = pd.read_csv(rs_file, index_col='Unnamed: 0', parse_dates=True)
            if hourly:
                rs = rs.resample('h').ffill()
            idx = [i for i in rs.index if i in ts.index]
            if not idx:
                missing['rs_obs_time_misalign'] += 1
                continue
            ts.loc[idx, rs.columns] = rs.loc[idx, rs.columns]

            if fid not in sol_df.columns.to_list():
                missing['sol_fid'] += 1
                continue

            sol = sol_df[fid].to_dict()
            ts['doy'] = ts.index.dayofyear
            ts['rsun'] = ts['doy'].map(sol) * 0.0036

            ts = ts.loc[~ts.index.duplicated(keep='first')]

            try:
                nan_shape = ts.shape[0]
                ts.dropna(how='any', axis=0, inplace=True)
                no_nan_shape = ts.shape[0]
                # if nan_shape != no_nan_shape:
                #     print('dropped {} of {}, nan'.format(nan_shape - no_nan_shape, nan_shape))
            except TypeError:
                pass

            if scaling_json:
                for c in ts.columns:

                    for mm in ['max', 'min']:

                        p = '{}_{}'.format(c, mm)
                        v = ts[c].agg(mm, axis=0)

                        if p not in scaling.keys():
                            scaling[p] = v.item()

                        if mm == 'max' and v > scaling[p]:
                            scaling[p] = v.item()

                        if mm == 'min' and v < scaling[p]:
                            scaling[p] = v.item()

            if ts.empty:
                # print('{} is empty, skipped it'.format(f))
                continue

            # mark consecutive days
            try:
                if hourly:
                    deltas = ts.index.to_series().diff().dt.seconds.fillna(3600)
                    ts.loc[deltas != 3600, 'dt_diff'] = 2
                    ts.loc[deltas == 3600, 'dt_diff'] = 1
                else:
                    ts['dt_diff'] = ts.index.to_series().diff().dt.days.fillna(1)
            except ValueError:
                continue

            if first:
                shape = ts.shape[1]
                first = False
            else:
                if ts.shape[1] != shape:
                    print('{} has {} cols, should have {}, skipped it'.format(f, ts.shape[1], shape))
                    continue

            # write csv without dt index
            ts.to_csv(outfile, index=False)
            ct += ts.shape[0]
            scaling['stations'].append(f)

    if scaling_json:
        with open(scaling_json, 'w') as fp:
            json.dump(scaling, fp, indent=4)

    print('\ntime series obs count {}\n{} stations'.format(ct, len(scaling['stations'])))
    print('wrote {} features'.format(ts.shape[1] - 3))
    print('missing', missing)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    target_var = 'vpd'
    glob_ = 'dads_stations_elev_mgrs'

    fields = os.path.join(d, 'met', 'stations', '{}.csv'.format(glob_))
    landsat_ = os.path.join(d, 'rs', 'dads_stations', 'landsat', 'station_data')
    solrad = os.path.join(d, 'dem', 'rsun_tables')

    zoran = '/home/dgketchum/training'
    nvm = '/media/nvm/training'
    if os.path.exists(zoran):
        print('writing to zoran')
        training = zoran
    elif os.path.exists(nvm):
        print('writing to nvm drive')
        training = nvm
    else:
        print('writing to UM drive')
        training = os.path.join(d, 'training')

    param_dir = os.path.join(training, target_var)
    out_csv = os.path.join(param_dir, 'compiled_csv')

    hourly = True
    overwrite_ = False
    write_scaling = True
    remove_existing = False

    if hourly:
        sta = os.path.join(d, 'met', 'joined', 'hourly')
    else:
        sta = os.path.join(d, 'met', 'joined', 'daily')

    if overwrite_ or remove_existing:
        l = [os.path.join(out_csv, f) for f in os.listdir(out_csv)]
        [os.remove(f) for f in l]
        print('removed existing data in {}'.format(out_csv))

    if write_scaling:
        scaling_ = os.path.join(param_dir, 'scaling_metadata.json')
    else:
        scaling_ = None

    if not os.path.exists(training):
        os.mkdir(training)

    if not os.path.exists(param_dir):
        os.mkdir(param_dir)

    if not os.path.exists(out_csv):
        os.mkdir(out_csv)

    print('========================== writing {} =========================='.format(target_var))

    # W. MT: (-117., 42.5, -110., 49.)
    join_training(fields, sta, landsat_, solrad, out_csv, scaling_json=scaling_, var=target_var,
                  bounds=(-125., 25., -96., 49.), shuffle=True, overwrite=overwrite_, hourly=True, sample_frac=0.2)

# ========================= EOF ==============================================================================
