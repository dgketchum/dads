import os
import json

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing

MET_FEATURES = [
    'CAPE_nl_hr',
    'CRainf_frac_nl_hr',
    'LWdown_nl_hr',
    'PSurf_nl_hr',
    'PotEvap_nl_hr',
    'Qair_nl_hr',
    'Rainf_nl_hr',
    'SWdown_nl_hr',
    'Tair_nl_hr',
    'Wind_E_nl_hr',
    'Wind_N_nl_hr',
    'doy_obs',
    'dt_nl_hr',
    'lat_nl_hr',
    'lon_nl_hr',
]

GEO_FEATURES = ['lat', 'lon', 'B10', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'doy', 'rsun']

COMPARISON_FEATURES = ['mean_temp_dm', 'rsds_dm', 'vpd_dm']

TARGETS = ['mean_temp_obs', 'rn_obs', 'rsds_obs', 'u2_obs', 'vpd_obs']

ADDED_FEATURES = [
    'doy_sin',
    'doy_cos',
    'hour_sin',
    'hour_cos',
]

PTH_COLUMNS = TARGETS + COMPARISON_FEATURES + MET_FEATURES + GEO_FEATURES + ADDED_FEATURES


def process_station(fid, row, ts_dir, landsat_dir, dem_dir, out_dir, overwrite, chunk_size=72):
    """"""
    missing = {'sol_file': 0,
               'station_file': 0,
               'landsat_file': 0,
               'snotel': 0,
               'landsat_obs_time_misalign': 0,
               'sol_fid': 0,
               'cdr_file': 0,
               'exists': 0,
               'columns': 0}

    outfile = os.path.join(out_dir, '{}.parquet'.format(fid))
    if os.path.exists(outfile) and not overwrite:
        missing['exists'] += 1
        return fid, None, missing

    if row['source'] == 'snotel':
        missing['snotel'] += 1
        return fid, None, missing

    sta_file = os.path.join(ts_dir, '{}.parquet'.format(fid))
    if not os.path.exists(sta_file):
        missing['station_file'] += 1
        return fid, None, missing

    #  ========= Observed and Gridded Meteorology Record =========
    ts = pd.read_parquet(sta_file)

    try:
        ts.loc[:, 'lat'], ts.loc[:, 'lon'] = row['latitude'], row['longitude']
    except ValueError:
        return fid, None, missing

    # ========= Landsat Record =============
    landsat_file = os.path.join(landsat_dir, '{}.csv'.format(fid))
    if not os.path.exists(landsat_file):
        missing['landsat_file'] += 1
        return fid, None, missing

    landsat = pd.read_csv(landsat_file, index_col='Unnamed: 0', parse_dates=True)
    landsat = landsat.resample('h').ffill()
    idx = [i for i in landsat.index if i in ts.index]
    if not idx:
        missing['landsat_obs_time_misalign'] += 1
        return fid, None, missing
    ts.loc[idx, landsat.columns] = landsat.loc[idx, landsat.columns]

    # =========== Terrain-Adjusted Clear Sky Solar Radiation ================
    sol_file = os.path.join(dem_dir, f'{fid}.csv')
    try:
        sol_df = pd.read_csv(sol_file, index_col=0)
        sol = sol_df[fid].to_dict()
        ts['doy'] = ts.index.dayofyear
        ts['rsun'] = ts['doy'].map(sol) * 0.0036
    except FileNotFoundError:
        missing['sol_file'] += 1
        return fid, None, missing
    except KeyError:
        missing['sol_fid'] += 1
        return fid, None, missing

    ts['time_diff'] = ts.index.to_series().diff().dt.total_seconds() / 3600

    doy = torch.tensor(ts.index.dayofyear.values, dtype=torch.float32)
    ts['doy_sin'] = torch.sin(2 * torch.pi * doy / 365.25)
    ts['doy_cos'] = torch.cos(2 * torch.pi * doy / 365.25)

    hour = torch.tensor(ts.index.hour.values, dtype=torch.float32)
    ts['hour_sin'] = torch.sin(2 * torch.pi * hour / 24)
    ts['hour_cos'] = torch.cos(2 * torch.pi * hour / 24)

    ts = ts.loc[~ts.index.duplicated(keep='first')]
    try:
        ts.dropna(how='all', subset=TARGETS, axis=0, inplace=True)
    except TypeError:
        pass

    if ts.empty:
        return fid, None, missing

    try:
        ts = ts[PTH_COLUMNS + ['time_diff']].astype(float)
    except KeyError:
        missing['columns'] += 1
        return fid, None, missing

    chunk_ct = 0
    num_chunks = len(ts) // chunk_size
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = ts.iloc[start:end]

        chunk = chunk[chunk['time_diff'] == 1]
        chunk.drop(columns=['time_diff'], inplace=True)

        if len(chunk) == chunk_size:
            outfile = os.path.join(out_dir, f'{fid}_{i}.pth')
            if os.path.exists(outfile) and not overwrite:
                missing['exists'] += 1
            else:
                torch.save(torch.tensor(chunk.values, dtype=torch.float32), outfile)
                chunk_ct += 1

    target_cols = [col for col in ts.columns if col in TARGETS + COMPARISON_FEATURES]
    stats = {}
    for col in target_cols:
        valid_entries = ts[col].notna().sum()
        if valid_entries == 0:
            stats[col] = {'valid_entries': int(valid_entries / 24),
                          'chunks': chunk_ct,
                          'min': np.nan,
                          'max': np.nan}
        else:
            stats[col] = {'valid_entries': int(valid_entries / 24),
                          'chunks': chunk_ct,
                          'min': np.nanmin(ts[col].values),
                          'max': np.nanmax(ts[col].values)}

    return fid, stats, missing


def process_station_wrapper(args):
    return process_station(*args)

def join_training(stations, ts_dir, landsat_dir, dem_dir, out_dir, stats_dir, bounds=None, debug=False,
                  shuffle=False, overwrite=False, sample_frac=1.0, workers=4, chunk_size=72):
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

    rows = [{'index': f, 'latitude': float(row['latitude']), 'longitude': float(row['longitude']),
             'orig_netid': str(row['orig_netid']), 'source': str(row['source'])} for f, row in stations.iterrows()]

    fids = [str(f) for f in stations.index.to_list()]

    args = [(f, row, ts_dir, landsat_dir, dem_dir, out_dir, overwrite, chunk_size)
            for f, row in zip(fids, rows)]

    if debug:
        results = []
        for arg_tuple in args:
            f, stat, missing = process_station(*arg_tuple)
            results.append((f, stat, missing))

    else:
        with multiprocessing.Pool(processes=workers) as pool:
            results = list(tqdm(pool.imap(process_station_wrapper, args), total=len(fids)))

    combined_stats = {}
    total_missing = {'sol_file': 0,
                     'station_file': 0,
                     'landsat_file': 0,
                     'snotel': 0,
                     'landsat_obs_time_misalign': 0,
                     'sol_fid': 0,
                     'cdr_file': 0,
                     'exists': 0}

    for f, stat, missing in results:
        if stat is not None:
            for col, values in stat.items():
                if f not in combined_stats.keys():
                    combined_stats[f] = {}
                combined_stats[f].update({col: values})

        for k, v in missing.items():
            total_missing[k] += v

    print('missing', total_missing)

    # Write stats to JSON
    json_file = os.path.join(stats_dir, 'combined_stats.json')
    with open(json_file, 'w') as f:
        json.dump(combined_stats, f, indent=4)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    glob_ = 'dads_stations_10FEB2025'

    fields = os.path.join(d, 'met', 'stations', '{}.csv'.format(glob_))
    landsat_ = os.path.join(d, 'rs', 'landsat', 'station_data')
    solrad = os.path.join(d, 'dem', 'rsun_tables', 'station_rsun')

    zoran = '/data/ssd2/dads/training/simple_lstm'
    nvm = '/media/nvm/training/simple_lstm'
    if os.path.exists(zoran):
        print('writing to zoran')
        training = zoran
    elif os.path.exists(nvm):
        print('writing to nvm drive')
        training = nvm
    else:
        print('writing to UM drive')
        training = os.path.join(d, 'training')

    out_csv = os.path.join(training, 'pth')

    overwrite_ = False

    sta = os.path.join(d, 'met', 'joined', 'hourly')

    with open(os.path.join(training, 'pth_columns.json'), 'w') as fp:
        json.dump({'columns': PTH_COLUMNS}, fp, indent=4)

    print('========================== writing joined training data ==========================')

    join_training(fields, sta, landsat_, solrad, out_csv, stats_dir=training, bounds=None, debug=True,
                  shuffle=True, overwrite=overwrite_, sample_frac=1.0, chunk_size=72, workers=12)

# ========================= EOF ==============================================================================
