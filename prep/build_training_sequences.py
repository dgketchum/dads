import json
import multiprocessing
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.station_parameters import station_par_map
from prep.columns_desc import PTH_COLUMNS, TARGETS, COMPARISON_FEATURES, TERRAIN_FEATURES


def process_station(fid, row, ts_dir, landsat_dir, cdr_dir, dem_dir, terrain_dir, out_dir, overwrite, chunk_size=72):
    """"""

    missing = file_check(fid, ts_dir, landsat_dir, cdr_dir, dem_dir, terrain_dir)

    if any(v > 0 for k, v in missing.items() if k not in ['exists', 'columns']):
        return fid, None, missing

    # Observed Meteorology ==============================================================================
    sta_file = os.path.join(ts_dir, '{}.parquet'.format(fid))
    ts = pd.read_parquet(sta_file)

    try:
        ts.loc[:, 'lat'], ts.loc[:, 'lon'] = row['latitude'], row['longitude']
    except ValueError:
        return fid, None, missing

    # Landsat Surface Reflectance/Radiance ==============================================================
    landsat_file = os.path.join(landsat_dir, '{}.csv'.format(fid))
    landsat = pd.read_csv(landsat_file, index_col='Unnamed: 0', parse_dates=True)
    landsat = landsat.resample('h').ffill()
    idx = [i for i in landsat.index if i in ts.index]
    if not idx:
        missing['landsat_obs_time_misalign'] += 1
        return fid, None, missing
    ts.loc[idx, landsat.columns] = landsat.loc[idx, landsat.columns]

    # NOAA Climate Data Record ==========================================================================
    cdr_file = os.path.join(cdr_dir, '{}.csv'.format(fid))
    cdr = pd.read_csv(cdr_file, index_col='Unnamed: 0', parse_dates=True)
    cdr = cdr.resample('h').ffill()
    idx = [i for i in cdr.index if i in ts.index]
    if not idx:
        missing['cdr_obs_time_misalign'] += 1
        return fid, None, missing
    ts.loc[idx, cdr.columns] = cdr.loc[idx, cdr.columns]

    # Clear-sky Solar Irradiance ========================================================================
    sol_file = os.path.join(dem_dir, f'{fid}.csv')
    try:
        sol_df = pd.read_csv(sol_file, index_col=0)
        sol = sol_df[fid].to_dict()
        ts['doy'] = ts.index.dayofyear
        ts['rsun'] = ts['doy'].map(sol) * 0.0036
    except (KeyError, pd.errors.EmptyDataError):
        missing['sol_fid'] += 1
        return fid, None, missing

    # Terrain Information ===============================================================================
    terrain_file = os.path.join(terrain_dir, f'{fid}.csv')
    try:
        terrain_df = pd.read_csv(terrain_file, index_col=0).T.iloc[0].to_dict()
        for t_param in TERRAIN_FEATURES:
            ts[t_param] = terrain_df[t_param]
    except (KeyError, pd.errors.EmptyDataError):
        missing['terrain_fid'] += 1
        return fid, None, missing

    ts['dt_obs'] = [dt.strftime('%Y%m%d') for dt in ts.index]

    ts['time_diff'] = ts.index.to_series().diff().dt.total_seconds() / 3600

    doy = torch.tensor(ts.index.dayofyear.values, dtype=torch.float32)
    ts['doy_sin'] = torch.sin(2 * torch.pi * doy / 365.25)
    ts['doy_cos'] = torch.cos(2 * torch.pi * doy / 365.25)

    hour = torch.tensor(ts.index.hour.values, dtype=torch.float32)
    ts['hour_sin'] = torch.sin(2 * torch.pi * hour / 24)
    ts['hour_cos'] = torch.cos(2 * torch.pi * hour / 24)

    ts = ts.loc[~ts.index.duplicated(keep='first')]

    add_missing = [c for c in TARGETS if c not in ts.columns]
    for missed in add_missing:
        ts[missed] = np.nan

    try:
        ts.dropna(how='all', subset=TARGETS, axis=0, inplace=True)
    except TypeError:
        pass

    if ts.empty:
        return fid, None, missing

    try:
        ts = ts[PTH_COLUMNS].astype(float)
    except KeyError:
        missing['columns'] += 1
        return fid, None, missing

    out_parquet = os.path.join(out_dir, 'parquet', '{}.parquet'.format(fid))
    if os.path.exists(out_parquet) and not overwrite:
        missing['exists'] += 1
        return fid, None, missing

    if not os.path.exists(out_parquet) or overwrite:
        daily_df = ts.resample('D').mean()
        daily_df.dropna(how='all', subset=TARGETS, axis=0, inplace=True)
        if not daily_df.empty:
            daily_df.to_parquet(out_parquet)

    chunk_ct = 0
    num_chunks = len(ts) // chunk_size
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = ts.iloc[start:end]

        chunk = chunk[chunk['time_diff'] == 1]
        chunk.drop(columns=['time_diff'], inplace=True)

        if len(chunk) == chunk_size:
            outfile = os.path.join(out_dir, 'pth', f'{fid}_{i}.pth')
            if os.path.exists(outfile) and not overwrite:
                missing['exists'] += 1
            else:
                try:
                    torch.save(torch.tensor(chunk.values, dtype=torch.float32), outfile)
                except Exception as exc:
                    print(exc, fid)
                    continue

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


def file_check(fid, ts_dir, landsat_dir, cdr_dir, sol_dir, terrain_dir):
    """"""
    missing = {'sol_file': 0,
               'terrain_file': 0,
               'terrain_fid': 0,
               'station_file': 0,
               'landsat_file': 0,
               'snotel': 0,
               'landsat_obs_time_misalign': 0,
               'cdr_obs_time_misalign': 0,
               'sol_fid': 0,
               'cdr_file': 0,
               'exists': 0,
               'columns': 0}

    sta_file = os.path.join(ts_dir, '{}.parquet'.format(fid))
    if not os.path.exists(sta_file):
        missing['station_file'] += 1
        return missing

    landsat_file = os.path.join(landsat_dir, '{}.csv'.format(fid))
    if not os.path.exists(landsat_file):
        missing['landsat_file'] += 1
        return missing

    cdr_file = os.path.join(cdr_dir, '{}.csv'.format(fid))
    if not os.path.exists(cdr_file):
        missing['cdr_file'] += 1
        return missing

    sol_file = os.path.join(sol_dir, f'{fid}.csv')
    if not os.path.exists(sol_file):
        missing['sol_file'] += 1
        return missing

    terrain_file = os.path.join(terrain_dir, f'{fid}.csv')
    if not os.path.exists(terrain_file):
        missing['terrain_file'] += 1
        return missing

    return missing


def process_station_wrapper(args):
    return process_station(*args)


def join_training(stations, ts_dir, landsat_dir, cdr, sol_dir, terrain_dir, out_dir, bounds=None, debug=False,
                  shuffle=False, overwrite=False, sample_frac=1.0, workers=4, chunk_size=72, source='madis'):
    """"""
    kw = station_par_map(source)

    stations = pd.read_csv(stations, index_col=kw['index'])
    stations.sort_index(inplace=True)

    if source == 'ghcn':
        stations['orig_netid'] = stations.index
        stations['source'] = source

    if bounds:
        w, s, e, n = bounds
        stations = stations[(stations[kw['lat']] < n) & (stations[kw['lat']] >= s)]
        stations = stations[(stations[kw['lon']] < e) & (stations[kw['lon']] >= w)]

    if shuffle:
        stations = stations.sample(frac=sample_frac)

    rows = [{'index': f, 'latitude': float(row[kw['lat']]), 'longitude': float(row[kw['lon']]),
             'orig_netid': str(row['orig_netid']), 'source': str(row['source'])} for f, row in stations.iterrows()]

    fids = [str(f) for f in stations.index.to_list()]

    args = [(f, row, ts_dir, landsat_dir, cdr, sol_dir, terrain_dir, out_dir, overwrite, chunk_size)
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
                     'terrain_file': 0,
                     'terrain_fid': 0,
                     'landsat_file': 0,
                     'snotel': 0,
                     'cdr_obs_time_misalign': 0,
                     'landsat_obs_time_misalign': 0,
                     'sol_fid': 0,
                     'cdr_file': 0,
                     'exists': 0}

    # TODO: remove stats gathering as we now sample the training data for scaling parameters
    for f, stat, missing in results:
        if stat is not None:
            for col, values in stat.items():
                if f not in combined_stats.keys():
                    combined_stats[f] = {}
                combined_stats[f].update({col: values})

        for k, v in missing.items():
            total_missing[k] += v

    print('missing', total_missing)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    # glob_ = 'dads_stations_10FEB2025'
    # _source = 'madis'

    glob_ = 'ghcn_CANUSA_stations_mgrs'
    _source = 'ghcn'

    fields = os.path.join(d, 'met', 'stations', '{}.csv'.format(glob_))
    landsat_ = os.path.join(d, 'rs', 'landsat', 'station_data')
    cdr_ = os.path.join(d, 'rs', 'cdr', 'joined')
    solrad = os.path.join(d, 'dem', 'rsun_tables', 'station_rsun')
    terrain = os.path.join(d, 'dem', 'terrain', 'station_data')

    zoran = '/data/ssd2/dads/training'
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

    overwrite_ = False

    sta = os.path.join(d, 'met', 'joined', 'hourly')

    with open(os.path.join(training, 'pth_columns.json'), 'w') as fp:
        json.dump({'columns': PTH_COLUMNS}, fp, indent=4)

    print('========================== writing joined training data ==========================')

    join_training(fields, sta, landsat_, cdr_, solrad, terrain, training, bounds=None, debug=False, shuffle=True,
                  overwrite=overwrite_, sample_frac=1.0, workers=8, chunk_size=72, source=_source)

# ========================= EOF ==============================================================================
