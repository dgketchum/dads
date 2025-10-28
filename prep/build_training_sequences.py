import os
from datetime import datetime
import multiprocessing


import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from prep.qaqc_schema import remap_known_sentinels
from models.dads.value_limits import FEATURE_LIMITS, TARGET_LIMITS

from prep.columns_desc import TARGETS, TERRAIN_FEATURES, GEO_FEATURES
from prep.columns_desc import LANDSAT_FEATURES, CDR_FEATURES, RS_MISS_FEATURES
from utils.station_parameters import station_par_map


def process_station(fid, row, ts_dir, landsat_dir, cdr_dir, dem_dir, terrain_dir, out_dir, overwrite,
                    require_landsat=True, require_cdr=True):
    """"""

    missing = file_check(fid, ts_dir, landsat_dir, cdr_dir, dem_dir, terrain_dir)

    if any(v > 0 for k, v in missing.items() if k not in ['exists', 'columns']):
        if require_cdr or require_landsat and any(missing[k] for k in ['landsat_file',
                                                                       'landsat_obs_time_misalign',
                                                                       'cdr_obs_time_misalign',
                                                                       'cdr_file']):
            pass

        else:
            return fid, None, missing

    # Observed Meteorology ==============================================================================
    sta_file = os.path.join(ts_dir, '{}.parquet'.format(fid))
    ts = pd.read_parquet(sta_file)
    # Ensure DatetimeIndex with no NaT before downstream date ops
    if not isinstance(ts.index, pd.DatetimeIndex):
        ts.index = pd.to_datetime(ts.index, errors='coerce')
    try:
        ts.index = ts.index.tz_localize(None)
    except (TypeError, AttributeError):
        # Index already naive or non-timezone; proceed
        pass

    # Drop NaT index rows preemptively to avoid strftime errors
    ts = ts[ts.index.notna()]
    ts.sort_index(inplace=True)
    # ensure unique datetime index before adding tensor/object columns
    ts = ts.loc[~ts.index.duplicated(keep='first')]

    # ignore added missingness flag suffix when selecting comparator suffix
    non_obs_cols = np.array([c.split('_')[-1] for c in ts.columns if 'obs' not in c and not c.endswith('_miss')])
    cols_cts = np.unique(non_obs_cols, return_counts=True)
    comparison_glob = cols_cts[0][np.argmax(cols_cts[1])]

    try:
        ts.loc[:, 'lat'], ts.loc[:, 'lon'] = row['latitude'], row['longitude']
    except ValueError:
        return fid, None, missing

    # Landsat Surface Reflectance/Radiance ==============================================================
    landsat_file = os.path.join(landsat_dir, '{}.csv'.format(fid))
    try:
        landsat = pd.read_csv(landsat_file, index_col='Unnamed: 0', parse_dates=True)
        idx = [i for i in landsat.index if i in ts.index]
        if not idx:
            missing['landsat_obs_time_misalign'] += 1
            if require_landsat:
                return fid, None, missing

        ts.loc[idx, landsat.columns] = landsat.loc[idx, landsat.columns]
    except (pd.errors.EmptyDataError, FileNotFoundError):
        if require_landsat:
            return fid, None, missing

    # ensure RS columns exist in unified schema even if not provided
    for b in LANDSAT_FEATURES:
        if b not in ts.columns:
            ts[b] = np.nan

    # NOAA Climate Data Record ==========================================================================
    cdr_file = os.path.join(cdr_dir, '{}.csv'.format(fid))
    try:
        cdr = pd.read_csv(cdr_file, index_col='Unnamed: 0', parse_dates=True)
        idx = [i for i in cdr.index if i in ts.index]
        if not idx:
            missing['cdr_obs_time_misalign'] += 1
            return fid, None, missing
        ts.loc[idx, cdr.columns] = cdr.loc[idx, cdr.columns]
    except (pd.errors.EmptyDataError, FileNotFoundError):
        if require_cdr:
            return fid, None, missing
    # ensure RS columns exist in unified schema even if not provided
    for b in CDR_FEATURES:
        if b not in ts.columns:
            ts[b] = np.nan

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

    # Vectorized, safe dt string
    ts['dt_obs'] = ts.index.strftime('%Y%m%d')

    ts['time_diff'] = ts.index.to_series().diff().dt.total_seconds() / 3600

    doy = torch.tensor(ts.index.dayofyear.values, dtype=torch.float32)
    ts['doy_sin'] = torch.sin(2 * torch.pi * doy / 365.25)
    ts['doy_cos'] = torch.cos(2 * torch.pi * doy / 365.25)


    add_missing = [c for c in TARGETS if c not in ts.columns]
    for missed in add_missing:
        ts[missed] = np.nan

    try:
        ts.dropna(how='all', subset=TARGETS, axis=0, inplace=True)
    except TypeError:
        pass

    if ts.empty:
        return fid, None, missing

    obs_targets = ['tmax_obs', 'tmin_obs', 'rsds_obs', 'ea_obs', 'wind_obs', 'prcp_obs']
    gridded_suffix = comparison_glob

    # ensure RS feature columns are numeric before missingness flags
    for b in LANDSAT_FEATURES:
        if b in ts.columns:
            ts[b] = pd.to_numeric(ts[b], errors='coerce')
    for b in CDR_FEATURES:
        if b in ts.columns:
            ts[b] = pd.to_numeric(ts[b], errors='coerce')

    # QA: remap coded sentinels to NaN for NOAA CDR and available targets
    remap_known_sentinels(ts, [b for b in CDR_FEATURES if b in ts.columns], include_huge=True)
    remap_known_sentinels(ts, [c for c in ['tmax_obs', 'tmin_obs', 'rsds_obs', 'ea_obs', 'wind_obs', 'prcp_obs'] if c in ts.columns], include_huge=True)

    # Apply feature-level clamps to NaN out-of-range values
    for fname, lim in FEATURE_LIMITS.items():
        if fname in ts.columns and lim is not None:
            lo, hi = float(lim[0]), float(lim[1])
            v = pd.to_numeric(ts[fname], errors='coerce')
            ts.loc[(v < lo) | (v > hi), fname] = np.nan

    # add RS missingness flags once per station timeline
    for b in LANDSAT_FEATURES:
        miss_col = f'{b}_miss'
        ts[miss_col] = ts[b].isna().astype(np.uint8)
    for b in CDR_FEATURES:
        miss_col = f'{b}_miss'
        ts[miss_col] = ts[b].isna().astype(np.uint8)

    for obs_var in obs_targets:
        if obs_var in ts.columns and ts[obs_var].notna().any():
            base_var_name = obs_var.replace('_obs', '')
            gridded_var = f'{base_var_name}_{gridded_suffix}'

            # Apply target-specific clamps when available
            lim = TARGET_LIMITS.get(base_var_name)
            if lim is not None:
                lo, hi = float(lim[0]), float(lim[1])
                vv = pd.to_numeric(ts[obs_var], errors='coerce')
                ts.loc[(vv < lo) | (vv > hi), obs_var] = np.nan

            # add comparator column if absent, and a missingness column
            comp_miss_col = f'{gridded_var}_miss'
            if gridded_var not in ts.columns:
                ts[gridded_var] = np.nan  # ensure unified schema across files
                ts[comp_miss_col] = 1
            else:
                ts[comp_miss_col] = ts[gridded_var].isna().astype(np.uint8)

            cols_to_keep = [obs_var, gridded_var] + GEO_FEATURES + RS_MISS_FEATURES + [comp_miss_col]
            existing_cols = [col for col in cols_to_keep if col in ts.columns]

            var_df = ts[existing_cols].copy()
            var_df = var_df.dropna(subset=[obs_var]).astype(float)

            if not var_df.empty:

                var_out_dir = os.path.join(out_dir, obs_var)
                out_file = os.path.join(var_out_dir, f'{fid}.parquet')

                if not os.path.exists(out_file) or overwrite:
                    var_df.to_parquet(out_file)
                    now = datetime.now().strftime('%m%d %H%M')
                    print(f'{fid}: {var_df.shape[0]} written to {out_file} {now}', flush=True)
                else:
                    missing['exists'] += 1

    return fid, None, missing


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


def join_training(stations, ts_dir, landsat_dir, cdr, sol_dir, terrain_dir, out_dir,
                  bounds=None, debug=False, shuffle=False, overwrite=False,
                  workers=4, source='madis', require_landsat=True, require_cdr=True):
    """"""
    kw = station_par_map(source)

    stations = pd.read_csv(stations, index_col=kw['index'])
    stations.sort_index(inplace=True)

    stations['orig_netid'] = stations.index
    stations['source'] = source

    if bounds:
        w, s, e, n = bounds
        stations = stations[(stations[kw['lat']] < n) & (stations[kw['lat']] >= s)]
        stations = stations[(stations[kw['lon']] < e) & (stations[kw['lon']] >= w)]

    if shuffle:
        stations = stations.sample(frac=1.0)

    rows = [{'index': f, 'latitude': float(row[kw['lat']]), 'longitude': float(row[kw['lon']]),
             'orig_netid': str(row['orig_netid']), 'source': str(row['source'])} for f, row in stations.iterrows()]

    fids = [str(f) for f in stations.index.to_list()]

    args = [(f, row, ts_dir, landsat_dir, cdr, sol_dir, terrain_dir, out_dir, overwrite,
             require_landsat, require_cdr)
            for f, row in zip(fids, rows)]

    if debug or workers <= 1:
        results = []
        for arg_tuple in args:
            fid = arg_tuple[0]
            if fid != 'CMTI2':
                continue
            f, stat, missing = process_station(*arg_tuple)
            results.append((f, stat, missing))

    else:
        with multiprocessing.Pool(processes=workers) as pool:
            results = list(tqdm(pool.imap(process_station_wrapper, args), total=len(fids)))

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

    for f, _, missing in results:
        if missing:
            for k, v in missing.items():
                if k in total_missing:
                    total_missing[k] += v

    print('missing', total_missing)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    _source = 'ndbc'

    if _source == 'madis':
        glob_ = 'madis_02JULY2025_mgrs'
        fields = os.path.join(d, 'dads', 'met', 'stations', '{}.csv'.format(glob_))

    elif _source == 'ghcn':
        glob_ = 'ghcn_CANUSA_stations_mgrs'
        fields = os.path.join(d, 'climate', 'ghcn', 'stations', '{}.csv'.format(glob_))

    elif _source == 'ndbc':
        glob_ = 'ndbc_stations'
        fields = os.path.join(d, 'climate', 'ndbc', 'ndbc_meta', '{}.csv'.format(glob_))

    else:
        raise ValueError

    landsat_ = os.path.join(d, 'dads', 'rs', 'landsat', 'station_data')
    cdr_ = os.path.join(d, 'dads', 'rs', 'cdr', 'joined')
    solrad = os.path.join(d, 'dads', 'dem', 'rsun_stations')
    terrain = os.path.join(d, 'dads', 'dem', 'terrain', 'station_data')

    training = '/data/ssd2/dads/training/parquet'
    joined = '/data/ssd2/dads/met/joined'

    overwrite_ = True

    # For pre-remote-sensing records, disable remote sensing requirements.
    require_landsat_ = False
    require_cdr_ = False

    join_training(fields, joined, landsat_, cdr_, solrad, terrain,
                  out_dir=training,
                  source=_source,
                  overwrite=overwrite_,
                  workers=12,
                  debug=False,
                  require_landsat=require_landsat_,
                  require_cdr=require_cdr_,
                  )

# ========================= EOF ==============================================================================
