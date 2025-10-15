import os
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

OBS_TARGETS = ['rsds', 'tmax', 'tmin', 'ea', 'prcp', 'wind']


def _assess_madis_file(path):
    stn = os.path.splitext(os.path.basename(path))[0]
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None

    if df.empty:
        return {'station': stn}

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors='coerce')
        except Exception:
            pass

    row = {'station': stn}
    for var in OBS_TARGETS:
        if var in df.columns:
            mask = df[var].notna()
            cnt = int(mask.sum())
            row[var] = cnt
            if cnt > 0:
                first_idx = df.index[mask][0]
                last_idx = df.index[mask][-1]
                s = None if pd.isna(first_idx) else pd.to_datetime(first_idx).strftime('%Y-%m-%d')
                e = None if pd.isna(last_idx) else pd.to_datetime(last_idx).strftime('%Y-%m-%d')
                row[f'start_{var}'] = s if s is not None else 'NULL'
                row[f'end_{var}'] = e if e is not None else 'NULL'
            else:
                row[f'start_{var}'] = 'NULL'
                row[f'end_{var}'] = 'NULL'
        else:
            row[var] = 0
            row[f'start_{var}'] = 'NULL'
            row[f'end_{var}'] = 'NULL'
    # Include coords if present for optional downstream mapping
    for c in ['latitude', 'longitude', 'elevation']:
        if c in df.columns:
            try:
                row[c] = float(df[c].iloc[0])
            except Exception:
                pass
    return row


def assess_downloaded_madis(records_dir, out_csv, joined_dir, training_dir,
                            landsat_dir, cdr_dir, solrad_dir, terrain_dir,
                            out_shp=None, num_workers=12, debug_limit=None):
    # Required directories must exist
    req = [records_dir, joined_dir, training_dir, landsat_dir, cdr_dir, solrad_dir, terrain_dir]
    for r in req:
        if not os.path.exists(r):
            raise FileNotFoundError(r)

    files = [os.path.join(records_dir, f) for f in os.listdir(records_dir) if f.endswith('.parquet')]
    if debug_limit:
        files = files[:int(debug_limit)]

    summaries = []
    if num_workers is None or num_workers <= 1:
        for p in tqdm(files, total=len(files), desc='Assessing MADIS'):
            row = _assess_madis_file(p)
            if row:
                stn = row['station']
                row['in_joined'] = int(os.path.exists(os.path.join(joined_dir, f'{stn}.parquet')))
                # training_dir contains subfolders per target; consider present if any exists
                in_training = 0
                try:
                    subs = [s for s in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, s))]
                    for s in subs:
                        pth = os.path.join(training_dir, s, f'{stn}.parquet')
                        if os.path.exists(pth):
                            in_training = 1
                            break
                except Exception:
                    in_training = 0
                row['in_training'] = in_training

                row['has_landsat'] = int(os.path.exists(os.path.join(landsat_dir, f'{stn}.csv')))
                row['has_cdr'] = int(os.path.exists(os.path.join(cdr_dir, f'{stn}.csv')))
                row['has_rsun'] = int(os.path.exists(os.path.join(solrad_dir, f'{stn}.csv')))
                row['has_terrain'] = int(os.path.exists(os.path.join(terrain_dir, f'{stn}.csv')))
                summaries.append(row)
    else:
        with ProcessPoolExecutor(max_workers=int(num_workers)) as ex:
            for row in tqdm(ex.map(_assess_madis_file, files), total=len(files), desc='Assessing MADIS'):
                if row:
                    stn = row['station']
                    row['in_joined'] = int(os.path.exists(os.path.join(joined_dir, f'{stn}.parquet')))
                    in_training = 0
                    try:
                        subs = [s for s in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, s))]
                        for s in subs:
                            pth = os.path.join(training_dir, s, f'{stn}.parquet')
                            if os.path.exists(pth):
                                in_training = 1
                                break
                    except Exception:
                        in_training = 0
                    row['in_training'] = in_training

                    row['has_landsat'] = int(os.path.exists(os.path.join(landsat_dir, f'{stn}.csv')))
                    row['has_cdr'] = int(os.path.exists(os.path.join(cdr_dir, f'{stn}.csv')))
                    row['has_rsun'] = int(os.path.exists(os.path.join(solrad_dir, f'{stn}.csv')))
                    row['has_terrain'] = int(os.path.exists(os.path.join(terrain_dir, f'{stn}.csv')))
                    summaries.append(row)

    if len(summaries) == 0:
        return pd.DataFrame(columns=['station'])

    df = pd.DataFrame(summaries).sort_values('station').reset_index(drop=True)

    # derive overall start/end across available observed vars
    start_cols = [f'start_{v}' for v in OBS_TARGETS if f'start_{v}' in df.columns]
    end_cols = [f'end_{v}' for v in OBS_TARGETS if f'end_{v}' in df.columns]
    if start_cols:
        s_list = []
        for c in start_cols:
            s = pd.to_datetime(df[c].replace('NULL', pd.NaT), errors='coerce')
            s_list.append(s)
        svals = pd.concat(s_list, axis=1)
        smin = svals.min(axis=1)
        df['start_date'] = smin.dt.strftime('%Y-%m-%d')
        df.loc[smin.isna(), 'start_date'] = 'NULL'
    if end_cols:
        e_list = []
        for c in end_cols:
            e = pd.to_datetime(df[c].replace('NULL', pd.NaT), errors='coerce')
            e_list.append(e)
        evals = pd.concat(e_list, axis=1)
        emax = evals.max(axis=1)
        df['end_date'] = emax.dt.strftime('%Y-%m-%d')
        df.loc[emax.isna(), 'end_date'] = 'NULL'

    try:
        df.to_csv(out_csv, index=False)
    except Exception as e:
        print(f'Failed writing {out_csv}: {e}')

    in_joined = df.get('in_joined', pd.Series(0, index=df.index)).fillna(0).astype(int)
    in_training = df.get('in_training', pd.Series(0, index=df.index)).fillna(0).astype(int)

    # analogous pre-1987 summary using overall end_date if available
    end_dt = pd.to_datetime(df.get('end_date', 'NULL'), errors='coerce')
    pre01_mask = end_dt.notna() & (end_dt < pd.Timestamp('2001-01-01'))
    pre01_ct = int(pre01_mask.sum())
    remain = ~pre01_mask
    join_fail_mask = remain & (in_joined == 0)
    join_fail_ct = int(join_fail_mask.sum())

    train_fail_mask = remain & (in_joined == 1) & (in_training == 0)

    def mcount(col):
        s = df.get(col, pd.Series(0, index=df.index)).fillna(0).astype(int)
        return int((train_fail_mask & (s == 0)).sum())

    miss_rsun = mcount('has_rsun')
    miss_landsat = mcount('has_landsat')
    miss_cdr = mcount('has_cdr')
    miss_terrain = mcount('has_terrain')

    print('Not-in-training summary:')
    print(f'- Ended before 2001-01-01: {pre01_ct}')
    print(f'- Missing joined companion series: {join_fail_ct}')
    print('- Missing ancillary among training-step failures:')
    print(f'  rsun={miss_rsun}, landsat={miss_landsat}, cdr={miss_cdr}, terrain={miss_terrain}')

    # write shapefile if requested and coords present
    if out_shp:
        try:
            if 'latitude' in df.columns and 'longitude' in df.columns:
                m = df.dropna(subset=['latitude', 'longitude']).copy()
                if not m.empty:
                    gdf = gpd.GeoDataFrame(m,
                                           geometry=gpd.points_from_xy(m['longitude'], m['latitude']),
                                           crs='EPSG:4326')
                    gdf.to_file(out_shp, crs='EPSG:4326', engine='fiona')
        except Exception as e:
            print(f'Failed writing shapefile {out_shp}: {e}')

    return df


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    madis_daily = '/data/ssd2/madis/daily'

    out_csv_ = os.path.join('/data/ssd2/madis', 'madis_station_counts.csv')
    out_shp_ = os.path.join('/data/ssd2/madis', 'madis_station_counts.shp')

    landsat_ = os.path.join(d, 'dads', 'rs', 'landsat', 'station_data')
    cdr_ = os.path.join(d, 'dads', 'rs', 'cdr', 'joined')
    solrad = os.path.join(d, 'dads', 'dem', 'rsun_stations')
    terrain = os.path.join(d, 'dads', 'dem', 'terrain', 'station_data')
    training = '/data/ssd2/dads/training/parquet'
    joined = '/data/ssd2/dads/met/joined'

    debug_limit_ = None

    assess_downloaded_madis(madis_daily,
                            out_csv_,
                            joined,
                            training,
                            landsat_,
                            cdr_,
                            solrad,
                            terrain,
                            out_shp=out_shp_,
                            num_workers=12,
                            debug_limit=debug_limit_)

# ========================= EOF ====================================================================
