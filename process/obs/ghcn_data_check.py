import os
import pandas as pd
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from utils.station_parameters import station_par_map


def _assess_station_file(path):
    station = os.path.splitext(os.path.basename(path))[0]
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None

    try:
        dt_index = pd.to_datetime(df.index, errors='coerce')
    except Exception:
        dt_index = df.index
    df.index = dt_index

    row = {'station': station}
    wanted = ['PRCP', 'TMIN', 'TMAX']
    for var in wanted:
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
    return row


def assess_downloaded_data(records_dir, out_csv=None, inventory=None, out_shp=None, num_workers=12,
                           joined_dir=None, training_dir=None, landsat_dir=None, cdr_dir=None,
                           solrad_dir=None, terrain_dir=None, stations_csv=None):
    """
    Assess downloaded GHCN station CSV files.

    For each station file in `records_dir`, report the earliest date with any
    non-NaN observation and latest date, plus the count of non-NaN observations
    for each available meteorological variable.
    Optionally, check downstream presence (joined, training) and required
    ancillary datasets (rsun, landsat, cdr, terrain) per station by file existence.
    """
    summaries = []
    # Load optional stations CSV to reuse START/END dates
    start_end_map = {}
    if stations_csv and os.path.exists(stations_csv):
        try:
            kw = station_par_map('ghcn')
            st_df = pd.read_csv(stations_csv)
            idx_col = kw.get('index')
            if idx_col in st_df.columns:
                st_df.set_index(idx_col, inplace=True)
            start_col = kw.get('start', 'START DATE')
            end_col = kw.get('end', 'END DATE')
            if start_col in st_df.columns and end_col in st_df.columns:
                for sid, r in st_df[[start_col, end_col]].iterrows():
                    s = r[start_col]
                    e = r[end_col]
                    s_out = s if isinstance(s, str) and s else 'NULL'
                    e_out = e if isinstance(e, str) and e else 'NULL'
                    start_end_map[str(sid)] = (s_out, e_out)
        except Exception:
            start_end_map = {}

    if not os.path.isdir(records_dir):
        raise FileNotFoundError(f"records_dir not found: {records_dir}")

    files = [os.path.join(records_dir, f) for f in os.listdir(records_dir) if f.endswith('.csv')]

    if num_workers is None or num_workers <= 1:
        for path in tqdm(files, total=len(files), desc='Assessing stations'):
            row = _assess_station_file(path)
            if row:
                stn = row['station']
                se = start_end_map.get(stn, ('NULL', 'NULL'))
                row['start_date'] = se[0]
                row['end_date'] = se[1]
                if joined_dir:
                    row['in_joined'] = int(os.path.exists(os.path.join(joined_dir, f'{stn}.parquet')))
                if training_dir:
                    in_training = 0
                    try:
                        subs = [s for s in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, s))]
                        for s in subs:
                            p = os.path.join(training_dir, s, f'{stn}.parquet')
                            if os.path.exists(p):
                                in_training = 1
                                break
                    except Exception:
                        in_training = 0
                    row['in_training'] = in_training
                if landsat_dir:
                    row['has_landsat'] = int(os.path.exists(os.path.join(landsat_dir, f'{stn}.csv')))
                if cdr_dir:
                    row['has_cdr'] = int(os.path.exists(os.path.join(cdr_dir, f'{stn}.csv')))
                if solrad_dir:
                    row['has_rsun'] = int(os.path.exists(os.path.join(solrad_dir, f'{stn}.csv')))
                if terrain_dir:
                    row['has_terrain'] = int(os.path.exists(os.path.join(terrain_dir, f'{stn}.csv')))
                summaries.append(row)
    else:
        with ProcessPoolExecutor(max_workers=int(num_workers)) as ex:
            for row in tqdm(ex.map(_assess_station_file, files), total=len(files), desc='Assessing stations'):
                if row:
                    stn = row['station']
                    se = start_end_map.get(stn, ('NULL', 'NULL'))
                    row['start_date'] = se[0]
                    row['end_date'] = se[1]
                    if joined_dir:
                        row['in_joined'] = int(os.path.exists(os.path.join(joined_dir, f'{stn}.parquet')))
                    if training_dir:
                        in_training = 0
                        try:
                            subs = [s for s in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, s))]
                            for s in subs:
                                p = os.path.join(training_dir, s, f'{stn}.parquet')
                                if os.path.exists(p):
                                    in_training = 1
                                    break
                        except Exception:
                            in_training = 0
                        row['in_training'] = in_training
                    if landsat_dir:
                        row['has_landsat'] = int(os.path.exists(os.path.join(landsat_dir, f'{stn}.csv')))
                    if cdr_dir:
                        row['has_cdr'] = int(os.path.exists(os.path.join(cdr_dir, f'{stn}.csv')))
                    if solrad_dir:
                        row['has_rsun'] = int(os.path.exists(os.path.join(solrad_dir, f'{stn}.csv')))
                    if terrain_dir:
                        row['has_terrain'] = int(os.path.exists(os.path.join(terrain_dir, f'{stn}.csv')))
                    summaries.append(row)

    if len(summaries) == 0:
        return pd.DataFrame(columns=['station', 'start_date', 'end_date'])

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values(['station']).reset_index(drop=True)

    if out_csv:
        try:
            summary_df.to_csv(out_csv, index=False)
        except Exception as e:
            print(f"Failed writing summary CSV {out_csv}: {e}")

    # Summary counts for stations not in final training
    end_dt = pd.to_datetime(summary_df.get('end_date', 'NULL'), errors='coerce')
    pre87_mask = end_dt.notna() & (end_dt < pd.Timestamp('1987-01-01'))
    pre87_ct = int(pre87_mask.sum())

    in_joined = summary_df.get('in_joined', pd.Series(0, index=summary_df.index)).fillna(0).astype(int)
    in_training = summary_df.get('in_training', pd.Series(0, index=summary_df.index)).fillna(0).astype(int)

    remain = pd.Series(True, index=summary_df.index)  # include pre-1987 in diagnostics post pre-RS acceptance
    join_fail_mask = remain & (in_joined == 0)
    join_fail_ct = int(join_fail_mask.sum())

    train_fail_mask = remain & (in_joined == 1) & (in_training == 0)

    def mcount(col):
        s = summary_df.get(col, pd.Series(0, index=summary_df.index)).fillna(0).astype(int)
        return int((train_fail_mask & (s == 0)).sum())

    miss_rsun = mcount('has_rsun')
    miss_landsat = mcount('has_landsat')
    miss_cdr = mcount('has_cdr')
    miss_terrain = mcount('has_terrain')

    print('Not-in-training summary:')
    print(f'- Ended before 1987-01-01: {pre87_ct}')
    print(f'- Missing joined companion series: {join_fail_ct}')
    print('- Missing ancillary among training-step failures:')
    print(f'  rsun={miss_rsun}, landsat={miss_landsat}, cdr={miss_cdr}, terrain={miss_terrain}')

    # Totals by variable across all stations
    var_cols = [c for c in ['PRCP', 'TMIN', 'TMAX'] if c in summary_df.columns]
    if var_cols:
        print('Total observations by variable:')
        for c in var_cols:
            tot = pd.to_numeric(summary_df[c], errors='coerce').fillna(0).astype(int).sum()
            print(f'  {c}: {int(tot):,}')

    # Optionally write shapefile if requested and inventory available
    if out_shp:
        if not inventory or not os.path.exists(inventory):
            print('inventory path required and must exist to write shapefile')
            return summary_df

        try:
            with open(inventory) as fh:
                inv_lines = fh.readlines()
            inv_df = pd.DataFrame([row.split() for row in inv_lines],
                                  columns=['station', 'latitude', 'longitude', 'element', 'firstyear', 'lastyear'])
            inv_df['latitude'] = pd.to_numeric(inv_df['latitude'], errors='coerce')
            inv_df['longitude'] = pd.to_numeric(inv_df['longitude'], errors='coerce')
            inv_df = inv_df.drop_duplicates(subset=['station'])[['station', 'latitude', 'longitude']]

            merged = summary_df.merge(inv_df, on='station', how='left')
            merged = merged.dropna(subset=['latitude', 'longitude'])
            gdf = gpd.GeoDataFrame(merged,
                                   geometry=gpd.points_from_xy(merged['longitude'], merged['latitude']),
                                   crs='EPSG:4326')
            gdf.to_file(out_shp, crs='EPSG:4326', engine='fiona')
        except Exception as e:
            print(f"Failed writing shapefile {out_shp}: {e}")

    return summary_df


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    ghcn = os.path.join(d, 'climate', 'ghcn')
    inventroy_ = os.path.join(ghcn, 'ghcnd-inventory.txt')
    rec_dir = os.path.join(ghcn, 'station_data')

    out_csv_ = os.path.join(ghcn, 'station_counts.csv')
    out_shp_ = os.path.join(ghcn, 'station_counts.shp')
    glob_ = 'ghcn_CANUSA_stations_mgrs'
    stations_csv = os.path.join(d, 'climate', 'ghcn', 'stations', f'{glob_}.csv')
    landsat_ = os.path.join(d, 'dads', 'rs', 'landsat', 'station_data')
    cdr_ = os.path.join(d, 'dads', 'rs', 'cdr', 'joined')
    solrad = os.path.join(d, 'dads', 'dem', 'rsun_stations')
    terrain = os.path.join(d, 'dads', 'dem', 'terrain', 'station_data')
    training = '/data/ssd2/dads/training/parquet'
    joined = '/data/ssd2/dads/met/joined'

    assess_downloaded_data(rec_dir,
                           out_csv=out_csv_,
                           inventory=inventroy_,
                           out_shp=out_shp_,
                           joined_dir=joined,
                           training_dir=training,
                           landsat_dir=landsat_,
                           cdr_dir=cdr_,
                           solrad_dir=solrad,
                           terrain_dir=terrain,
                           stations_csv=stations_csv,
                           num_workers=12)

# ========================= EOF ====================================================================
