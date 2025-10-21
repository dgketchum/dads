import calendar
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import geopandas as gpd
import pandas as pd


def _process_tile(args):
    tile, i, total, rs_dir, out_dir, glob, overwrite, extrapolate, index_col = args
    test_file = os.path.join(rs_dir, f'{glob}_500_2023_{tile}.csv')
    if not os.path.exists(test_file):
        print('{} not processed'.format(os.path.basename(test_file)))
        return tile
    complete = check_exists(rs_file=test_file, out_directory=out_dir, index_col=index_col)
    if complete == 'all' and not overwrite:
        print('\n{} complete, {} of {}\n'.format(tile, i, total))
        return tile
    elif complete == 'partial' and extrapolate:
        print('\n{} partially complete, processing {} of {}\n'.format(tile, i, total))
        rs_dict = build_landsat_tables(rs_directory=rs_dir, _tile=tile, glob=glob,
                                       extrapolate=extrapolate, index_col=index_col)
    else:
        print('\n{} not yet processed, {} of {}\n'.format(tile, i, total))
        rs_dict = build_landsat_tables(rs_directory=rs_dir, _tile=tile, glob=glob,
                                       extrapolate=extrapolate, index_col=index_col)
    if rs_dict is None:
        return tile

    for k, v in rs_dict.items():

        out_file = os.path.join(out_dir, f'{k}.csv')

        if os.path.exists(out_file) and not overwrite:

            try:
                ex = pd.read_csv(out_file, index_col=0, parse_dates=True)
            except pd.errors.EmptyDataError:
                continue

            try:
                df_ = ex.copy()
                for c in v.columns:
                    df_.loc[v.index, c] = v[c]
                if 'repeated' in df_.columns:
                    df_.loc[v.index, 'repeated'] = 0
            except KeyError:
                continue

            to_write = df_
        else:
            to_write = v

        to_write.to_csv(out_file)
        print('wrote', os.path.basename(out_file))
    return tile


def process_landsat(stations, rs_dir, out_dir, glob=None, shuffle=False, overwrite=False, extrapolate=False,
                    index_col='fid', num_workers=1):
    """"""

    if stations.endswith('.csv'):
        stations = pd.read_csv(stations)
    else:
        stations = gpd.read_file(stations)

    stations.index = stations[index_col]
    stations.sort_index(inplace=True)

    if shuffle:
        stations = stations.sample(frac=1)

    ts, ct, scaling, first, shape = None, 0, {}, True, None

    scaling['stations'] = []

    tiles = stations['MGRS_TILE'].unique()

    if num_workers == 1:
        for i, tile in enumerate(tiles, start=1):

            # if tile != '20TQS':
            #     continue

            test_file = os.path.join(rs_dir, f'{glob}_500_2023_{tile}.csv')
            if not os.path.exists(test_file):
                print('{} not processed'.format(os.path.basename(test_file)))
                continue

            complete = check_exists(rs_file=test_file, out_directory=out_dir, index_col=index_col)

            if complete == 'all' and not overwrite:
                print('\n{} complete, {} of {}\n'.format(tile, i, len(tiles)))
                continue

            elif complete == 'partial' and extrapolate:
                print('\n{} partially complete, processing {} of {}\n'.format(tile, i, len(tiles)))
                rs_dict = build_landsat_tables(rs_directory=rs_dir, _tile=tile, glob=glob,
                                               extrapolate=extrapolate, index_col=index_col)

            else:
                print('\n{} not yet processed, {} of {}\n'.format(tile, i, len(tiles)))
                rs_dict = build_landsat_tables(rs_directory=rs_dir, _tile=tile, glob=glob,
                                               extrapolate=extrapolate, index_col=index_col)

            if rs_dict is None:
                continue

            for k, v in rs_dict.items():
                out_file = os.path.join(out_dir, f'{k}.csv')
                if os.path.exists(out_file) and not overwrite:
                    ex = pd.read_csv(out_file, index_col=0, parse_dates=True)
                    df_ = ex.copy()
                    for c in v.columns:
                        df_.loc[v.index, c] = v[c]
                    if 'repeated' in df_.columns:
                        df_.loc[v.index, 'repeated'] = 0
                    to_write = df_
                else:
                    to_write = v

                to_write.to_csv(out_file)
                print('wrote', os.path.basename(out_file))
    else:
        total = len(tiles)
        args = [(tile, i, total, rs_dir, out_dir, glob, overwrite, extrapolate, index_col)
                for i, tile in enumerate(tiles, start=1)]
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for _ in ex.map(_process_tile, args):
                pass


def check_exists(rs_file, out_directory, index_col):
    mdf = pd.read_csv(rs_file)
    fids = mdf[index_col].to_list()
    files = [os.path.join(out_directory, '{}.csv'.format(f)) for f in fids]
    if all([os.path.exists(f) for f in files]):
        return 'all'
    elif any([os.path.exists(f) for f in files]):
        return 'partial'
    else:
        return 'none'


def landsat_periods(yr):
    winter_s, winter_e = '{}-01-01'.format(yr), '{}-03-01'.format(yr),
    spring_s, spring_e = '{}-03-01'.format(yr), '{}-05-01'.format(yr),
    late_spring_s, late_spring_e = '{}-05-01'.format(yr), '{}-07-15'.format(yr)
    summer_s, summer_e = '{}-07-15'.format(yr), '{}-09-30'.format(yr)
    fall_s, fall_e = '{}-09-30'.format(yr), '{}-12-31'.format(yr)

    periods = [('0', winter_s, spring_s),
               ('1', spring_s, spring_e),
               ('2', late_spring_s, late_spring_e),
               ('3', summer_s, summer_e),
               ('4', fall_s, fall_e)]

    periods = {i[0]: pd.to_datetime(i[1]) for i in periods}
    return periods


def build_landsat_tables(rs_directory, _tile, glob, index_col, extrapolate=False):
    """"""

    rs_files = [(y, os.path.join(rs_directory, f'{glob}_500_{y}_{_tile}.csv')) for y in range(1987, 2025)]

    exist = [os.path.exists(f) for y, f in rs_files]

    if not all(exist) and not extrapolate:
        s = sum(exist)
        print(f'{_tile}, {s} of {len(exist)} extracts available, skipping')
        return None

    elif not all(exist) and extrapolate:
        rs_files = [f for f in rs_files if os.path.exists(f[1])]
        rs_files = rs_files[-1:]

    dfl, data, exclude = [], None, []

    for y, f in rs_files:

        periods = landsat_periods(y)

        mdf = pd.read_csv(f)
        mdf['year'] = y
        mdf.index = mdf[index_col]

        if data is None:
            data = {i: [] for i in mdf.index}

        for index, row in mdf.iterrows():

            df = pd.DataFrame(mdf.loc[index].T.copy())

            try:
                df.columns = ['val']
                df['param'] = [p.split('_')[0] for p in df.index]
            except ValueError:
                exclude.append(index)
                continue

            df['period'] = [c.split('_')[-1] for c in df.index]
            ls_vars = [c if c.split('_')[-1] in periods.keys() else None for c in df.index]
            ls_vars = [c for c in ls_vars if c is not None]

            df = df.loc[ls_vars]
            df['date'] = df['period'].map(periods)
            df.drop(columns=['period'], inplace=True)
            df = df.pivot(columns=['param'], index='date')
            df.columns = [c[1] for c in df.columns]
            df = df.apply(pd.to_numeric)

            dt = pd.DatetimeIndex(pd.date_range(f'{y}-01-01', f'{y}-12-31'))
            df = df.reindex(dt)
            df = df.ffill().bfill()

            if not all(exist) and extrapolate:
                years = list(range(1987, 2025))
                years.remove(2023)
                def _safe_replace_year(ts, year):
                    # Handle leap-day and month-end differences safely when changing the year
                    month = ts.month
                    day = ts.day
                    max_day = calendar.monthrange(year, month)[1]
                    return ts.replace(year=year, day=min(day, max_day))

                def repeat_data_for_year(year):
                    new_df = df.copy()
                    new_df.index = new_df.index.map(lambda d: _safe_replace_year(d, year))
                    return new_df

                df = pd.concat([repeat_data_for_year(year) for year in years])
                # Ensure unique, sorted DatetimeIndex before upsampling
                if not df.index.is_monotonic_increasing:
                    df = df.sort_index()
                if df.index.has_duplicates:
                    # Collapse duplicates conservatively by taking the last occurrence
                    df = df[~df.index.duplicated(keep='last')]
                df = df.resample('D').ffill()
            data[index].append(df)

    data = {k: pd.concat(v) for k, v in data.items() if k not in exclude}

    return data


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    out = os.path.join(d, 'dads', 'rs', 'landsat', 'station_data')

    # glob_ = 'ghcn_CANUSA_stations_mgrs'
    # index_ = 'STAID'
    # fields = os.path.join(d, 'climate', 'ghcn', 'stations', f'{glob_}.csv')
    # rs =  os.path.join(d, 'dads', 'rs', 'landsat', 'updates', 'ghcn')

    # glob_ = 'madis_17MAY2025_mgrs'
    # index_ = 'fid'
    # fields = os.path.join(d, 'dads', 'met', 'stations', f'{glob_}.csv')
    # rs = os.path.join(d, 'dads', 'rs', 'landsat', 'updates', 'madis')

    # NDBC configuration
    glob_ = 'ndbc_stations'
    index_ = 'station_id'
    fields = os.path.join(d, 'climate', 'ndbc', 'ndbc_meta', 'ndbc_stations.csv')
    rs = os.path.join(d, 'dads', 'rs', 'landsat', 'updates', 'ndbc')

    num_workers = 6
    if 'madis' in glob_:
        raise ValueError('MADIS data needs to be appended to exising files, functionality that does not exist')
    process_landsat(fields, rs, out, glob=glob_, shuffle=True, overwrite=False,
                    extrapolate=True, index_col=index_, num_workers=num_workers)

# ========================= EOF ====================================================================
