import os
import pandas as pd
from tqdm import tqdm


def process_landsat(stations, rs_dir, out_dir, glob=None, shuffle=False, overwrite=False, extrapolate=False,
                    index_col='fid'):
    """"""

    stations = pd.read_csv(stations)
    stations.index = stations[index_col]
    stations.sort_index(inplace=True)

    if shuffle:
        stations = stations.sample(frac=1)

    ts, ct, scaling, first, shape = None, 0, {}, True, None

    scaling['stations'] = []

    tiles = stations['MGRS_TILE'].unique()

    for i, tile in enumerate(tiles, start=1):

        # if tile != '10UFU':
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
            v.to_csv(out_file)
            print('wrote', os.path.basename(out_file))


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

    rs_files = [(y, os.path.join(rs_directory, f'{glob}_500_{y}_{_tile}.csv')) for y in range(1990, 2024)]

    exist = [os.path.exists(f) for y, f in rs_files]

    if not all(exist) and not extrapolate:
        s = sum(exist)
        print(f'{_tile}, {s} of {len(exist)} extracts available, skipping')
        return None

    elif not all(exist) and extrapolate:
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
                years = range(1990, 2023)

                def repeat_data_for_year(year):
                    new_df = df.copy()
                    new_df.index = new_df.index.map(lambda d: d.replace(year=year))
                    return new_df

                df = pd.concat([repeat_data_for_year(year) for year in years])
                df = df.resample('D').ffill()
            data[index].append(df)

    data = {k: pd.concat(v) for k, v in data.items() if k not in exclude}

    return data


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'

    glob_ = 'ghcn_CANUSA_stations_mgrs'

    fields = os.path.join(d, 'met', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    rs = os.path.join(d, 'rs', 'ghcn_stations', 'landsat', 'tiles')
    out = os.path.join(d, 'rs', 'landsat', 'station_data')
    process_landsat(fields, rs, out, glob=glob_, shuffle=True, overwrite=False, extrapolate=True, index_col='STAID')

# ========================= EOF ====================================================================
