import os
import pandas as pd


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


def build_landsat_tables(rs_directory, tile, glob, terrain_feats):
    """"""

    rs_files = [(y, os.path.join(rs_directory, f'{glob}_500_{y}_{tile}.csv')) for y in range(1990, 2024)]
    dfl, data = [], None

    for y, f in rs_files:

        periods = landsat_periods(y)

        mdf = pd.read_csv(f)
        mdf['year'] = y
        mdf.index = mdf['fid']

        if data is None:
            data = {i: [] for i in mdf.index}

        for index, row in mdf.iterrows():
            df = pd.DataFrame(mdf.loc[index].T.copy())
            df.columns = ['val']
            df['param'] = [p.split('_')[0] for p in df.index]
            t = df.loc[terrain_feats].copy()

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

            for p in terrain_feats:
                df[p] = t.loc[p, 'val']

            data[index].append(df)

    data = {k: pd.concat(v) for k, v in data.items()}

    return data


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
