import os
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import haversine_distances
import numpy as np


def read_and_merge_rs_files(directory, stations_csv, out_csv):
    """"""
    stations = pd.read_csv(stations_csv)
    sites = [f.strip() for f in stations['fid']]
    stations.index = sites
    stations = stations.groupby(stations.index).agg({c: 'first' for c in stations.columns})

    df, ct, dates = pd.DataFrame(), 0, []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            day_df = pd.read_csv(filepath, index_col='fid')

            if day_df.empty:
                continue

            date_str = filename.replace('.csv', '').split('_')[-3:]
            date_str = '_'.join(date_str)
            dates.append(date_str)
            day_df.columns = [date_str]
            try:
                df = pd.concat([df, day_df], axis=1, ignore_index=False)
            except pd.errors.InvalidIndexError:
                day_df = day_df.groupby(day_df.index).agg({date_str: 'first'})
                df = pd.concat([df, day_df], axis=1, ignore_index=False)
            ct += 1
            if ct > 365:
                break

    match = [i for i in stations.index if i in df.index]
    df.loc[match, 'longitude'] = stations.loc[match, 'longitude']
    df.loc[match, 'latitude'] = stations.loc[match, 'latitude']

    df['latitude_rad'] = np.radians(df['latitude'])
    df['longitude_rad'] = np.radians(df['longitude'])

    tree = cKDTree(df[['latitude_rad', 'longitude_rad']].dropna().values)

    for date in dates:
        day_df = df[[date, 'latitude_rad', 'longitude_rad']].copy()
        for index, row in day_df[day_df[date].isnull()].iterrows():
            _, indices = tree.query([row['latitude_rad'], row['longitude_rad']], k=6)
            distances = haversine_distances(
                np.array([[row['latitude_rad'], row['longitude_rad']]]),
                df.loc[indices[1:], ['latitude_rad', 'longitude_rad']].values)[0]
            weights = 1 / distances
            weights /= weights.sum()
            df.at[index, 'value'] = np.sum(df.loc[indices[1:], 'value'].dropna().values * weights)

    df.drop(columns=['latitude_rad', 'longitude_rad'], inplace=True)
    df.to_csv(out_csv)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/dads'

    dem_d = os.path.join(root, 'dem')
    mgrs = os.path.join(root, 'training', 'w17_tiles.csv')

    sites_ = os.path.join(root, 'met', 'stations', 'dads_stations_WMT_mgrs.csv')
    raster_dir_ = os.path.join(root, 'rs', 'dads_stations', 'modis', 'extracts')
    out_csv_ = os.path.join(root, 'dem', 'rsun_tables')
    read_and_merge_rs_files(raster_dir_, sites_, out_csv_)
# ========================= EOF ====================================================================
