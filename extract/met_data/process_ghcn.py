import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def process_climate_data(stations, climate_data_dir, output_dir):
    df_stations = pd.read_csv(stations)
    station_ids = df_stations['STAID'].values

    coverage_by_day, exclude = {}, []
    metadata, station_scales = {}, {}
    all_max = 0.

    for station_id in station_ids:
        csv_path = os.path.join(climate_data_dir, f'{station_id}.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, parse_dates=['DATE'], index_col='DATE')
                df = df[['PRCP']] / 10.
                df = df.loc['2000-01-01': '2020-12-31']
            except KeyError:
                exclude.append(station_id)
                continue

            prcp_max = df['PRCP'].max()
            if prcp_max > all_max:
                all_max = prcp_max

            for date in df.index:
                coverage_by_day.setdefault(date, set()).add(station_id)

    metadata['max'] = all_max.item()

    date_index = pd.date_range(start=min(coverage_by_day.keys()), end=max(coverage_by_day.keys()), freq='D')
    coverage_df = pd.DataFrame(0, index=date_index, columns=station_ids)

    for date, stations_with_data in coverage_by_day.items():
        coverage_df.loc[date, list(stations_with_data)] = 1
    coverage_df['coverage'] = coverage_df.sum(axis=1)

    coverage_threshold = 0.9
    total_days = (coverage_df.index[-1] - coverage_df.index[0]).days + 1
    filtered_ids = list(coverage_df.columns[coverage_df.sum() > coverage_threshold * total_days])
    filtered_ids.remove('coverage')
    metadata['stations'] = filtered_ids

    for station_id in filtered_ids:
        if station_id in exclude:
            continue
        csv_path = os.path.join(climate_data_dir, f'{station_id}.csv')
        df = pd.read_csv(csv_path, parse_dates=['DATE'], index_col='DATE')[['PRCP']]
        df = df.loc['2000-01-01': '2020-12-31']
        df['precip'] = df['PRCP'] * 0.1 / metadata['max']
        scaled_csv = os.path.join(output_dir, 'scaled', f'{station_id}.csv')
        df.to_csv(scaled_csv)
        print(scaled_csv)

    coverage_df = coverage_df[filtered_ids]
    coverage_df.to_csv(os.path.join(output_dir, 'daily_station_coverage.csv'))

    js = os.path.join(output_dir, 'metadata.json')
    with open(js, 'w') as f:
        json.dump(metadata, f)
    print(js)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/dads'
    clim = '/media/research/IrrigationGIS/climate'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/dads'
        clim = '/home/dgketchum/data/IrrigationGIS/climate'

    fields = os.path.join(d, 'met', 'stations', 'ghcn_gallatin.csv')
    ghcn = os.path.join(clim, 'ghcn', 'ghcn_daily_summaries_4FEB2022')
    ghcn_out = os.path.join(d, 'met', 'obs', 'ghcn')

    process_climate_data(fields, ghcn, ghcn_out)
# ========================= EOF ====================================================================
