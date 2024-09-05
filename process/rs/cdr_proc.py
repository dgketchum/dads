import os
import json
import concurrent.futures

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

AVHRR_VARS = ['SREFL_CH1',
              'SREFL_CH2',
              'SREFL_CH3',
              'BT_CH3',
              'BT_CH4',
              'BT_CH5']

VIIRS_VARS = ['BRDF_corrected_I1_SurfRefl_CMG',
              'BRDF_corrected_I2_SurfRefl_CMG',
              'BRDF_corrected_I3_SurfRefl_CMG',
              'BT_CH12',
              'BT_CH15',
              'BT_CH16']

HARMONIZED_VARS = ['SR1', 'SR2',
                   'SR3', 'BT1',
                   'BT2', 'BT3']


def extract_surface_reflectance(stations, gridded_dir, incomplete_out, out_data, overwrite=False, bounds=None,
                                num_workers=1):
    if os.path.exists(incomplete_out):
        with open(incomplete_out, 'r') as f:
            incomplete = json.load(f)
    else:
        incomplete = {'missing': []}

    station_list = pd.read_csv(stations, index_col='fid')

    if bounds:
        w, s, e, n = bounds
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]
    else:
        ln = station_list.shape[0]
        w, s, e, n = (-125.0, 25.0, -67.0, 53.0)
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]
        print('dropped {} stations outside NLDAS-2 extent'.format(ln - station_list.shape[0]))

    start, end = datetime(2000, 1, 1), datetime(2024, 8, 1)

    for year in range(start.year, end.year + 1):

        if year <= 2013:
            extract_vars = AVHRR_VARS
        else:
            extract_vars = VIIRS_VARS

        for month in range(1, 13):
            if year == start.year and month < start.month:
                continue
            if year == end.year and month > end.month:
                break

            month_start = datetime(year, month, 1)
            date_string = month_start.strftime('%Y%m')

            nc_files = [f for f in os.listdir(gridded_dir) if date_string == f.split('_')[-2][:6]]
            if not nc_files:
                print(f"No NetCDF files found for {year}-{month}")
                continue

            datasets, complete = [], True
            for f in nc_files:
                try:
                    nc_file = os.path.join(gridded_dir, f)
                    ds = xr.open_dataset(nc_file, engine='netcdf4', decode_cf=False)
                    datasets.append(ds.sel(latitude=slice(n, s), longitude=slice(w, e)))
                except Exception as exc:
                    incomplete['missing'].append(f)
                    print(f"Unreadable NetCDF files found for {year}-{month}")
                    complete = False
                    break

            if not complete:
                continue

            ds = xr.concat(datasets, dim='time')
            time_values = pd.to_datetime(ds['time'].values, unit='D', origin=pd.Timestamp('1981-01-01'))
            ds = ds.assign_coords(time=time_values).set_index(time='time')

            indexer = station_list[['latitude', 'longitude']].to_xarray()
            ds = ds.sel(latitude=indexer.latitude, longitude=indexer.longitude, method='nearest')

            # TODO: consider making use of zenith, overpass time and QA data
            fids = np.unique(indexer.fid.values).tolist()
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_fid, fid, year, month, ds, extract_vars,
                                           out_data, overwrite) for fid in fids]
                concurrent.futures.wait(futures)

    if len(incomplete) > 0:
        with open(incomplete_out, 'w') as fp:
            json.dump(incomplete, fp, indent=4)


def process_fid(fid, year, month, ds, extract_vars, out_data, overwrite):
    dst_dir = os.path.join(out_data, fid)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    _file = os.path.join(dst_dir, '{}_{}_{}.csv'.format(fid, year, month))

    if not os.path.exists(_file) or overwrite:
        df_station = ds.sel(fid=fid).to_dataframe()
        df_station = df_station.groupby(df_station.index.get_level_values('time').date).first()
        df_station = df_station[extract_vars]
        df_station.to_csv(_file)
        print(_file)


def join_station_data(in_dir, dst_dir, n_workers, overwrite=False):
    """"""

    dir_list = [f for f in os.listdir(in_dir)]
    station_names = set(f.split('_')[0] for f in dir_list)

    if n_workers == 1:
        for fid in station_names:
            process_station(fid, in_dir, dst_dir, overwrite)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_station, fid, in_dir, dst_dir, overwrite) for fid in station_names]
            concurrent.futures.wait(futures)


def process_station(fid, in_dir, dst_dir, overwrite=False):
    """"""

    out_file = os.path.join(dst_dir, f'{fid}.csv')
    if os.path.exists(out_file) and not overwrite:
        print(os.path.basename(out_file), 'exists')
        return

    _dir = os.path.join(in_dir, fid)
    if not os.path.exists(_dir):
        print(_dir, 'does not exist')
        return

    files_ = [os.path.join(_dir, f) for f in os.listdir(_dir) if f.endswith('.csv')]
    adfs, vdfs = [], []
    for f in files_:
        df = pd.read_csv(os.path.join(in_dir, f), index_col=0, parse_dates=True)
        year = int(f.split('_')[1])
        if year <= 2013:
            df.rename(columns=dict(zip(AVHRR_VARS, HARMONIZED_VARS)), inplace=True)
            df[df[HARMONIZED_VARS] < 0] = 0
            adfs.append(df)
        else:
            df.rename(columns=dict(zip(VIIRS_VARS, HARMONIZED_VARS)), inplace=True)
            df[df[HARMONIZED_VARS] < 0] = 0
            vdfs.append(df)

    avhrr_df = pd.concat(adfs, axis=0, ignore_index=False)
    avhrr_mean = avhrr_df.mean()
    avhrr_std = avhrr_df.std()

    viirs_df = pd.concat(vdfs, axis=0, ignore_index=False)
    viirs_mean = viirs_df.mean()
    viirs_std = viirs_df.std()
    normalized_viirs_df = (viirs_df - viirs_mean) * (avhrr_std / viirs_std) + avhrr_mean

    df = pd.concat([avhrr_df, normalized_viirs_df], axis=0, ignore_index=False)
    df = df.sort_index()
    df = df.resample('D').asfreq()
    df = df.ffill()
    df = df[HARMONIZED_VARS].astype(int)
    df.to_csv(out_file)
    print(os.path.basename(out_file))


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS')

    # pandarallel.initialize(nb_workers=6)

    madis_data_dir_ = os.path.join(d, 'climate', 'madis')
    sites = os.path.join(d, 'dads', 'met', 'stations', 'dads_stations_elev_mgrs.csv')

    grid_dir = os.path.join(d, 'dads', 'rs', 'cdr', 'nc')
    csv_m_dir = os.path.join(d, 'dads', 'rs', 'cdr', 'csv')
    incomp = os.path.join(d, 'dads', 'rs', 'cdr', 'incomplete_files.json')

    workers = 8
    # extract_surface_reflectance(sites, grid_dir, incomp, csv_m_dir, num_workers=workers,
    #                             overwrite=False, bounds=(-180., 25., -60., 85.))

    joined_dir = os.path.join(d, 'dads', 'rs', 'cdr', 'joined')
    join_station_data(csv_m_dir, joined_dir, workers, overwrite=False)

# ========================= EOF ====================================================================
