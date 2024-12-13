import calendar
import concurrent.futures
import gc
import json
import os
from datetime import datetime

import boto3
import cfgrib
import numpy as np
import pandas as pd
import xarray as xr
from herbie import FastHerbie
from botocore import UNSIGNED
from botocore.client import Config
from sklearn.neighbors import BallTree
from tqdm import tqdm

"""
NOAA GRIB Projection Info: https://www.nco.ncep.noaa.gov/pmb/docs/on388/tableb.html
"""


def extract_rtma(stations, out_data, grib, netcdf, model="rtma", workers=8, overwrite=False, bounds=None,
                 debug=False, start_yr=1990, end_yr=2024):
    """"""
    station_list = pd.read_csv(stations)
    if 'LAT' in station_list.columns:
        station_list = station_list.rename(columns={'STAID': 'fid', 'LAT': 'latitude', 'LON': 'longitude'})
    station_list.index = station_list['fid']

    w, s, e, n = bounds
    print(f'\n{os.path.basename(stations)}\nGeo Bounds Extent: x ({w}, {e}), y ({s}, {n})')

    station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
    station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]

    if len(station_list) < 1:
        print('No stations found in this area')
        return

    print(f'{len(station_list)} stations to write')

    station_list = station_list.sample(frac=1)
    station_list = station_list.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

    if 'END' in station_list.columns:
        station_list['end_dt'] = [pd.to_datetime(r['END']) for i, r in station_list.iterrows()]
        station_list = station_list[station_list['end_dt'] > pd.to_datetime('2016-01-01')]

    station_list = station_list.rename(columns={'lat': 'slat', 'lon': 'slon'})
    indexer = station_list[['slat', 'slon']].to_xarray()

    yrmo = [f'{year}{month:02}' for year in range(start_yr, end_yr + 1) for month in range(1, 13)]
    yrmo.reverse()

    print(f'{len(yrmo)} months to write')

    if debug:
        for dts in yrmo:
            proc_time_slice(indexer, dts, grib, netcdf, out_data, overwrite, model)

    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(proc_time_slice, indexer, dts, grib, netcdf, out_data, overwrite, model)
                       for dts in yrmo]
            concurrent.futures.wait(futures)


def proc_time_slice(indexer_, date_string_, grb_, nc_, out_, overwrite_, model_):
    """"""
    nc_file = os.path.join(nc_, f'{model}_{date_string_}.nc')

    if not os.path.exists(nc_file):
        ds = get_grb_files(date_string_, model_, grb_, nc_file)
    else:
        ds = xr.open_dataset(nc_file)

    # was unable to use xoak here for some reason it dropped FID index
    # must use a tree approach as the only meaningful spatial coordinates are in an irregular grid
    grid_points_radians = np.deg2rad(np.stack((ds['latitude'].values.ravel(), ds['longitude'].values.ravel()), axis=-1))
    station_points_radians = np.deg2rad(np.stack((indexer_['slat'].values, indexer_['slon'].values), axis=-1))
    tree = BallTree(grid_points_radians, metric='haversine')
    distances, indices = tree.query(station_points_radians, k=1)
    earth_radius = 6371
    distances_km = distances * earth_radius
    indexer_['distance'] = xr.DataArray(distances_km.flatten(), dims='fid')
    indices = indices.reshape(-1)
    y_indices = indices // ds.dims['x']
    x_indices = indices % ds.dims['x']
    ds = ds.sel(x=x_indices, y=y_indices)
    ds = xr.merge([ds, indexer_],  compat='override')
    df_all = ds.to_dataframe()
    ct, skip = 0, 0
    now = datetime.strftime(datetime.now(), '%Y%m%d %H:%M')
    print(f'prepare to write {date_string_}: {now}')

    fids_ = np.unique(indexer_.fid.values).tolist()
    for fid in fids_:

        try:
            dst_dir = os.path.join(out_, fid)
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)

            _file = os.path.join(dst_dir, '{}_{}.parquet'.format(fid, date_string_.replace('-', '')))

            if os.path.exists(_file) and os.path.getsize(_file) == 0:
                os.remove(_file)

            if not os.path.exists(_file) or overwrite_:
                df_station = df_all.loc[(0, 0, slice(None), fid)].copy()
                df_station['dt'] = [i.strftime('%Y%m%d%H') for i in df_station.index]
                df_station.to_parquet(_file, index=False)
                ct += 1
                if ct % 1000 == 0.:
                    print(f'{ct} of {len(fids_)} for {date_string_}')
            else:
                skip += 1
        except Exception as exc:
            print(f'{exc} on {fid}')
            return

    del ds, df_all
    gc.collect()
    now = datetime.strftime(datetime.now(), '%Y%m%d %H:%M')
    print(f'wrote {ct} for {date_string_}, skipped {skip}, {now}')


def get_grb_files(date_str, model, dst, nc_file, max_threads=6):
    """"""

    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    year, month = int(date_str[:4]), int(date_str[-2:])
    month_end = calendar.monthrange(year, month)[1]
    start, end = f'{year}-{month}-01 00:00', f'{year}-{month:02}-{month_end:02} 23:00'

    dates = pd.date_range(start=start, end=end, freq='1h')
    FH = FastHerbie(dates, model=model, max_threads=max_threads, **{'save_dir': dst,
                                                                    'priority': ['aws', 'nomads']})
    print(f'{date_str} found {len(FH.objects)} objects')

    target_dir = os.path.join(dst, date_str)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    dwn_first, dwn_ct = True, 0
    for obj in tqdm(FH.objects, desc='Download {model}', total=len(FH.objects)):
        try:
            remote_file = '/'.join(obj.SOURCES['aws'].split('/')[-2:])
            splt = obj.grib.split('/')
            dt = splt[-2].split('.')[-1]
            dt_hr_st = splt[-1].split('.')
            dt_hr_st.insert(1, dt)
            local_file = '.'.join(dt_hr_st)
            local_path = os.path.join(target_dir, local_file)
            obj.SOURCES['local'] = local_path

            if not os.path.exists(local_path):
                s3.download_file(f'noaa-{model}-pds', remote_file, local_path)
                dwn_ct += 1
                if dwn_first:
                    print(f'downloaded {local_path}')
                    dwn_first = False

        except Exception as exc:
            print(exc, date_str, 'on download')
            return None

    print(f'downloaded {dwn_ct} grib files for {date_str}')

    try:
        ds_list = [to_xarray(obj, model) for obj in FH.objects]
        ds_list = [xr.merge(dso, compat='override') for dso in ds_list]
        ds_list.sort(key=lambda x: x.time.data.max())
    except Exception as exc:
        print(exc, date_str, 'on list, merge, sort')
        return None

    ds_ = xr.combine_nested(ds_list, combine_attrs='drop_conflicts', concat_dim='time')
    ds_ = ds_.squeeze()

    ds_['longitude'] = ds_['longitude'] - 360.

    ds_ = ds_.chunk({'time': ds_.time.values.shape[0], 'y': 40, 'x': 60})
    ds_.to_netcdf(nc_file)

    return ds_


def to_xarray(h_object, model_):
    backend_kwargs = {}
    backend_kwargs.setdefault('indexpath', '')
    backend_kwargs.setdefault(
        'read_keys',
        ['parameterName', 'parameterUnits', 'stepRange', 'uvRelativeToGrid'],
    )
    backend_kwargs.setdefault('errors', 'raise')

    Hxr = cfgrib.open_datasets(h_object.SOURCES['local'], backend_kwargs=backend_kwargs)

    with open('rtma_cf_params.json', 'r') as f:
        cf_params = json.load(f)

    for ds in Hxr:
        ds.attrs['model'] = str(model_)
        ds.attrs['product'] = 'surface_meteorology'
        ds.attrs['description'] = f'{model_} surface meteorology'
        ds.attrs['remote_grib'] = str(h_object.SOURCES['aws'])
        ds.attrs['local_grib'] = str(h_object.SOURCES['local'])
        ds.attrs['search'] = 'all'
        ds.coords['gribfile_projection'] = None
        ds.coords['gribfile_projection'].attrs = cf_params
        ds.coords['gribfile_projection'].attrs['long_name'] = (
            f'{model_.upper()} model grid projection'
        )
        for var in list(ds):
            ds[var].attrs['grid_mapping'] = 'gribfile_projection'

    return Hxr


def get_quadrants(b):
    mid_longitude = (b[0] + b[2]) / 2
    mid_latitude = (b[1] + b[3]) / 2
    quadrant_nw = (b[0], mid_latitude, mid_longitude, b[3])
    quadrant_ne = (mid_longitude, mid_latitude, b[2], b[3])
    quadrant_sw = (b[0], b[1], mid_longitude, mid_latitude)
    quadrant_se = (mid_longitude, b[1], b[2], mid_latitude)
    quadrants = [quadrant_nw, quadrant_ne, quadrant_sw, quadrant_se]
    return quadrants


if __name__ == '__main__':

    home = os.path.expanduser('~')

    model = 'urma'

    d = '/media/research/IrrigationGIS'
    rtma = '/media/nvm/{}'.format(model)

    if not os.path.isdir(rtma):
        d = os.path.join(home, 'data', 'IrrigationGIS')
        rtma = os.path.join('/data', 'ssd1', f'{model}')

    if not os.path.isdir(rtma):
        d = os.path.join('/data', 'IrrigationGIS')
        rtma = os.path.join('/data', f'{model}')

    sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_29OCT2024.csv')

    out_files = os.path.join(rtma, 'station_data')
    out_netcdf_ = os.path.join(rtma, 'netcdf')
    out_grib_ = os.path.join(rtma, 'grib')

    bounds_ = (-124.0, 19.2, -66.0, 53.0)
    quadrants = get_quadrants(bounds_)

    for e, quad in enumerate(quadrants, start=1):
        print(f'\n\n\n\n Quadrant {e} \n\n\n\n')

        extract_rtma(sites, out_files, out_grib_, out_netcdf_, model=model, workers=1, overwrite=False,
                     bounds=quad, debug=True, start_yr=2014, end_yr=2023)

        sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
        extract_rtma(sites, out_files, out_grib_, out_netcdf_, model=model, workers=1, overwrite=False,
                     bounds=quad, debug=True, start_yr=2014, end_yr=2023)
# ========================= EOF ====================================================================
