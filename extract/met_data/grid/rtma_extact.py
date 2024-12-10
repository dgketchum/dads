import calendar
import concurrent.futures
import gc
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from herbie import FastHerbie
import boto3
from botocore.exceptions import ClientError
from botocore import UNSIGNED
from botocore.client import Config
import cfgrib
import pygrib
from pyproj import CRS


def extract_rtma(stations, out_data, grib, netcdf, model="rtma", workers=8, overwrite=False, bounds=None,
                 debug=False, stream=False, start_yr=1990, end_yr=2024):
    """"""
    # TODO: remove the nrows arg!!!
    station_list = pd.read_csv(stations, nrows=100)
    if 'LAT' in station_list.columns:
        station_list = station_list.rename(columns={'STAID': 'fid', 'LAT': 'latitude', 'LON': 'longitude'})
    station_list.index = station_list['fid']

    if bounds:
        w, s, e, n = bounds
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]
    else:
        ln = station_list.shape[0]
        w, s, e, n = (-125.0, 25.0, -67.0, 53.0)
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]
        print('dropped {} stations outside RTMA extent'.format(ln - station_list.shape[0]))

    print(f'{len(station_list)} stations to write')

    station_list = station_list.sample(frac=1)
    station_list = station_list.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

    if 'END' in station_list.columns:
        station_list['end_dt'] = [pd.to_datetime(r['END']) for i, r in station_list.iterrows()]
        station_list = station_list[station_list['end_dt'] > pd.to_datetime('2016-01-01')]

    indexer = station_list[['lat', 'lon']].to_xarray()
    fids = np.unique(indexer.fid.values).tolist()

    yrmo = [f'{year}{month:02}' for year in range(start_yr, end_yr + 1) for month in range(1, 13)]
    yrmo = [yms for yms in yrmo if int(yms[:4]) > 2014]

    print(f'{len(yrmo)} months to write')

    if debug:
        for dts in yrmo:
            proc_time_slice(indexer, dts, fids, grib, netcdf, out_data, overwrite, model, stream)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(proc_time_slice, indexer, dts, fids, grib, netcdf, out_data, overwrite, model, stream)
                   for dts in yrmo]
        concurrent.futures.wait(futures)


def proc_time_slice(indexer_, date_string_, fids_, grb_, nc_, out_, overwrite_, model_, stream_=False):
    """"""
    nc_file = os.path.join(nc_, f'{model}_{date_string_}.nc')

    if not os.path.exists(nc_file):
        ds = get_grb_files(date_string_, model_, None, stream_data=stream_)
    else:
        ds = xr.open_dataset(nc_file)
    return

    ds = ds.chunk({'time': ds.time.values.shape[0], 'y': 40, 'x': 60})
    ds.to_netcdf(nc_file)

    try:
        ds = ds.sel(latitude=indexer_.lat, longitude=indexer_.lon, method='nearest')
        df_all = ds.to_dataframe()
        ct, skip = 0, 0
        now = datetime.strftime(datetime.now(), '%Y%m%d %H:%M')
        print(f'prepare to write {date_string_}: {now}')
    except Exception as exc:
        print(f'{exc} on {date_string_}')
        return

    for fid in fids_:

        try:
            dst_dir = os.path.join(out_, fid)
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)

            _file = os.path.join(dst_dir, '{}_{}.csv'.format(fid, date_string_.replace('-', '')))

            if os.path.exists(_file) and os.path.getsize(_file) == 0:
                os.remove(_file)

            if not os.path.exists(_file) or overwrite_:
                df_station = df_all.loc[(slice(None), fid)].copy()
                df_station['dt'] = [i.strftime('%Y%m%d%H') for i in df_station.index]
                df_station.to_csv(_file, index=False)
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


def get_grb_files(date_str, model, dst=None, max_threads=6, stream_data=False):
    """"""

    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    year, month = int(date_str[:4]), int(date_str[-2:])
    month_end = calendar.monthrange(year, month)[1]
    start, end = f'{year}-{month}-01 00:00', f'{year}-{month:02}-{month_end:02} 23:00'

    dates = pd.date_range(start=start, end=end, freq='1h')
    FH = FastHerbie(dates, model=model, max_threads=max_threads, **{'save_dir': dst})

    if dst:
        target_dir = os.path.join(dst, date_str)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

    if not stream_data and dst:
        for obj in FH.objects:
            try:
                remote_file = '/'.join(obj.SOURCES['aws'].split('/')[-2:])
                local_file = os.path.basename(obj.SOURCES['aws'])
                local_path = os.path.join(target_dir, local_file)
                obj.SOURCES['local'] = local_path

                if not os.path.exists(local_path):
                    s3.download_file(f'noaa-{model}-pds', remote_file, local_path)
                    print(f'downloaded {local_path}')

            except Exception as exc:
                print(exc, date_str, 'on download')
                return None

    try:
        ds_list = [to_xarray(H) for H in FH.objects]
        ds_list = [xr.merge(dso, compat='override') for dso in ds_list]
        ds_list.sort(key=lambda x: x.time.data.max())
    except Exception as exc:
        print(exc, date_str, 'on list, merge, sort')
        return None

    ds_ = xr.combine_nested(ds_list, combine_attrs='drop_conflicts', concat_dim='time')
    ds_ = ds_.squeeze()

    return ds_


def to_xarray(h_object, model):
    backend_kwargs = {}
    backend_kwargs.setdefault('indexpath', '')
    backend_kwargs.setdefault(
        'read_keys',
        ['parameterName', 'parameterUnits', 'stepRange', 'uvRelativeToGrid'],
    )
    backend_kwargs.setdefault('errors', 'raise')


    Hxr = cfgrib.open_datasets(h_object, backend_kwargs=backend_kwargs)

    with pygrib.open(str(local_file)) as grb:
        msg = grb.message(1)
        cf_params = CRS(msg.projparams).to_cf()

    for ds in Hxr:
        ds.attrs['model'] = str(self.model)
        ds.attrs['product'] = str(self.product)
        ds.attrs['description'] = self.DESCRIPTION
        ds.attrs['remote_grib'] = str(self.grib)
        ds.attrs['local_grib'] = str(local_file)
        ds.attrs['search'] = str(search)
        ds.coords['gribfile_projection'] = None
        ds.coords['gribfile_projection'].attrs = cf_params
        ds.coords['gribfile_projection'].attrs['long_name'] = (
            f'{self.model.upper()} model grid projection'
        )

        # Assign this grid_mapping for all variables
        for var in list(ds):
            ds[var].attrs['grid_mapping'] = 'gribfile_projection'

        if len(Hxr) == 1:
            return Hxr[0]
        else:
            raise ValueError

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

    model = 'rtma'

    d = '/media/research/IrrigationGIS'
    rtma = '/media/nvm/{}'.format(model)

    if not os.path.isdir(rtma):
        d = os.path.join(home, 'data', 'IrrigationGIS')
        rtma = os.path.join('/data', 'ssd1', f'{model}')

    if not os.path.isdir(rtma):
        d = os.path.join(home, 'data', 'IrrigationGIS')
        rtma = os.path.join(home, 'data', f'{model}')

    sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_29OCT2024.csv')

    out_files = os.path.join(rtma, 'station_data')
    out_netcdf_ = os.path.join(rtma, 'netcdf')
    out_grib_ = os.path.join(rtma, 'grib')

    bounds_ = (-124.0, 23.0, -66.0, 52.0)
    quadrants = get_quadrants(bounds_)

    for e, quad in enumerate(quadrants, start=1):
        print(f'\n\n\n\n Quadrant {e} \n\n\n\n')

        extract_rtma(sites, out_files, out_grib_, out_netcdf_, model='rtma', workers=1, overwrite=False,
                     bounds=quad, debug=True, stream=True, start_yr=2020, end_yr=2024)

# ========================= EOF ====================================================================
