import calendar
import os
import gc
from datetime import datetime, date
from dateutil.rrule import rrule, DAILY
import urllib.request
import zipfile
import concurrent.futures

import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr


PRISM_VARIABLES = ['ppt', 'tmin', 'tmax', 'tdmean', 'vpdmax', 'vpdmin', 'tmean']


def process_prism_data(stations, nc_dir, out_data, tmp_dir, start_year=1990, end_year=2023, workers=32,
                       overwrite=False, bounds=None, debug=True):
    """"""
    station_list = pd.read_csv(stations)
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
        print('dropped {} stations outside NLDAS-2 extent'.format(ln - station_list.shape[0]))

    print(f'{len(station_list)} stations to write')

    station_list = station_list.sample(frac=1)
    station_list = station_list.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    indexer = station_list[['lat', 'lon']].to_xarray()
    fids = np.unique(indexer.fid.values).tolist()

    years = [year for year in range(start_year, end_year + 1)]
    urls = [create_url_list(year) for year in years]

    if debug:
        for url_set in urls:
            proc_time_slice(url_set, indexer, fids, nc_dir, out_data, tmp_dir, overwrite)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(proc_time_slice, url_set, indexer, fids,  nc_dir, out_data, tmp_dir, overwrite)
                   for url_set in urls]
        concurrent.futures.wait(futures)


def proc_time_slice(urls_, indexer_, fids_, nc_dir_, out_, temp, overwrite_):

    year_str = os.path.basename(urls_[0]).split('_')[4][:4]
    nc_path = os.path.join(nc_dir_, f'prism_{year_str}.nc')

    if not os.path.exists(nc_path):
        print(f'netcdf {os.path.basename(nc_path)} does not exist, building')
        time_coords = []
        var_dct = {k: [] for k in PRISM_VARIABLES}
        for url in urls_:
            try:
                temp_zip = os.path.join(temp, 'temp.zip')
                urllib.request.urlretrieve(url, temp_zip)
                with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                    zip_ref.extractall('.')
                    bil_file = zip_ref.namelist()[0]

                split = bil_file.split('_')
                date_str = split[4]
                variable = split[1]
                year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:])
                dt = datetime(year, month, day)
                if dt not in time_coords:
                    time_coords.append(dt)

                da = rxr.open_rasterio(bil_file, masked=True, crs='EPSG:4269')
                da = da.squeeze('band', drop=True)
                da.attrs['crs'] = da.rio.crs.to_wkt()
                da = da.expand_dims(time=[dt])
                var_dct[variable].append(da)

                os.remove(temp_zip)
                [os.remove(f) for f in zip_ref.namelist()]

            except Exception as e:
                print(f'Error processing {url}: {e}')

        monthly_datasets = {var: [] for var in PRISM_VARIABLES}
        for var in PRISM_VARIABLES:
            daily_data = []
            for dt in time_coords:
                for da in var_dct[var]:
                    if da.time.values[0].astype('datetime64[D]') == np.datetime64(dt, 'D'):
                        daily_data.append(da)
                        break

            ds_month = xr.concat(daily_data, dim=pd.Index(time_coords, name='time'))
            ds_month = ds_month.to_dataset(name=var)
            monthly_datasets[var] = ds_month

        ds = xr.merge(monthly_datasets.values())
        ds = ds.rename({'x': 'lon', 'y': 'lat'})
        ds = ds.chunk({'time': len(time_coords), 'lat': 100, 'lon': 100})
        ds.to_netcdf(nc_path)
    else:
        ds = xr.open_dataset(nc_path, chunks={'time': -1, 'lat': 100, 'lon': 100})

    ds = ds.sel(lat=indexer_.lat, lon=indexer_.lon, method='nearest', tolerance=4000.)
    df_all = ds.to_dataframe().reset_index()

    ct, skip = 0, 0
    for fid in fids_:

        try:
            dst_dir = os.path.join(out_, fid)
            _file = os.path.join(dst_dir, '{}_{}.csv'.format(fid, year_str))

            if os.path.exists(_file) and os.path.getsize(_file) == 0:
                os.remove(_file)

            if not os.path.exists(_file) or overwrite_:
                df_station = df_all[df_all['fid'] == fid].copy()
                if np.isnan(df_station[PRISM_VARIABLES].values.sum()) > 0:
                    print(f'skipping {fid}')
                    skip += 1
                    continue
                df_station['dt'] = [dt.strftime('%Y%m%d%H') for dt in df_station['time']]
                df_station.drop(columns=['time'], inplace=True)

                if not os.path.exists(dst_dir):
                    os.mkdir(dst_dir)

                df_station.to_csv(_file, index=False)
                ct += 1
                if ct % 1000 == 0.:
                    print(f'{ct} of {len(fids_)} for {year_str} on {fid}')
            else:
                skip += 1
        except Exception as exc:
            print(f'{exc} on {fid}')
            return

    del ds, df_all
    gc.collect()
    print(f'wrote {ct} for {year_str}, skipped {skip}, {datetime.strftime(datetime.now(), '%Y%m%d %H:%M')}')


def create_url_list(year):
    url_list = []
    base_url = 'https://ftp.prism.oregonstate.edu/daily'

    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)

    for dt in rrule(DAILY, dtstart=start_date, until=end_date):
        date_str = dt.strftime('%Y%m%d')
        for var in PRISM_VARIABLES:
            url = f'{base_url}/{var}/{dt.year}/PRISM_{var}_stable_4kmD2_{date_str}_bil.zip'
            url_list.append(url)
    return url_list


if __name__ == '__main__':

    d = os.path.join('/home', 'ubuntu', 'data', 'IrrigationGIS')
    prism = os.path.join('/home', 'ubuntu', 'data', 'prism')

    if not os.path.isdir(d):
        d = os.path.join('/data', 'IrrigationGIS')
        prism = os.path.join('/data', 'prism')

    # sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_29OCT2024.csv')

    out_files = os.path.join(prism, 'station_data')
    nc_files_ = os.path.join(prism, 'netcdf')
    temp_ = os.path.join(prism, 'temp')

    process_prism_data(sites, nc_files_, out_files, temp_, start_year=1990, workers=32, overwrite=False,
                       bounds=None, debug=True)

# ========================= EOF ====================================================================
