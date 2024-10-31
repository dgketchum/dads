import os

import dask
import earthaccess
import pandas as pd
import xarray as xr
from dask.distributed import Client


def get_nldas(dst):
    earthaccess.login()
    print('earthdata access authenticated')
    results = earthaccess.search_data(
        doi='10.5067/THUF4J1RLSYG',
        temporal=('1990-01-01', '2023-12-31'))
    earthaccess.download(results, dst)


def extract_nldas(stations, nc_data, out_data, overwrite=False, bounds=None,
                  index_col='fid', debug=False):

    station_list = pd.read_csv(stations, index_col=index_col)

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

    station_list = station_list.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

    ds = xr.open_dataset(nc_data, engine='zarr', chunks={'time': -1})

    if debug:
        for fid, row in station_list.iterrows():
            process_fid(fid, row, ds, out_data, overwrite)
    else:
        delayed_process = [dask.delayed(process_fid)(fid, row, ds, out_data, overwrite)
                           for fid, row in station_list.iterrows()]
        dask.compute(*delayed_process)


def process_fid(fid, row, ds, out_data, overwrite):
    dst_dir = os.path.join(out_data, fid)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    _file = os.path.join(dst_dir, '{}.csv'.format(fid))

    if not os.path.exists(_file) or overwrite:
        df_station = ds.sel(lat=row['lat'], lon=row['lon'], method='nearest').to_dataframe()
        df_station = df_station.groupby(df_station.index.get_level_values('time')).first()
        df_station['dt'] = [i.strftime('%Y%m%d%H') for i in df_station.index]
        df_station.to_csv(_file, index=False)
        print(_file)


def main():
    client = Client(n_workers=64, memory_limit='256GB')

    d = '/media/research/IrrigationGIS'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS')

    nc_data_ = '/data/ssd1/nldas2_ts.zarr'
    # get_nldas(nc_data)

    # sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_mgrs_28OCT2024.csv')
    grid_dir = '/data/ssd1/nldas_raw/'

    extract_nldas(sites, nc_data_, grid_dir, overwrite=False, bounds=None, debug=True)


if __name__ == '__main__':
    main()
# ========================= EOF ====================================================================
