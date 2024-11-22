import concurrent.futures
import os

import earthaccess
import numpy as np
import pandas as pd
import xarray as xr
import pyproj
import cartopy.crs as ccrs


def get_daymet(start_date, end_date, down_dst=None):
    results = earthaccess.search_data(
        doi='10.3334/ORNLDAAC/2129',
        temporal=(start_date, end_date))
    if down_dst:
        print(f'downloading netcdf')
        for granule in results:
            nc_id = granule['meta']['native-id']
            split = nc_id.split('_')
            region, param = split[5], split[6]
            file_name = '.'.join(nc_id.split('.')[1:])
            if param in ['tmax', 'tmin', 'vp', 'prcp', 'srad'] and region == 'na':
                earthaccess.download(granule, down_dst)
                print(f'downloaded {file_name}')
    else:
        return results


def extract_daymet(stations, out_data, nc_dir=None, workers=8, overwrite=False, bounds=None, debug=False,
                   start_yr=1990, end_yr=2024):
    station_list = pd.read_csv(stations)
    if 'LAT' in station_list.columns:
        station_list = station_list.rename(columns={'STAID': 'fid', 'LAT': 'latitude', 'LON': 'longitude'})
    station_list.index = station_list['fid']

    if bounds:
        w, s, e, n = bounds
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]
        w, s = projected_coords({'lon': w, 'lat': s})
        e, n = projected_coords({'lon': e, 'lat': n})
        proj_bounds = w, s, e, n
        if w >= e or s >= n:
            raise ValueError(f'Invalid projected bounds: {proj_bounds}')
        else:
            print(f'Bounds Extent: x({w}, {e}), ({s}, {n})')
    else:
        proj_bounds = None
        ln = station_list.shape[0]
        w, s, e, n = (-178.1333, 14.0749, -53.0567, 82.9143)
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]
        print('dropped {} stations outside DAYMET NA extent'.format(ln - station_list.shape[0]))

    print(f'{len(station_list)} stations to write')

    station_list = station_list.sample(frac=1)
    station_list = station_list.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    station_list[['x', 'y']] = station_list.apply(projected_coords, axis=1, result_type='expand')
    indexer = station_list[['x', 'y']].to_xarray()
    fids = np.unique(indexer.fid.values).tolist()

    vars_list = ['tmax', 'tmin', 'vp', 'prcp', 'srad']
    target_file_fmt = 'daymet_v4_daily_{}_{}_{}.nc'
    target_files = [[os.path.join(nc_dir, target_file_fmt.format('na', var_, year)) for var_ in vars_list]
                    for year in range(start_yr, end_yr)]
    years = list(range(start_yr, end_yr))

    if not all([os.path.exists(tp) for l in target_files for tp in l]):
        raise NotImplementedError('Missing files')

    if debug:
        for fileset, dts in zip(target_files, years):
            proc_time_slice(fileset, indexer, dts, fids, out_data, overwrite, bounds_=proj_bounds)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(proc_time_slice, fileset, indexer, dts, fids, out_data, overwrite,
                                   bounds_=proj_bounds) for fileset, dts in zip(target_files, years)]
        concurrent.futures.wait(futures)


def proc_time_slice(fileset_, indexer_, date_string_, fids_, out_, overwrite_, bounds_=None):
    ds = xr.open_mfdataset(fileset_, engine='netcdf4')
    ds = ds.chunk({'time': len(ds.time.values)})
    if bounds_ is not None:
        ds = ds.sel(y=slice(bounds_[3], bounds_[1]), x=slice(bounds_[0], bounds_[2]))
    ds = ds.sel(y=indexer_.y, x=indexer_.x, method='nearest', tolerance=1000)
    df_all = ds.to_dataframe()

    ct = 0
    for fid in fids_:
        dst_dir = os.path.join(out_, fid)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        _file = os.path.join(dst_dir, '{}_{}.csv'.format(fid, date_string_))

        if not os.path.exists(_file) or overwrite_:
            df_station = df_all.loc[(fid, slice(None), 0)].copy()
            df_station['dt'] = [i.strftime('%Y%m%d') for i in df_station.index]
            df_station.to_csv(_file, index=False)
            ct += 1
        if ct % 1000 == 0.:
            print(f'{ct} of {len(fids_)} for {date_string_}')
    print(f'wrote {ct} for {date_string_}')


def projected_coords(row):
    lcc = ccrs.LambertConformal(
        central_longitude=-100,
        central_latitude=42.5,
        standard_parallels=(25, 60),
        false_easting=0,
        false_northing=0,
        globe=ccrs.Globe(ellipse='WGS84')
    )
    source_crs = 'epsg:4326'
    target_crs = lcc.proj4_init
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x, y = transformer.transform(row['lon'], row['lat'])
    return x, y


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

    d = os.path.join('/home', 'ubuntu', 'data', 'IrrigationGIS')
    daymet = os.path.join('/home', 'ubuntu', 'data', 'daymet')

    if not os.path.isdir(d):
        d = os.path.join('/data', 'IrrigationGIS')
        daymet = os.path.join('/data', 'daymet')

    earthaccess.login()
    print('earthdata access authenticated')

    # sites = os.path.join(d, 'climate', 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_29OCT2024.csv')

    out_files = os.path.join(daymet, 'station_data')
    nc_files_ = os.path.join(daymet, 'netcdf')
    # bounds = (-68.0, 17.0, -64.0, 20.0)
    bounds = (-178., 7., -53., 83.)
    quadrants = get_quadrants(bounds)
    sixteens = [get_quadrants(q) for q in quadrants]
    sixteens = [x for xs in sixteens for x in xs]

    # get_daymet(f'{1990}-01-01', f'{1990}-01-31', nc_files_)

    for e, sector in enumerate(sixteens, start=1):
        print(f'\n\n\n\n Sector {e} of {len(sixteens)} \n\n\n\n')

        extract_daymet(sites, out_files, nc_dir=nc_files_, workers=32, overwrite=False,
                       bounds=sector, debug=True)

# ========================= EOF ====================================================================
