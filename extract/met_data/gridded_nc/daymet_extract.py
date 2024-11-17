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
        earthaccess.download(results, down_dst)
    else:
        return results


def extract_daymet(stations, out_data, nc_data=None, workers=8, overwrite=False, bounds=None, debug=False,
                   parquet_check=None, missing_list=None, nc_dir=None):
    station_list = pd.read_csv(stations)
    if 'LAT' in station_list.columns:
        station_list = station_list.rename(columns={'STAID': 'fid', 'LAT': 'latitude', 'LON': 'longitude'})
    station_list.index = station_list['fid']

    if bounds:
        w, s, e, n = bounds
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]

    print(f'{len(station_list)} stations to write')

    station_list = station_list.sample(frac=1)
    station_list = station_list.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    station_list[['x', 'y']] = station_list.apply(projected_coords, axis=1, result_type='expand')
    indexer = station_list[['x', 'y']].to_xarray()
    fids = np.unique(indexer.fid.values).tolist()

    years, files = [], []

    for year in range(1990, 1991):
        nc_files = []
        granules = get_daymet(f'{year}-01-01', f'{year}-01-31')
        for granule in granules:
            nc_id = granule['meta']['native-id']
            split = nc_id.split('_')
            region, param = split[5], split[6]
            file_name = '.'.join(nc_id.split('.')[1:])
            if param in ['tmax', 'tmin', 'vp', 'prcp', 'srad'] and region == 'pr':
                target_file = os.path.join(nc_dir, file_name)
                nc_files.append((target_file, granule))

        if not nc_files:
            print(f"No NetCDF files found for {year}")
            continue

        years.append(year)
        files.append(nc_files)

    print(f'{len(years)} years to write')

    if debug:
        for fileset, dts in zip(files, years):
            proc_time_slice(fileset, indexer, dts, fids, out_data, overwrite, par_check=parquet_check, nc_dir_=nc_dir)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(proc_time_slice, fileset, indexer, dts, fids, out_data, overwrite, tmpdir=nc_dir)
                   for fileset, dts in zip(files, years)]
        concurrent.futures.wait(futures)


def proc_time_slice(fileset_, indexer_, date_string_, fids_, out_, overwrite_, par_check=None, nc_dir_=None):
    local_files, granules = [f[0] for f in fileset_], [f[1] for f in fileset_]
    try:
        if not all([os.path.exists(f) for f in local_files]):
            local_files = earthaccess.download(granules, nc_dir_, threads=4)
        ds = xr.open_mfdataset(local_files, engine='netcdf4')
    except Exception as exc:
        print(f'{exc}')
        return

    ds = ds.chunk({'time': len(ds.time.values)})
    ds = ds.sel(y=indexer_.y, x=indexer_.x, method='nearest')
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
    print(f'wrote {ct} for {date_string_}')


def projected_coords(row):
    globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6370000, semiminor_axis=6370000)
    lcc = ccrs.LambertConformal(globe=globe,
                                central_longitude=-100,
                                central_latitude=42.5,
                                standard_parallels=(25, 60))

    lcc_wkt = lcc.to_wkt()
    source_crs = 'epsg:4326'
    transformer = pyproj.Transformer.from_crs(source_crs, lcc_wkt)

    x, y = transformer.transform(row['lat'], row['lon'])
    return x, y

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
    nc_files = os.path.join(daymet, 'netcdf')
    bounds = (-68.0, 17.0, -64.0, 20.0)

    extract_daymet(sites, out_files, workers=14, overwrite=False, bounds=bounds, debug=True, nc_dir=nc_files)

# ========================= EOF ====================================================================
