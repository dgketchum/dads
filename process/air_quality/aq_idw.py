import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
import dask
from dask.distributed import Client

from extract.air_quality.aqepa import AQS_PARAMETERS


def csv_to_netcdf(in_dir, metadata, output_file):
    aqm = pd.read_csv(metadata)
    aqm['st_fips'] = aqm['st_fips'].astype(int)
    aqm['co_fips'] = aqm['co_fips'].astype(int)
    aqm['site_no'] = aqm['site_no'].astype(int)
    aqm.index = aqm.apply(lambda r: f"{str(r['st_fips']).rjust(2, '0')}"
                                    f"{str(r['co_fips']).rjust(3, '0')}"
                                    f"{str(r['site_no']).rjust(4, '0')}", axis=1)

    files_ = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith('.csv')]

    dfs, lats, lons, fids = [], [], [], []
    for f in files_:
        fid = os.path.basename(f).split('.')[0]
        if fid not in aqm.index:
            continue
        lat, lon = aqm.loc[fid, 'latitude'], aqm.loc[fid, 'longitude']
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df['date'] = df.index
        df['fid'] = fid
        lats.append(lat)
        lons.append(lon)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True, sort=False)
    df = df.set_index(['date', 'fid']).sort_index()
    ds = df.to_xarray()

    fid_to_coords = {fid: (lat, lon) for fid, lat, lon in zip(aqm.index, aqm['latitude'], aqm['longitude'])}
    ds.coords['latitude'] = ('fid', [fid_to_coords[fid][0] for fid in ds.fid.values])
    ds.coords['longitude'] = ('fid', [fid_to_coords[fid][1] for fid in ds.fid.values])
    ds.to_netcdf(output_file)


def calculate_raster(data, parameter, resolution=0.05, lat=(42.0, 49.0), lon=(-115, -103), tif=None):
    """"""
    latitudes = np.arange(lat[0], lat[1], resolution)
    longitudes = np.arange(lon[0], lon[1], resolution)

    param_data = data[parameter].dropna(dim='fid')
    grid_lon, grid_lat = np.meshgrid(longitudes, latitudes)
    obs_coords = np.column_stack((param_data['longitude'].values, param_data['latitude'].values))
    obs_values = param_data.values

    tree = cKDTree(obs_coords)
    distances, indices = tree.query(np.column_stack((grid_lon.ravel(), grid_lat.ravel())), k=len(obs_coords))
    weights = 1.0 / distances ** 5
    weights /= weights.sum(axis=1, keepdims=True)

    idw_values = np.sum(weights * obs_values[indices], axis=1)
    idw_raster = idw_values.reshape(grid_lat.shape)

    if tif:
        arr = xr.DataArray(idw_raster, coords=[('latitude', latitudes), ('longitude', longitudes)],
                           name=parameter)
        arr.rio.to_raster(tif)

    return idw_raster, latitudes, longitudes


def interpolate_at_points(data, parameter, points, resolution=0.05, lat=(42.0, 49.0), lon=(-115, -103),
                          out_raster=None):
    idw_raster, latitudes, longitudes = calculate_raster(data, parameter, resolution, lat, lon, tif=out_raster)
    lat_indices = ((points[:, 1] - lat[0]) / resolution).astype(int)
    lon_indices = ((points[:, 0] - lon[0]) / resolution).astype(int)
    interpolated_values = idw_raster[lat_indices, lon_indices]
    return interpolated_values


def interpolate_station_aq(nc_file, output, stations, raster_dst=None):
    """"""
    stations = pd.read_csv(stations, index_col=0)

    params = ['pm2.5', 'pm10', 'no2', 'ozone', 'so2']
    mds = xr.open_dataset(nc_file)
    west, east = np.min(mds.longitude.values), np.max(mds.longitude.values)
    south, north = np.min(mds.latitude.values), np.max(mds.latitude.values)

    north = 49.0

    stations = stations[(stations['latitude'] < north) & (stations['latitude'] >= south)]
    stations = stations[(stations['longitude'] < east) & (stations['longitude'] >= west)]
    points = stations[['longitude', 'latitude']]

    client = Client()

    tasks = []
    for year in range(2000, 2025):
        tasks.append(process_year(year, mds, points, params, south, north, west, east, output, raster_dst))

    dask.compute(*tasks)
    client.close()


@dask.delayed
def process_year(year, mds, points, params, south, north, west, east, ouput_dir, raster_dst):
    start, end = f'{year}-01-01', f'{year}-12-31'
    dt = pd.date_range(start, end, freq='D')
    ds = mds.sel(date=slice(start, end))

    results = np.ones((len(dt), len(params), points.shape[0])) * np.nan

    for d, day in enumerate(ds.date.values):
        dt = pd.to_datetime(day)
        dt_str = dt.strftime("%Y%m%d")
        daily_data = ds.sel(date=day)

        for i, parameter in enumerate(params):

            if np.all(np.isnan(daily_data[parameter])):
                continue

            if np.count_nonzero(~np.isnan(daily_data[parameter].values)) == 1:
                continue

            if raster_dst:
                out_file = os.path.join(raster_dst, f'idw_{parameter}_{dt_str}.tif')
            else:
                out_file = None

            try:
                interp_vals = interpolate_at_points(daily_data, parameter, points.values, resolution=0.05,
                                                    lat=(south, north), lon=(west, east), out_raster=out_file)
            except Exception as exception_:
                print(parameter, dt_str, exception_)
                continue

            results[d, i, :] = interp_vals

    for k, fid in enumerate(points.index):
        df = pd.DataFrame(data=results[:, :, k], columns=params)
        df['datetime'] = dt

        out_dir = os.path.join(ouput_dir, fid)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        out_file = os.path.join(out_dir, f'{fid}_{year}.csv')
        df.to_csv(out_file, index=False)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/dads'

    aq_d = os.path.join(root, 'aq')
    aq_csv_data = os.path.join(root, 'aq', 'joined_data')
    aq_summary = os.path.join(aq_d, 'aqs.csv')

    # aq_nc = os.path.join(aq_d, 'netCDF', 'aqs_mt.nc')
    aq_nc = os.path.join(aq_d, 'netCDF', 'aqs.nc')

    aq_output = os.path.join(aq_d, 'dads_stations')
    aq_tif = os.path.join(aq_d, 'tif')

    # csv_to_netcdf(aq_csv_data, aq_summary, aq_nc)

    workers = 10
    sites = os.path.join(root, 'met', 'stations', 'dads_stations_res_elev_mgrs.csv')
    interpolate_station_aq(aq_nc, aq_output, sites, raster_dst=None)

# ========================= EOF ====================================================================
