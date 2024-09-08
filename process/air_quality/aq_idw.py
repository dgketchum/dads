import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


def csv_to_netcdf(in_dir, metadata, output_file):
    aqm = pd.read_csv(metadata)
    aqm.index = aqm.apply(lambda r: f"{str(r['st_fips'])}"
                                    f"{str(r['co_fips']).rjust(3, '0')}"
                                    f"{str(r['site_no']).rjust(4, '0')}", axis=1)
    files_ = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith('.csv')]

    dfs, lats, lons, fids = [], [], [], []
    for f in files_:
        fid = os.path.basename(f).split('.')[0]
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


def calculate_raster(data, parameter, resolution=0.05, lat=(42.0, 49.0), lon=(-115, -103)):
    """"""
    latitudes = np.arange(lat[0], lat[1], resolution)
    longitudes = np.arange(lon[0], lon[1], resolution)

    param_data = data[parameter].dropna(dim='fid')
    grid_lon, grid_lat = np.meshgrid(longitudes, latitudes)
    obs_coords = np.column_stack((param_data['longitude'].values, param_data['latitude'].values))
    obs_values = param_data.values

    tree = cKDTree(obs_coords)
    distances, indices = tree.query(np.column_stack((grid_lon.ravel(), grid_lat.ravel())), k=len(obs_coords))
    weights = 1.0 / distances ** 2
    weights /= weights.sum(axis=1, keepdims=True)

    idw_values = np.sum(weights * obs_values[indices], axis=1)
    idw_raster = idw_values.reshape(grid_lat.shape)
    arr = xr.DataArray(idw_raster, coords=[('latitude', latitudes), ('longitude', longitudes)],
                       name=parameter)

    return arr


def write_rasters(nc_file, output):
    ds = xr.open_dataset(nc_file)

    for day in ds.date.values:
        dt = pd.to_datetime(day)
        dt_str = dt.strftime("%Y%m%d")
        if dt_str != '20170907':
            continue

        daily_data = ds.sel(date=day)
        for parameter in ['pm2.5', 'pm10', 'no2', 'ozone', 'so2']:
            idw_result = calculate_raster(daily_data, parameter)
            out_file = os.path.join(output, f'idw_{parameter}_{dt_str}.tif')
            idw_result.rio.to_raster(out_file)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/dads'

    aq_d = os.path.join(root, 'aq')
    aq_csv_data = os.path.join(root, 'aq', 'joined_data')
    aq_summary = os.path.join(aq_d, 'aqs.csv')
    aq_nc = os.path.join(aq_d, 'netcdf', 'aqs_mt.nc')
    aq_tif = os.path.join(aq_d, 'tif')

    # csv_to_netcdf(aq_csv_data, aq_summary, aq_nc)

    write_rasters(aq_nc, aq_tif)

# ========================= EOF ====================================================================
