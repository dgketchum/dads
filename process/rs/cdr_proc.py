import os
import json
import pandas as pd
import xarray as xr
from datetime import datetime

from utils.station_parameters import station_par_map

import os
import pandas as pd
import xarray as xr
from datetime import datetime

from utils.station_parameters import station_par_map


def extract_surface_reflectance(stations, gridded_dir, incomplete_out, overwrite=False, bounds=None):
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
        for month in range(1, 13):
            if year == start.year and month < start.month:
                continue
            if year == end.year and month > end.month:
                break

            month_start = datetime(year, month, 1)
            month_end = (month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
            date_string = month_start.strftime('%Y%m')

            nc_files = [f for f in os.listdir(gridded_dir) if date_string in f]
            if not nc_files:
                print(f"No NetCDF files found for {year}-{month}")
                continue

            datasets = []
            complete = True
            for f in nc_files:
                try:
                    nc_file = os.path.join(gridded_dir, f)
                    ds = xr.open_dataset(nc_file, engine='netcdf4', decode_cf=False)
                    datasets.append(ds.sel(latitude=slice(n + 0.05, s - 0.05), longitude=slice(w - 0.05, e + 0.05)))
                except (AttributeError, OSError) as exc:
                    incomplete['missing'].append(f'{year}{month}')
                    complete = False
                    continue

            if not complete:
                print('failed to open {}-{}'.format(year, month))
                continue

            combined = xr.concat(datasets, dim='time')

            record_ct = station_list.shape[0]
            for i, (fid, row) in enumerate(station_list.iterrows(), start=1):
                lon, lat, elv = row['longitude', 'latitude', 'elevation']
                print('{}: {} of {}; {:.2f}, {:.2f}'.format(fid, i, record_ct, lat, lon))

                dst_dir = os.path.join(gridded_dir, 'cdr_raw', fid)
                if not os.path.exists(dst_dir):
                    os.mkdir(dst_dir)

                _file = os.path.join(dst_dir, '{}_{}.csv'.format(fid, year, month))

                if not os.path.exists(_file) or overwrite:
                    subset = combined.sel(longitude=lon, latitude=lat, method='nearest')
                    subset = subset.loc[dict(day=slice(month_start, month_end))]
                    date_ind = pd.date_range(month_start, month_end, freq='d')
                    subset['time'] = date_ind

                    time = subset['time'].values
                    series = subset[var_].values
                    df = pd.DataFrame(data=series, index=time)
                    df.columns = [var_]

                    df.to_csv(_file, mode='a', header=not os.path.exists(_file))
                    print('cdr', fid)

    if len(incomplete) > 0:
        with open(incomplete_out, 'w') as fp:
            json.dump(incomplete, fp, indent=4)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS')

    # pandarallel.initialize(nb_workers=6)

    madis_data_dir_ = os.path.join(d, 'climate', 'madis')
    sites = os.path.join(d, 'dads', 'met', 'stations', 'dads_stations_elev_mgrs.csv')

    grid_dir = os.path.join(d, 'dads', 'rs', 'cdr', 'nc')
    incomp = os.path.join(d, 'dads', 'rs', 'cdr', 'incomplete_files.json')

    extract_surface_reflectance(sites, grid_dir, incomp, overwrite=False, bounds=(-125.0, 25.0, -67.0, 53.0))

# ========================= EOF ====================================================================
