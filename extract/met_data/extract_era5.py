import os
import os
import pandas as pd
import xarray as xr

from utils.station_parameters import station_par_map

def extract_met_data(stations, gridded_dir, nc_file, overwrite=False,
                     station_type='dads', shuffle=True, bounds=None):

    kw = station_par_map(station_type)

    station_list = pd.read_csv(stations, index_col='index')

    if shuffle:
        station_list = station_list.sample(frac=1)

    if bounds:
        w, s, e, n = bounds
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]
    else:
        # Use the bounds from the NetCDF file
        ds = xr.open_dataset(nc_file)
        w = float(ds.longitude[0].values)
        e = float(ds.longitude[-1].values)
        s = float(ds.latitude[-1].values)  # Latitude is decreasing
        n = float(ds.latitude[0].values)
        ds.close()
        ln = station_list.shape[0]
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]
        print('dropped {} stations outside the NetCDF extent'.format(ln - station_list.shape[0]))

    record_ct = station_list.shape[0]

    # Open the NetCDF dataset
    ds = xr.open_dataset(nc_file)

    for i, (fid, row) in enumerate(station_list.iterrows(), start=1):
        lon, lat, elv = row[kw['lon']], row[kw['lat']], row[kw['elev']]
        print('{}: {} of {}; {:.2f}, {:.2f}'.format(fid, i, record_ct, lat, lon))

        _file = os.path.join(gridded_dir, '{}.csv'.format(fid))
        if not os.path.exists(_file) or overwrite:
            # Extract data for the nearest grid cell
            df = ds.sel(longitude=lon, latitude=lat, method="nearest").to_dataframe()

            # Remove unnecessary columns and rename for consistency
            df = df[['u10', 'v10', 'd2m', 't2m', 'sp', 'tp', 'ssr']]
            df = df.rename(columns={
                'u10': 'wind_u',
                'v10': 'wind_v',
                'd2m': 'tdew',
                't2m': 'temp',
                'sp': 'psurf',
                'tp': 'precip',
                'ssr': 'rsds'
            })

            # Convert temperature to Celsius
            df['temp'] -= 273.15
            df['tdew'] -= 273.15

            # Save to CSV
            df.to_csv(_file)
            print('Data extracted and saved to', _file)
        else:
            print('{} exists'.format(_file))

    ds.close()


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
