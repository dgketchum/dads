import calendar
import os

import cdsapi
import pandas as pd
import xarray as xr

from utils.station_parameters import station_par_map


def download_era5(target_dir):
    dataset = "reanalysis-era5-land"
    request = {
        "variable": [
            "2m_dewpoint_temperature",
            "2m_temperature",
            "surface_solar_radiation_downwards",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure",
            "total_precipitation"
        ],
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [53, -125, 25, -67]
    }

    client = cdsapi.Client()

    for year in range(1990, 20244):
        for month in range(1, 13):
            _, num_days = calendar.monthrange(year, month)
            request["day"] = [str(day) for day in range(1, num_days + 1)]
            request["year"] = str(year)
            request["month"] = f"{month:02d}"
            target_nc = os.path.join(target_dir, f"era5_land_{year}_{month:02d}.nc")
            client.retrieve(dataset, request, target_nc)
            print(f'downloaded {target_nc}')


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

    d = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS')

    if not os.path.isdir(d):
        d = os.path.join('/data', 'era5', 'netcdf')

    nc_dir = os.path.join(d)

    download_era5(nc_dir)

# ========================= EOF ====================================================================
