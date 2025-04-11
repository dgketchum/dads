import calendar
import os
import zipfile
import io

import cdsapi
import pandas as pd
import xarray as xr

from utils.station_parameters import station_par_map


def download_era5(target_dir, overwrite=False):
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

    for year in range(2000, 2025):
        for month in range(1, 13):
            _, num_days = calendar.monthrange(year, month)
            request["day"] = [str(day) for day in range(1, num_days + 1)]
            request["year"] = str(year)
            request["month"] = f"{month:02d}"
            target_nc = os.path.join(target_dir, f"era5_land_{year}_{month:02d}.nc")
            if not overwrite and os.path.exists(target_nc):
                print(f'{os.path.basename(target_nc)} exists, skippping')
                continue
            client.retrieve(dataset, request, target_nc)
            print(f'downloaded {target_nc}')


def extract_met_data(stations, gridded_dir, nc_dir, overwrite=False,
                     station_type='dads', shuffle=True, bounds=None):
    """"""
    kw = station_par_map(station_type)

    station_list = pd.read_csv(stations, index_col=kw['index'])

    nc_files = sorted([os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if 'era5_land_' in f])

    if shuffle:
        station_list = station_list.sample(frac=1)

    if bounds:
        w, s, e, n = bounds
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]
    else:
        # Use the bounds from the NetCDF file
        ds = xr.open_dataset(nc_files[0])
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

    for nc_file in nc_files:

        ds = xr.open_dataset(nc_file)
        splt = os.path.basename(nc_file).strip('.nc').split('_')
        year, month = splt[-2], splt[-1]

        for i, (fid, row) in enumerate(station_list.iterrows(), start=1):
            lon, lat, elv = row[kw['lon']], row[kw['lat']], row[kw['elev']]
            print('{}: {} of {}; {:.2f}, {:.2f}'.format(fid, i, record_ct, lat, lon))

            sub_dir = os.path.join(gridded_dir, fid)
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)

            _file = os.path.join(sub_dir, f'{fid}_{year}_{month}.parquet')
            if not os.path.exists(_file) or overwrite:
                # Extract data for the nearest grid cell
                df = ds.sel(longitude=lon, latitude=lat, method="nearest").to_dataframe()

                # Remove unnecessary columns and rename for consistency
                print(ds.variables)
                df = df[['u10', 'v10', 'd2m', 't2m', 'sp', 'tp', 'ssrd', 'latitude', 'longitude']]
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

                df['slat'] = lat
                df['slon'] = lon

                # Save to CSV
                df.to_parquet(_file)
                print('Data extracted and saved to', _file)
            else:
                print('{} exists'.format(_file))

        ds.close()


if __name__ == '__main__':

    home = os.path.expanduser('~')

    d = os.path.join('/data/ssd2/dads')
    era5 = os.path.join(d, 'era5_land')

    if not os.path.isdir(d):
        d = os.path.join(home, 'data', 'IrrigationGIS')
        era5 = os.path.join(d, 'climate', 'era5')

    nc_dir_ = os.path.join(era5, 'netCDF')
    out_files = os.path.join(era5, 'raw_parquet')

    # download_era5(nc_dir_, overwrite=False)

    dads = os.path.join(home, 'data', 'IrrigationGIS', 'dads')
    climate = os.path.join(home, 'data', 'IrrigationGIS', 'climate')
    if not os.path.exists(dads):
        dads = os.path.join('/media/research', 'IrrigationGIS', 'dads')
        climate = os.path.join('/media/research', 'IrrigationGIS', 'climate')

    # sites = os.path.join(climate, 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    # stype = 'ghcn'

    sites = os.path.join(dads, 'met', 'stations', 'dads_stations_10FEB2025.csv')
    stype = 'dads'

    extract_met_data(sites, out_files, nc_dir=nc_dir_, bounds=None, overwrite=True, station_type=stype)

# ========================= EOF ====================================================================
