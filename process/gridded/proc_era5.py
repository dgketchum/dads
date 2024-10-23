import os
import math
import pandas as pd
import xarray as xr
import pyproj

from utils.station_parameters import station_par_map


def calculate_bearing(lat1, lon1, lat2, lon2):
    geod = pyproj.Geod(ellps='WGS84')

    fwd_azimuth, back_azimuth, distance = geod.inv(lon1, lat1, lon2, lat2)
    bearing_rad = math.radians(fwd_azimuth)

    return bearing_rad


def extract_cell_centroid_data(stations, gridded_dir, nc_file, overwrite=False,
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

    ds = xr.open_dataset(nc_file)

    for i, (fid, row) in enumerate(station_list.iterrows(), start=1):
        lon, lat, elv = row[kw['lon']], row[kw['lat']], row[kw['elev']]
        print('{}: {} of {}; {:.2f}, {:.2f}'.format(fid, i, record_ct, lat, lon))

        _file = os.path.join(gridded_dir, '{}_centroid.csv'.format(fid))
        if not os.path.exists(_file) or overwrite:
            # Find nearest cell
            ds_sel = ds.sel(longitude=lon, latitude=lat, method="nearest")

            # Extract cell centroid coordinates
            cell_lat = float(ds_sel.latitude.values)
            cell_lon = float(ds_sel.longitude.values)

            # Calculate bearing, distance, and relative elevation
            bearing = calculate_bearing(lat, lon, cell_lat, cell_lon)
            geod = pyproj.Geod(ellps='WGS84')

            _, _, distance = geod.inv(lon, lat, cell_lon, cell_lat)
            rel_elev = elv - 0  # Assuming elevation in NetCDF is 0 (check this!)

            # Create a DataFrame with the results
            df_centroid = pd.DataFrame({
                'cell_latitude': [cell_lat],
                'cell_longitude': [cell_lon],
                'bearing': [bearing],
                'distance': [distance],
                'rel_elev': [rel_elev]
            })

            # Save to CSV
            df_centroid.to_csv(_file, index=False)
            print('Centroid data extracted and saved to', _file)
        else:
            print('{} exists'.format(_file))

    ds.close()


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
