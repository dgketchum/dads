import os
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime
import numpy as np


def extract_modis_data(url="https://opendap.cr.usgs.gov/opendap/hyrax/MOD09GA.061/h00v08.ncml",
                       variable_name='variable_name',
                       start_date='2023-01-01',
                       end_date='2023-01-31',
                       lat_min=30, lat_max=40,
                       lon_min=-120, lon_max=-110):
    dataset = Dataset(url)
    time_var, lat_var, lon_var = dataset.variables['time'], dataset.variables['lat'], dataset.variables['lon']
    time_units, time_calendar, time_values = time_var.units, time_var.calendar if hasattr(time_var,
                                                                                          'calendar') else 'standard', time_var[
                                                                                                                       :]

    start_time_num = date2num(datetime.strptime(start_date, '%Y-%m-%d'), time_units, calendar=time_calendar)
    end_time_num = date2num(datetime.strptime(end_date, '%Y-%m-%d'), time_units, calendar=time_calendar)
    start_time_idx, end_time_idx = np.where(time_values >= start_time_num)[0][0], \
    np.where(time_values <= end_time_num)[0][-1]

    lat_indices = np.where((lat_var[:] >= lat_min) & (lat_var[:] <= lat_max))[0]
    lon_indices = np.where((lon_var[:] >= lon_min) & (lon_var[:] <= lon_max))[0]

    data = dataset.variables[variable_name][start_time_idx:end_time_idx + 1, lat_indices, lon_indices]
    dataset.close()
    return data


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
