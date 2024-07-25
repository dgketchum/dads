import os
import gzip
import tempfile
import xarray as xr

import xarray as xr
import pandas as pd
import glob
import os
import gzip
import tempfile
import pandas as pd
from pathlib import Path
import numpy as np


def process_single_file(filename, required_vars, output_directory, chunk_size=5000):
    """Process a single gzipped NetCDF file, filter by variables,
       and write to a CSV in chunks."""

    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Decompress the gzipped file into a temporary file
    with gzip.open(filename, 'rb') as f_in:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(f_in.read())
            tmp_file_path = tmp_file.name

    try:
        with xr.open_dataset(tmp_file_path, engine="netcdf4") as ds:
            station_dim = 'recNum'
            num_records = ds.dims[station_dim]
            first = True
            for i in range(0, num_records, chunk_size):
                chunk = ds.isel({station_dim: slice(i, i + chunk_size)})
                valid_data = chunk[required_vars[0]].notnull()
                for var in required_vars[1:]:
                    valid_data = valid_data & chunk[var].notnull()
                filtered_chunk = chunk.where(valid_data, drop=True)
                data_array = filtered_chunk[required_vars].to_array().values.T
                c = pd.DataFrame(data_array, columns=required_vars)
                c['stationId'] = c['stationId'].astype(str)
                c.index = c['stationId']
                if first:
                    df = c
                    first = False
                else:
                    df = pd.concat([df, c], axis=0)

    finally:
        os.remove(tmp_file_path)

    return df


def read_madis_hourly(data_directory, date, output_directory):
    file_pattern = os.path.join(data_directory, f"*{date}*.gz")
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        print(f"No files found for date: {date}")
        return
    required_vars = ['relHumidity', 'precipAccum', 'solarRadiation', 'stationId', 'temperature', 'windSpeed']
    for filename in file_list:
        process_single_file(filename, required_vars, output_directory)  # Call the new function


if __name__ == "__main__":
    madis_data_dir = '/home/dgketchum/data/IrrigationGIS/climate/madis/LDAD/mesonet/netCDF'
    out_dir = '/home/dgketchum/data/IrrigationGIS/climate/madis/LDAD/mesonet/csv'
    read_madis_hourly(madis_data_dir, '20191001', out_dir)

# ========================= EOF ====================================================================
