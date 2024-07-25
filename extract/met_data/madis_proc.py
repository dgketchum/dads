import os
import gzip
import tempfile
import xarray as xr


def process_single_file(filename, required_vars, chunk_size=10000):
    all_chunks = []
    # Decompress the gzipped file into a temporary file
    with gzip.open(filename, 'rb') as f_in:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(f_in.read())
            tmp_file_path = tmp_file.name

    # Now read the temporary file with xarray
    try:
        with xr.open_dataset(tmp_file_path, engine="netcdf4") as ds:
            # Process the data in chunks
            num_records = ds.dims['recNum']
            for i in range(0, num_records, chunk_size):
                # Select a chunk of the data
                chunk = ds.isel(recNum=slice(i, i + chunk_size))

                # Check for non-null values for each variable in required_vars
                valid_data = chunk[required_vars[0]].notnull()
                for var in required_vars[1:]:
                    valid_data = valid_data & chunk[var].notnull()

                # Filter dataset to only include records with non-null values for all required variables
                filtered_chunk = chunk.where(valid_data, drop=True)
                all_chunks.append(filtered_chunk)

        # Combine all filtered chunks along the 'recNum' dimension
        combined_data = xr.concat(all_chunks, dim="recNum")
    finally:
        os.remove(tmp_file_path)  # Clean up the temporary file
    return combined_data


def read_madis_hourly(data_directory, date, chunk_size=10000):
    # Get list of files that match the date
    file_list = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if date in f]

    if not file_list:
        print(f"No files found for date: {date}")
        return

    # List of required variables
    required_vars = ['relHumidity', 'precipAccum', 'solarRadiation', 'stationId', 'temperature', 'windSpeed']

    for filename in file_list:
        print(f"Processing file: {filename}")
        combined_data = process_single_file(filename, required_vars, chunk_size)
        print(combined_data)

        # Save or process the combined_data as needed
        # For example, save to a new file:
        output_filename = filename.replace(".nc.gz", "_filtered.nc")
        combined_data.to_netcdf(output_filename)

        print(f"Saved filtered data to: {output_filename}")


if __name__ == "__main__":
    madis_data_dir = '/home/dgketchum/data/IrrigationGIS/climate/madis/LDAD/mesonet/netCDF'
    read_madis_hourly(madis_data_dir, '20191001')

# ========================= EOF ====================================================================
