import os

import xarray as xr
import apache_beam as beam
import pandas as pd
import zarr
from dask.distributed import Client
from pangeo_forge_recipes import aggregation, dynamic_target_chunks


def process_batch_to_zarr(nc_files_batch, zarr_dir):
    try:
        ds = xr.open_mfdataset(nc_files_batch, concat_dim='time', combine='nested',
                               data_vars='minimal', coords='minimal', compat='override')
        ds.to_zarr(zarr_dir, mode='a', consolidated=True, append_dim='time')
        return True
    except Exception as e:
        print(e)
        return False


def convert_nc_to_zarr_in_batches(nc_dir, zarr_dir, batch_size=1000):
    nc_files = [os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith('.nc') and '200503' in f]
    nc_files.sort()
    print(f'{len(nc_files)} files')
    print(f'first: {os.path.basename(nc_files[0])}')
    print(f'last: {os.path.basename(nc_files[-1])}')

    first_batch = nc_files[:batch_size]
    ds = xr.open_mfdataset(first_batch, concat_dim='time', combine='nested',
                           data_vars='minimal', coords='minimal', compat='override')

    ds.to_zarr(zarr_dir, mode='w', consolidated=True)
    del ds

    for i in range(batch_size, len(nc_files), batch_size):
        batch = nc_files[i:i + batch_size]
        res = process_batch_to_zarr(batch, zarr_dir)
        if res:
            print(f'Processed batch {i // batch_size + 1}/{len(nc_files) // batch_size + 1}')
        else:
            print(f'Failed on batch {i // batch_size + 1}/{len(nc_files) // batch_size + 1}')

def chunk_for_time(nc_dir, zarr_dir):

    nc_files = [os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith('.nc') and '200503' in f]
    nc_files.sort()

    print(f'{len(nc_files)} files')
    print(f'first: {os.path.basename(nc_files[0])}')
    print(f'last: {os.path.basename(nc_files[-1])}')

    ds = xr.open_mfdataset(nc_files, chunks=False)
    d = ds.to_dict(data=False)
    schema = aggregation.XarraySchema(
        attrs=d.get('attrs'),
        coords=d.get('coords'),
        data_vars=d.get('data_vars'),
        dims=d.get('dims'),
        chunks=d.get('chunks', {}),
    )
    target_chunks = dynamic_target_chunks.dynamic_target_chunks_from_schema(
        schema,
        target_chunk_size='100MB',
        target_chunks_aspect_ratio={'time': -1, 'lat': 1, 'lon': 1},
        size_tolerance=0.5
    )

    ds = xr.open_mfdataset(nc_files, combine='by_coords', chunks=target_chunks)
    ds.to_zarr(zarr_dir)
    xr.open_zarr(zarr_dir)

def main():
    # client = Client(n_workers=64, memory_limit='256GB')

    nc_dir = '/data/ssd1/nldas2'
    zarr_dir = '/data/ssd1/nldas2_ts.zarr'

    # convert_nc_to_zarr_in_batches(nc_dir, zarr_dir, batch_size=300)
    chunk_for_time(nc_dir, zarr_dir)


if __name__ == '__main__':
    main()
