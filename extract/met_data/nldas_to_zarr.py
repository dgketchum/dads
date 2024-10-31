import os

import xarray as xr
from dask.distributed import Client


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

    nc_files = [os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith('.nc')]
    nc_files.sort()
    print(f'{len(nc_files)} files')
    print(f'first: {os.path.basename(nc_files[0])}')
    print(f'last: {os.path.basename(nc_files[-1])}')

    first_batch = nc_files[:batch_size]
    ds = xr.open_mfdataset(first_batch,  concat_dim='time', combine='nested',
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



def main():
    client = Client(n_workers=64, memory_limit='256GB')

    nc_dir = '/data/ssd1/nldas2'
    zarr_dir = '/data/ssd1/nldas2.zarr'

    convert_nc_to_zarr_in_batches(nc_dir, zarr_dir, batch_size=300)


if __name__ == '__main__':
    main()
