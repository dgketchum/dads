import os
import time
import subprocess
from pprint import pprint

import pandas as pd


def calculate_terrain_irradiance(terrain_dir, mapset='PERMANENT', overwrite=False):
    slp = os.path.join(terrain_dir, 'slope')
    asp = os.path.join(terrain_dir, 'aspect')
    d = os.path.join(terrain_dir, 'proj')

    dem_files = sorted(os.listdir(d))
    dem_names = sorted([f.split('.')[0] for f in dem_files if f.endswith('.tif')])
    dem_files = [os.path.join(d, f) for f in dem_files]

    for dem_name, dem_file in zip(dem_names, dem_files):

        tile = dem_name.split('_')[-1]
        print('\n', tile)

        result = subprocess.run(['g.list', f'type=raster', f'pattern=irradiance_day_*_{tile}'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            raster_list = result.stdout.strip().split('\n')
        else:
            print("Error:", result.stderr)
            continue

        if len(raster_list) >= 365 and not overwrite:
            print(tile, 'is processed, skipping')
            continue

        slope_output_path = os.path.join(slp, 'slope_{0}.tif'.format(tile))
        subprocess.call(['gdaldem', 'slope', f'{dem_file}', slope_output_path])
        slope_name = f'slope_{tile}'
        subprocess.call(['r.in.gdal', f'input={slope_output_path}', f'output={slope_name}'])

        aspect_output_path = os.path.join(asp, 'aspect_{0}.tif'.format(tile))
        subprocess.call(['gdaldem', 'aspect', f'{dem_file}', aspect_output_path])
        aspect_name = f'aspect_{tile}'
        subprocess.call(['r.in.gdal', f'input={aspect_output_path}', f'output={aspect_name}'])

        subprocess.call(['g.region', f'rast=dem_{tile}@{mapset}'])

        for day in range(1, 366):
            start = time.time()
            irradiance_output_path = 'irradiance_day_{0}_{1}'.format(day, tile)
            print('processing', irradiance_output_path)
            subprocess.call(['r.sun',
                             'elevation={0}@{1}'.format(dem_name, mapset),
                             'slope={0}@{1}'.format(slope_name, mapset),
                             'aspect={0}@{1}'.format(aspect_name, mapset),
                             'day={0}'.format(day),
                             'glob_rad={0}@{1}'.format(irradiance_output_path, mapset),
                             '--overwrite',
                             'nprocs=6'])

            dif = time.time() - start
            print('took {:.2f} seconds'.format(dif))
        print(tile, '\n')


def export_rasters(terrain_dir, out_dir, mapset='PERMANENT', overwrite=False, mgrs_list=None):

    if mgrs_list:
        mgrs_list = pd.read_csv(mgrs_list)['MGRS_TILE'].tolist()

    d = os.path.join(terrain_dir, 'proj')

    dem_files = sorted(os.listdir(d))
    dem_names = sorted([f.split('.')[0] for f in dem_files if f.endswith('.tif')])
    dem_files = [os.path.join(d, f) for f in dem_files]

    rsun_out = os.path.join(out_dir, 'rsun')

    for dem_name, dem_file in zip(dem_names, dem_files):

        tile = dem_name.split('_')[-1]
        if mgrs_list and tile not in mgrs_list:
            continue

        print('\n', tile)

        tile_dir = os.path.join(rsun_out, tile)
        if not os.path.isdir(tile_dir):
            os.mkdir(tile_dir)

        for day in range(1, 366):

            irradiance_output_tif = os.path.join(tile_dir, 'irradiance_day_{0}_{1}.tif'.format(day, tile))
            irradiance_input = 'irradiance_day_{0}_{1}'.format(day, tile)

            if os.path.exists(irradiance_output_tif) and not overwrite:
                print(irradiance_output_tif, 'exists')
                continue

            subprocess.call(
                ['r.out.gdal', '-c',
                 'input={0}@{1}'.format(irradiance_input, mapset),
                 'format=GTiff',
                 'createopt=COMPRESS=LZW',
                 '--overwrite',
                 'output={0}'.format(irradiance_output_tif)])

            print(tile, day)


def remove_rasters(terrain_dir, resolution=300):
    d = os.path.join(terrain_dir, 'proj_{}'.format(resolution))

    dem_files = sorted(os.listdir(d))
    dem_names = sorted([f.split('.')[0] for f in dem_files if f.endswith('.tif')])
    dem_files = [os.path.join(d, f) for f in dem_files]

    for dem_name, dem_file in zip(dem_names, dem_files):
        tile = dem_name.split('_')[-1]
        subprocess.call(['g.remove', 'type=raster', f'name=dem_{tile}', '-f'])
        subprocess.call(['g.remove', 'type=raster', f'name=slope_{tile}', '-f'])
        subprocess.call(['g.remove', 'type=raster', f'name=aspect_{tile}', '-f'])
        subprocess.call(['g.remove', 'type=raster', f'name=aspect_{tile}', '-f'])

        for day in range(1, 366):
            irradiance_output_path = 'irradiance_day_{0}_{1}'.format(day, tile)
            subprocess.call(['g.remove', 'type=raster', f'name={irradiance_output_path}', '-f'])

        print('removed {}'.format(tile))


if __name__ == '__main__':
    root = '/media/nvm/IrrigationGIS/dads'
    out = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/dads'
        out = '/home/dgketchum/data/IrrigationGIS/dads'

    dem_d = os.path.join(root, 'dem')
    out_dem = os.path.join(out, 'dem')
    mgrs = os.path.join(out, 'training', 'w17_tiles.csv')

    calculate_terrain_irradiance(dem_d, mapset="dads_map", overwrite=False)
    # export_rasters(dem_d, out_dem, mapset="dads_map", overwrite=False, mgrs_list=None)

    # remove_rasters(dem_d, resolution=30)

# ========================= EOF ====================================================================
