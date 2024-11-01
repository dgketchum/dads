import os
import time
import subprocess
from pprint import pprint

import pandas as pd


def ingest_rasters(in_dir, tiles, mapset):
    file_names = sorted(os.listdir(in_dir))
    dem_files = [os.path.join(in_dir, f) for f in file_names if f.endswith('.tif')]

    for in_dem in dem_files:

        tile = in_dem.split('.')[0][-5:]

        if tile not in tiles:
            continue

        dem_name = f'dem_{tile}'
        subprocess.call(['r.in.gdal', f'input={in_dem}', f'output={dem_name}@{mapset}', '--overwrite'])


def calculate_terrain_irradiance(terrain_dir, terrain_source, mapset='PERMANENT', tiles=None, overwrite=False):
    """
    This must be run from a GRASS command line with the Location appropriate to the tile placement

    For a new region, create a new location with an appropriate projection, e.g.,:
    grass -c EPSG:3978 /media/nvm/IrrigationGIS/dads/dem/grassdata/canada
    g.mapset -c mapset=dads_map_canada

    The dem_{tile}.tif data also needs to be ingested into the mapset:

    for file in /media/nvm/IrrigationGIS/dads/dem/dem_5071/*.tif; do
        r.in.gdal input="$file" output="$(basename "$file" .tif)"
    done


    """
    slp = os.path.join(terrain_dir, 'slope')
    asp = os.path.join(terrain_dir, 'aspect')
    d = os.path.join(terrain_dir, terrain_source)

    dem_files = sorted(os.listdir(d))
    dem_names = sorted([f.split('.')[0] for f in dem_files if f.endswith('.tif')])
    dem_files = [os.path.join(d, f) for f in dem_files]
    print(f'{len(dem_files)} dem files to process')

    for dem_name, dem_file in zip(dem_names, dem_files):

        tile = dem_name.split('_')[-1]
        # print('\n', tile)

        if tiles:
            if tile not in tiles:
                continue

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
        subprocess.call(['r.in.gdal', f'input={slope_output_path}', f'output={slope_name}', '--overwrite'])

        aspect_output_path = os.path.join(asp, 'aspect_{0}.tif'.format(tile))
        subprocess.call(['gdaldem', 'aspect', f'{dem_file}', aspect_output_path])
        aspect_name = f'aspect_{tile}'
        subprocess.call(['r.in.gdal', f'input={aspect_output_path}', f'output={aspect_name}', '--overwrite'])

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
        # print(tile, '\n')


def export_rasters(terrain_dir, out_dir, mapset='PERMANENT', overwrite=False, mgrs_list=None):
    """"""
    dem_files = sorted(os.listdir(terrain_dir))
    dem_names = sorted([f.split('.')[0] for f in dem_files if f.endswith('.tif')])
    dem_files = [os.path.join(terrain_dir, f) for f in dem_files if f.endswith('.tif')]
    print(f'{len(dem_files)} dem files to export')

    rsun_out = os.path.join(out_dir, 'rsun')

    for dem_name, dem_file in zip(dem_names, dem_files):

        tile = dem_name.split('_')[-1]
        if mgrs_list and tile not in mgrs_list:
            print(f'{tile} exists but is not in list')
            continue

        # print('\n', tile)

        tile_dir = os.path.join(rsun_out, tile)
        if not os.path.isdir(tile_dir):
            os.mkdir(tile_dir)

        subprocess.call(['g.region', f'rast=dem_{tile}@{mapset}'])

        first = True
        for day in range(1, 366):

            irradiance_output_tif = os.path.join(tile_dir, 'irradiance_day_{0}_{1}.tif'.format(day, tile))
            irradiance_input = 'irradiance_day_{0}_{1}'.format(day, tile)

            if os.path.exists(irradiance_output_tif) and not overwrite:
                if first:
                    print(irradiance_output_tif, 'exists')
                    first = False
                continue

            subprocess.call(
                ['r.out.gdal', '-c',
                 'input={0}@{1}'.format(irradiance_input, mapset),
                 'format=GTiff',
                 'createopt=COMPRESS=LZW',
                 '--overwrite',
                 'output={0}'.format(irradiance_output_tif)])

            # print(tile, day)


def reproject_dems(in_dir, tiles, output_dir):
    file_names = sorted(os.listdir(in_dir))
    dem_files = [os.path.join(in_dir, f) for f in file_names if f.endswith('.tif')]
    out_files = [os.path.join(output_dir, f) for f in file_names if f.endswith('.tif')]

    for in_dem, out_dem in zip(dem_files, out_files):
        tile = in_dem.split('.')[0][-5:]

        if tile not in tiles:
            continue
        subprocess.run([
            "gdalwarp", "-s_srs", "EPSG:5071", '-overwrite',
            "-t_srs", "EPSG:3978", "-r", "bilinear",
            "-of", "GTiff", in_dem, out_dem
        ])
        print(tile, os.path.basename(in_dem), os.path.basename(out_dem))


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
    root = '/media/nvm/IrrigationGIS'
    out = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'
        out = '/home/dgketchum/data/IrrigationGIS'

    dem_d = os.path.join(root, 'dads', 'dem')
    out_dem = os.path.join(out, 'dads', 'dem')

    tiles_ = os.path.join(out, 'boundaries', 'mgrs', 'mgrs_nldas_canada.csv')
    tiles_ = pd.read_csv(tiles_)['MGRS_TILE'].unique().tolist()

    conus_dir_ = os.path.join(dem_d, 'dem_5071')
    canada_dir_ = os.path.join(dem_d, 'dem_3978')
    # reproject_dems(conus_dir_, tiles_, canada_dir_)

    ingest_rasters(canada_dir_, tiles_, mapset="dads_map_canada")

    calculate_terrain_irradiance(dem_d, canada_dir_, mapset="dads_map_canada", tiles=tiles_, overwrite=True)

    # stations_out = os.path.join(out, 'met', 'stations', 'dads_stations_res_elev_mgrs.csv')
    # stations_out = os.path.join(out, 'met', 'stations', 'madis_mgrs_28OCT2024.csv')

    export_rasters(canada_dir_, out_dem, mapset="dads_map_canada", mgrs_list=tiles_, overwrite=True)

    # remove_rasters(dem_d, resolution=30)

# ========================= EOF ====================================================================
