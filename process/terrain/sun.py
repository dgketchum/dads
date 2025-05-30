import os
import subprocess
from multiprocessing import Pool, cpu_count

import pandas as pd

from utils.station_parameters import station_par_map


def ingest_rasters(in_dir, tiles, mapset, overwrite=False):
    file_names = sorted(os.listdir(in_dir))
    dem_files = [os.path.join(in_dir, f) for f in file_names if f.endswith('.tif')]

    for in_dem in dem_files:

        tile = in_dem.split('.')[0][-5:]

        if tile not in tiles:
            continue

        dem_name = f'dem_{tile}'
        cmd = ['r.in.gdal', f'input={in_dem}', f'output={dem_name}@{mapset}']

        if overwrite:
            cmd += ['--overwrite']

        subprocess.call(cmd)


def worker_calculate_single_tile_irradiance(args):
    """"""
    dem_name_grass, dem_file_path, terrain_dir, mapset, r_sun_nprocs, overwrite = args

    tile = dem_name_grass.split('_')[-1]

    slp_dir = os.path.join(terrain_dir, 'slope')
    asp_dir = os.path.join(terrain_dir, 'aspect')

    slope_output_tif = os.path.join(slp_dir, f'slope_{tile}.tif')
    aspect_output_tif = os.path.join(asp_dir, f'aspect_{tile}.tif')

    slope_grass_name = f'slope_{tile}'
    aspect_grass_name = f'aspect_{tile}'

    slope_cmd = ['gdaldem', 'slope', dem_file_path, slope_output_tif]
    aspect_cmd = ['gdaldem', 'aspect', dem_file_path, aspect_output_tif]

    if overwrite:
        slope_cmd += ['--overwrite']
        aspect_cmd += ['--overwrite']

    subprocess.run(slope_cmd, check=True)
    subprocess.run(aspect_cmd, check=True)

    slope_ingest_cmd = ['r.in.gdal', f'input={slope_output_tif}', f'output={slope_grass_name}', '--overwrite']
    subprocess.run(slope_ingest_cmd, check=True)

    aspect_ingest_cmd = ['r.in.gdal', f'input={aspect_output_tif}', f'output={aspect_grass_name}', '--overwrite']
    subprocess.run(aspect_ingest_cmd, check=True)

    subprocess.run(['g.region', f'rast={dem_name_grass}@{mapset}'], check=True)

    for day in range(1, 366):
        irradiance_grass_name = f'irradiance_day_{day}_{tile}'

        rsun_command_list = ['r.sun',
                             f'elevation={dem_name_grass}@{mapset}',
                             f'slope={slope_grass_name}@{mapset}',
                             f'aspect={aspect_grass_name}@{mapset}',
                             f'day={day}',
                             f'glob_rad={irradiance_grass_name}',
                             f'nprocs={r_sun_nprocs}', '--overwrite']

        try:
            subprocess.run(rsun_command_list, check=True)
        except subprocess.CalledProcessError:
            print(f'{irradiance_grass_name} failed')
            continue

    return f"Tile {tile}: Done"


def calculate_terrain_irradiance_parallel(terrain_dir, terrain_source_path,
                                          mapset='PERMANENT', tiles_filter=None,
                                          overwrite=False,
                                          num_parallel_tiles=None, r_sun_nprocs_per_tile=1):
    """"""
    source_dem_tif_filenames = sorted([f for f in os.listdir(terrain_source_path) if f.endswith('.tif')])

    task_args_list = []
    for dem_tif_filename in source_dem_tif_filenames:
        dem_name_grass = dem_tif_filename.split('.')[0]
        tile_id = dem_name_grass.split('_')[-1]

        if tiles_filter and tile_id not in tiles_filter:
            continue

        if not overwrite:
            try:
                glist_result = subprocess.run(
                    ['g.list', 'type=raster', f'pattern=irradiance_day_*_{tile_id}@{mapset}'],
                    capture_output=True, text=True, check=True
                )
                existing_irradiance_files = [r for r in glist_result.stdout.strip().split('\n') if r]
                if len(existing_irradiance_files) >= 365:
                    print(f'{tile_id} is complete, skipping')
                    continue
                else:
                    print(f'{len(existing_irradiance_files)} files complete: adding {tile_id}')
            except subprocess.CalledProcessError as e:
                print(f'error {e} on {tile_id}')

        dem_file_full_path = os.path.join(terrain_source_path, dem_tif_filename)
        task_args_list.append((dem_name_grass, dem_file_full_path, terrain_dir,
                               mapset, r_sun_nprocs_per_tile, overwrite))

    if num_parallel_tiles is None:
        num_parallel_tiles = cpu_count()

    collected_results = []
    if task_args_list and num_parallel_tiles > 0:
        with Pool(processes=num_parallel_tiles) as p:
            for worker_result in p.imap_unordered(worker_calculate_single_tile_irradiance, task_args_list):
                collected_results.append(worker_result)

    return collected_results


def export_rasters(terrain_dir, rsun_out, mapset='PERMANENT', overwrite=False, mgrs_list=None):
    """"""
    dem_files = sorted(os.listdir(terrain_dir))
    dem_names = sorted([f.split('.')[0] for f in dem_files if f.endswith('.tif')])
    dem_files = [os.path.join(terrain_dir, f) for f in dem_files if f.endswith('.tif')]
    print(f'{len(dem_files)} dem files to export')

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


if __name__ == '__main__':

    root = '/home/dgketchum/data/IrrigationGIS'
    dem_d = '/data/ssd2/dads/dem'

    _bucket = 'gs://wudr'
    station_set = 'ghcn'
    zone = 'conus'

    if station_set == 'madis':
        stations = 'madis_17MAY2025_gap_mgrs'
        sites = os.path.join(root, 'dads', 'met', 'stations', f'{stations}.csv')
        chk = os.path.join(root, 'dads', 'rs', 'landsat', stations)

    elif station_set == 'ghcn':
        stations = 'ghcn_CANUSA_stations_mgrs'
        sites = os.path.join(root, 'climate', 'ghcn', 'stations', 'ghcn_stations_mgrs_country.csv')
        chk = os.path.join(root, 'dads', 'rs', 'ghcn_stations', 'landsat', 'tiles')

    else:
        raise ValueError

    if zone == 'canada':
        bounds = (-141., 49., -60., 85.)
        epsg = '3978'
        tif_dem = os.path.join(dem_d, f'dem_{epsg}')
        mapset_ = "dads_map_canada"

    elif zone == 'conus':
        bounds = (-180., 23., -60., 49.)
        epsg = '5071'
        tif_dem = os.path.join(dem_d, f'dem_{epsg}')
        mapset_ = "dads_map"

    elif zone == 'alaska':
        bounds = (-180., 49., -60., 85.)
        epsg = '6393'
        tif_dem = os.path.join(dem_d, f'dem_{epsg}')
        mapset_ = "dads_map_alaska"

    else:
        raise ValueError

    sites_df = pd.read_csv(sites)

    kw = station_par_map(station_set)

    sites_df = sites_df[(sites_df[kw['lat']] < bounds[3]) & (sites_df[kw['lat']] >= bounds[1])]
    sites_df = sites_df[(sites_df[kw['lon']] < bounds[2]) & (sites_df[kw['lon']] >= bounds[0])]

    if zone == 'canada':
        sites_df = sites_df[sites_df['AFF_ISO'] == 'CA']

    tiles = sites_df['MGRS_TILE'].unique().tolist()
    tiles = [m for m in tiles if isinstance(m, str)]
    mgrs_tiles = list(set(tiles))
    mgrs_tiles.sort()

    # ingest_rasters(tif_dem, mgrs_tiles, mapset=mapset_, overwrite=True)

    calculate_terrain_irradiance_parallel(dem_d, tif_dem, mapset=mapset_, tiles_filter=mgrs_tiles, overwrite=False,
                                          num_parallel_tiles=1, r_sun_nprocs_per_tile=4)

    # rsun_out_ = os.path.join(dem_d, 'rsun_irradiance')
    # export_rasters(tif_dem, rsun_out_, mapset=mapset_, mgrs_list=tiles, overwrite=True)

# ========================= EOF ====================================================================
