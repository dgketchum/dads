import os
import subprocess

def calculate_terrain_irradiance(terrain_dir, mapset='PERMANENT', overwrite=False):
    s = os.path.join(terrain_dir, 'slope')
    a = os.path.join(terrain_dir, 'aspect')
    r = os.path.join(terrain_dir, 'rsun')
    d = os.path.join(terrain_dir, 'proj')

    dem_files = sorted(os.listdir(d))
    dem_names = sorted([f.split('.')[0] for f in dem_files if f.endswith('.tif')])
    dem_files = [os.path.join(d, f) for f in dem_files]

    for dem_name, dem_file in zip(dem_names, dem_files):
        tile = dem_name.split('_')[-1]

        if tile != '12TTS':
            continue

        tile_dir = os.path.join(r, tile)
        if not os.path.isdir(tile_dir):
            os.mkdir(tile_dir)

        slope_output_path = os.path.join(s, 'slope_{0}.tif'.format(tile))
        if os.path.exists(slope_output_path) and not overwrite:
            print(slope_output_path, 'exists, skipping')
        else:
            subprocess.call(['gdaldem', 'slope', f'{dem_file}', slope_output_path])
            slope_name = f'slope_{tile}'
            subprocess.call(['r.in.gdal', f'input={slope_output_path}', f'output={slope_name}'])

        aspect_output_path = os.path.join(a, 'aspect_{0}.tif'.format(tile))
        if os.path.exists(aspect_output_path) and not overwrite:
            print(aspect_output_path, 'exists, skipping')
        else:
            subprocess.call(['gdaldem', 'aspect', f'{dem_file}', aspect_output_path])
            aspect_name = f'aspect_{tile}'
            subprocess.call(['r.in.gdal', f'input={aspect_output_path}', f'output={aspect_name}'])

        subprocess.call(['g.region', f'rast=dem_{tile}@{mapset}'])
        for day in range(1, 366):
            irradiance_output_path = 'irradiance_day_{0}_{1}'.format(day, tile)
            if tile == '12TTS' and day in [1, 173, 304, 356]:
                subprocess.call(
                    ['r.sun',
                     'elevation={0}@{1}'.format(dem_name, mapset),
                     'slope={0}@{1}'.format(slope_name, mapset),
                     'aspect={0}@{1}'.format(aspect_name, mapset),
                     'day={0}'.format(day),
                     'glob_rad={0}@{1}'.format(irradiance_output_path, mapset),
                     '--overwrite',
                     ]
                )

                irradiance_output_tif = os.path.join(r, tile, 'irradiance_day_{0}_{1}.tif'.format(day, tile))
                subprocess.call(
                    ['r.out.gdal', '-c',
                     'input={0}@{1}'.format(irradiance_output_path, mapset),
                     'format=GTiff',
                     'createopt=COMPRESS=LZW',
                     '--overwrite',
                     'output={0}'.format(irradiance_output_tif)])
                print(day)

        break


if __name__ == '__main__':
    root = '/media/nvm/IrrigationGIS/dads'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/dads'

    dem_d = os.path.join(root, 'dem')
    calculate_terrain_irradiance(dem_d, mapset="dads_map", overwrite=True)

# ========================= EOF ====================================================================
