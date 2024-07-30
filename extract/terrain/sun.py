import os
import subprocess

def calculate_terrain_irradiance(terrain_dir):

    s = os.path.join(terrain_dir, 'slope')
    a = os.path.join(terrain_dir, 'aspect')
    r = os.path.join(terrain_dir, 'rsun')
    d = os.path.join(terrain_dir, 'proj')

    dem_files = sorted(os.listdir(d))

    for dem_file_path in dem_files:

        tile = dem_file_path.split('.')[0].split('_')[-1]
        tile_dir = os.path.join(r, tile)
        if not os.path.isdir(tile_dir):
            os.mkdir(tile_dir)

        slope_output_path = os.path.join(s, 'slope_{0}.tif'.format(tile))
        print slope_output_path
        subprocess.call(['gdaldem', 'slope', os.path.join(d, dem_file_path), slope_output_path])

        aspect_output_path = os.path.join(a, 'aspect_{0}.tif'.format(tile))
        print aspect_output_path
        subprocess.call(['gdaldem', 'aspect', os.path.join(d, dem_file_path), aspect_output_path])

        for day in range(1, 366):
            irradiance_output_path = os.path.join(tile_dir, 'irradiance_day_{0}'.format(day))
            print irradiance_output_path
            subprocess.call(
                ['r.sun', 'elevation={0}'.format(os.path.join(d, dem_file_path)), 'day={0}'.format(day),
                 'glob_rad={0}'.format(irradiance_output_path)])

        break

if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/dads'
    dem_d = os.path.join(root, 'dem')
    calculate_terrain_irradiance(dem_d)

# ========================= EOF ====================================================================
