import os
import sys
import time

import ee
import pandas as pd
import geopandas as gpd

sys.path.insert(0, os.path.abspath('../..'))

from extract.rs.earth_engine.ee_utils import is_authorized, landsat_composites

sys.setrecursionlimit(2000)

BOUNDARIES = 'users/dgketchum/boundaries'


def request_band_extract(file_prefix, points_layer, years, buffer, tiles, check_dir=None, export_tif=False,
                         dry_run=False):
    """

    """
    tasks, outfile = None, None

    # if there are tasks in process:
    # earthengine task list | grep -E '(READY|COMPLETED)' | awk '{print $3}' > processing.txt

    processing = os.path.join(os.path.dirname(__file__), 'processing.txt')
    if os.path.exists(processing):
        with open(processing, 'r') as f:
            tasks = f.read().splitlines()

        tasks = [os.path.join(check_dir, '{}.csv'.format(t)) for t in tasks]

    mgrs = ee.FeatureCollection('users/dgketchum/boundaries/MGRS_TILE')
    points = ee.FeatureCollection(points_layer)
    points = points.map(lambda x: x.buffer(buffer))

    failed, to_export = {}, []

    for tile in tiles:

        clip = mgrs.filterMetadata('MGRS_TILE', 'equals', tile)

        for yr in years:
            desc = '{}_{}_{}'.format(file_prefix, yr, tile)
            if check_dir:
                outfile = os.path.join(check_dir, '{}.csv'.format(desc))
                if os.path.exists(outfile):
                    print('{} exists'.format(os.path.basename(outfile)))
                    continue
                elif tasks is not None:
                    if desc in tasks:
                        print('{} is processing'.format(os.path.basename(outfile)))
                        continue
                else:
                    pass

            if export_tif:
                stack = stack_bands(yr, scale=True)
                stack = stack.clip(clip.first().geometry().buffer(250))
                task = ee.batch.Export.image.toCloudStorage(
                    image=stack,
                    description=desc,
                    bucket='wudr',
                    fileNamePrefix=desc,
                    scale=250,
                    crs='EPSG:5071',
                    maxPixels=1e13)

                try:
                    task.start()
                    print(desc)
                except ee.ee_exception.EEException as e:
                    print('{}, waiting on '.format(e), desc, '......')
                    time.sleep(600)
                    task.start()
                    print(desc)

            elif dry_run:
                if outfile:
                    to_export.append(outfile)
                else:
                    desc = '{}_{}_{}'.format(file_prefix, yr, tile)
                    to_export.append(desc)

            else:
                stack = stack_bands(yr, scale=False)
                stack = stack.clip(clip.first().geometry().buffer(1000))
                tile_pts = points.filterMetadata('MGRS_TILE', 'equals', tile)

                data = stack.reduceRegions(collection=tile_pts,
                                           reducer=ee.Reducer.mean(),
                                           scale=30,
                                           tileScale=16)

                bucket_file = os.path.join('dads_landsat', desc)
                task = ee.batch.Export.table.toCloudStorage(
                    collection=data,
                    description=desc,
                    fileNamePrefix=bucket_file,
                    bucket='wudr',
                    fileFormat='CSV')

                try:
                    task.start()
                    print(desc, yr, flush=True)

                except ee.ee_exception.EEException as e:
                    print('{}, waiting on '.format(e), desc, '......')
                    time.sleep(600)
                    task.start()
                    print(desc)

                except Exception as e:
                    print(tile, yr, e)
                    if tile not in failed.keys():
                        failed[tile] = [str(yr)]
                    else:
                        failed[tile].append(str(yr))

    if dry_run:
        print(f'{len(to_export)} exports to be run')


def request_updates_from_shapefile(stations_shp, index_col, station_out_dir, file_prefix, points_layer,
                                   years, buffer, update_root, tiles=None, run_partial_tiles=False,
                                   run_empty_tiles=True, dry_run=False):
    stations = gpd.read_file(stations_shp)
    if tiles is None:
        tiles = [m for m in stations['MGRS_TILE'].unique().tolist() if isinstance(m, str)]
        tiles = sorted(list(set(tiles)))

    missing_tiles = []
    complete_tiles = []
    partial_tiles = []

    for t in tiles:
        sub = stations[stations['MGRS_TILE'] == t]
        fids = sub[index_col].astype(str).tolist()
        files = [os.path.join(station_out_dir, f'{fid}.csv') for fid in fids]

        if not any(os.path.exists(p) for p in files):
            missing_tiles.append(t)
        elif not all(os.path.exists(p) for p in files):
            partial_tiles.append(t)
        else:
            complete_tiles.append(t)

    # proceed if either missing or partial tiles exist; only return when neither needs work
    if not missing_tiles and not partial_tiles:
        print('No tiles need updates based on station files.')
        return

    if run_partial_tiles:
        run_tiles = partial_tiles
    elif run_empty_tiles:
        run_tiles = missing_tiles
    else:
        raise ValueError('Choose to run either missing tiles or partial tiles')

    print(f'Running Earth Engine extract on {len(run_tiles)} tiles')
    request_band_extract(file_prefix, points_layer, years, buffer, run_tiles, check_dir=update_root,
                         export_tif=False, dry_run=dry_run)


def request_updates_from_missing_list(sites_path, index_col, missing_csv, file_prefix, points_layer,
                                      buffer, update_root):
    stations = gpd.read_file(sites_path)
    stations[index_col] = stations[index_col].astype(str)
    missing = pd.read_csv(missing_csv)
    missing['station'] = missing['station'].astype(str)
    m = missing.merge(stations[[index_col, 'MGRS_TILE']], left_on='station', right_on=index_col, how='left')
    m = m.dropna(subset=['MGRS_TILE'])
    if m.empty:
        return
    groups = m.groupby('MGRS_TILE')['year'].apply(lambda s: sorted(list(set(s)))).to_dict()
    date_tag = pd.Timestamp.now().strftime('%Y%m%d')
    update_dir = os.path.join(update_root, date_tag)
    os.makedirs(update_dir, exist_ok=True)
    for tile, yrs in groups.items():
        request_band_extract(file_prefix, points_layer, yrs, buffer, [tile], check_dir=update_dir,
                             export_tif=False, dry_run=False)


def stack_bands(yr, scale=False):
    """
    Create a stack of bands for the year and region of interest specified.
    :param yr:
    :param southern
    :param roi:
    :return:
    """

    winter_s, winter_e = '{}-01-01'.format(yr), '{}-03-01'.format(yr),
    spring_s, spring_e = '{}-03-01'.format(yr), '{}-05-01'.format(yr),
    late_spring_s, late_spring_e = '{}-05-01'.format(yr), '{}-07-15'.format(yr)
    summer_s, summer_e = '{}-07-15'.format(yr), '{}-09-30'.format(yr)
    fall_s, fall_e = '{}-09-30'.format(yr), '{}-12-31'.format(yr)

    periods = [('0', winter_s, spring_s),
               ('1', spring_s, spring_e),
               ('2', late_spring_s, late_spring_e),
               ('3', summer_s, summer_e),
               ('4', fall_s, fall_e)]

    first, input_bands, proj = True, None, None
    for name, start, end in periods:
        bands = landsat_composites(yr, start, end, name, scale=scale)
        if first:
            input_bands = bands
            proj = bands.select('B2_0').projection().getInfo()
            first = False
        else:
            input_bands = input_bands.addBands(bands)

    input_bands = input_bands.reproject(crs=proj['crs'], scale=30)

    return input_bands


def shapely_to_ee_polygon(shapely_geom):
    geojson = shapely_geom.__geo_interface__
    return ee.Geometry.Polygon(geojson)


if __name__ == '__main__':
    is_authorized()

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    _bucket = 'gs://wudr'
    station_set = 'madis'

    if station_set == 'madis':
        index_ = 'fid'
        stations_glob = 'madis_17MAY2025_mgrs'
        sites_shp = os.path.join(d, 'dads', 'met', 'stations', f'{stations_glob}.shp')
        ee_points = 'projects/ee-dgketchum/assets/dads/madis_17MAY2025_mgrs'
        updates_root = os.path.join(d, 'dads', 'rs', 'landsat', 'updates', 'madis')

    elif station_set == 'ghcn':
        index_ = 'STAID'
        stations_glob = 'ghcn_CANUSA_stations_mgrs'
        sites_shp = os.path.join(d, 'climate', 'ghcn', 'stations', f'{stations_glob}.shp')
        ee_points = 'projects/ee-dgketchum/assets/dads/ghcn_CANUSA_stations_mgrs'
        updates_root = os.path.join(d, 'dads', 'rs', 'landsat', 'updates', 'ghcn')

    else:
        raise NotImplementedError

    # Use shapefile directly to determine tiles and missing stations
    stations = gpd.read_file(sites_shp)
    bounds = (-178., 7., -53., 83.)
    w, s, e, n = bounds
    stations = stations[(stations['latitude'] < n) & (stations['latitude'] >= s)]
    stations = stations[(stations['longitude'] < e) & (stations['longitude'] >= w)]
    tiles = [m for m in stations['MGRS_TILE'].unique().tolist() if isinstance(m, str)]
    mgrs_tiles = sorted(list(set(tiles)))

    station_files = os.path.join(d, 'dads', 'rs', 'landsat', 'station_data')

    failed = []
    missing_csv = os.path.join(d, 'dads', 'rs', 'landsat', 'missing_station_years.csv')
    run_missing = False
    years_ = [1987, 1988, 1989, 2023, 2024]

    request_band_extract(stations_glob, points_layer=ee_points, years=years_, buffer=500.0, tiles=mgrs_tiles,
                         check_dir=updates_root, export_tif=False, dry_run=False)

# ========================= EOF ====================================================================
