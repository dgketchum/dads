import os
import shutil
import json
import multiprocessing

import geopandas as gpd


def organize_existing_samples(stations, csv_dir, output_dir):

    gdf = gpd.read_file(stations)
    gdf.index = gdf['fid']
    train_split = gdf[['train']]

    files_ = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    names = [f.split('.')[0] for f in files_]

    destiny, stations = [], []
    for s in names:
        try:
            train_status = train_split.loc[s, 'train']
        except KeyError:
            continue

        if train_status:
            destiny.append('train')
            stations.append(s)
        else:
            destiny.append('val')
            stations.append(s)

    for j, (fate, station) in enumerate(zip(destiny, stations), start=1):

        v_file = os.path.join(output_dir, 'val', '{}.pth'.format(station))
        t_file = os.path.join(output_dir, 'train', '{}.pth'.format(station))

        if fate == 'train' and os.path.exists(v_file):
            shutil.move(v_file, t_file)
            print(f'moved {os.path.basename(v_file)} from val to train')

        elif fate == 'val' and os.path.exists(t_file):
            shutil.move(t_file, v_file)
            print(f'moved {os.path.basename(t_file)} from train to val')

        else:
            pass


def copy_file(source_file, dest_file):
    if os.path.exists(dest_file):
        return
    try:
        shutil.copyfile(source_file, dest_file)
        if dest_file.endswith('202312.csv'):
            print(dest_file)
    except Exception as e:
        print(e, os.path.basename(source_file))


def transfer_list(src, dst, workers=2):

    sources = []
    targets = []

    dirs = os.listdir(src)
    print(f'{len(dirs)} directories to copy')

    for i, d in enumerate(dirs, start=1):

        dst_dir = os.path.join(dst, d)
        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)

        src_dir = os.path.join(src, d)
        src_filenames = [f for f in os.listdir(src_dir)]
        dst_filenames = [f for f in os.listdir(dst_dir)]

        sources.extend([os.path.join(src_dir, f) for f in src_filenames if f not in dst_filenames])
        targets.extend([os.path.join(dst_dir, f) for f in src_filenames if f not in dst_filenames])

        if i % 100 == 0:
            print(f'\n{i}')
            print(len(sources))
            print(sources[-1])
            print(len(targets))
            print(targets[-1])

        if len(sources) > 100000:

            with multiprocessing.Pool(processes=workers) as pool:
                pool.starmap(copy_file, zip(sources, targets))

            sources = []
            targets = []

if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    mesonet_dir = os.path.join(d, 'climate', 'madis', 'LDAD', 'mesonet')
    src_dir_ = os.path.join('/data/ssd1/madis', 'inclusive_csv')
    dst_dir_ = os.path.join(mesonet_dir, 'inclusive_csv')

    transfer_list(src_dir_, dst_dir_, workers=10)

# ========================= EOF ====================================================================
