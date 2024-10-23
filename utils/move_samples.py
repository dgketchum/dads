import os
import shutil

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

if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
