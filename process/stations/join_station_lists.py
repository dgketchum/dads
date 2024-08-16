import os
import re

import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from utils.elevation import elevation_from_coordinate


def join_stations(snotel, mesonet, agrimet, out_file, fill_elevation=False, bounds=None):
    ''''''
    snotel_data = pd.read_csv(snotel)
    snotel_data['source'] = 'snotel'
    mesonet_data = pd.read_csv(mesonet)
    mesonet_data['source'] = 'madis'
    agrimet_data = pd.read_csv(agrimet)
    agrimet_data['source'] = agrimet_data['Source'] + '_gwx'

    agrimet_data = agrimet_data.rename(columns={'original_network_id': 'fid',
                                                'original_station_name': 'name',
                                                'ELEV_FT': 'elevation',
                                                'STATION_LAT': 'latitude',
                                                'STATION_LON': 'longitude'})
    agrimet_data['fid'] = agrimet_data['fid'].astype(str)
    agrimet_data['fid'] = [s.upper() for s in agrimet_data['fid']]
    agrimet_data['name'] = agrimet_data['name'].astype(str)
    agrimet_data['name'] = [s.upper() for s in agrimet_data['name']]

    mesonet_data = mesonet_data.rename(columns={'index': 'fid', 'NAME': 'name', 'ELEV': 'elevation'})
    mesonet_data['fid'] = mesonet_data['fid'].astype(str)
    mesonet_data['fid'] = [s.upper() for s in mesonet_data['fid']]
    mesonet_data['name'] = mesonet_data['name'].astype(str)
    mesonet_data['name'] = [s.upper() for s in mesonet_data['name']]

    snotel_data = snotel_data.rename(columns={'ID': 'fid', 'Elevation_ft': 'elevation',
                                              'Name': 'name', 'Latitude': 'latitude',
                                              'Longitude': 'longitude'})
    snotel_data['elevation'] /= 3.28084
    agrimet_data['elevation'] /= 3.28084

    snotel_data = snotel_data[['fid', 'name', 'elevation', 'latitude', 'longitude', 'source']]
    snotel_data['fid'] = [f.strip() for f in snotel_data['fid']]
    snotel_data = snotel_data.drop_duplicates()
    d, suspects = {}, []
    add_records(d, snotel_data.to_dict(orient='index'), suspects)

    mesonet_data = mesonet_data[['fid', 'name', 'elevation', 'latitude', 'longitude', 'source']]
    add_records(d, mesonet_data.to_dict(orient='index'), suspects)

    agrimet_data = agrimet_data[['fid', 'name', 'elevation', 'latitude', 'longitude', 'source']]
    add_records(d, agrimet_data.to_dict(orient='index'), suspects)



    if fill_elevation:
        for e, (i, r) in enumerate(d.items(), start=1):
            try:
                if r['elevation'] > 4200.0 or r['elevation'] < 0.0:
                    d[i]['elevation'] = elevation_from_coordinate(r['latitude'], r['longitude'])
                    print(i, r['source'], e, 'of', len(d))
                if isinstance(r['elevation'], type(None)):
                    d[i]['elevation'] = elevation_from_coordinate(r['latitude'], r['longitude'])
                    print(i, r['source'], e, 'of', len(d))
                elif np.isnan(r['elevation']):
                    d[i]['elevation'] = elevation_from_coordinate(r['latitude'], r['longitude'])
                    print(i, r['source'], e, 'of', len(d))
                else:
                    pass
            except KeyError:
                print('Elevation error at {}'.format(i))
                continue
            except requests.exceptions.JSONDecodeError:
                print('Elevation error at {}'.format(i))
                continue
            except Exception as e:
                print('Elevation error {} at {}'.format(e, i))
                continue

    comb_df = pd.DataFrame.from_dict(d, orient='index')
    comb_df = comb_df.sort_index()
    comb_df['fid'] = comb_df.index

    geometry = [Point(xy) for xy in zip(comb_df['longitude'], comb_df['latitude'])]
    gdf = gpd.GeoDataFrame(comb_df, geometry=geometry)

    gdf.to_file(out_file, crs='EPSG:4326', engine='fiona')
    gdf.drop(columns=['geometry'], inplace=True)
    df = pd.DataFrame(gdf)
    df.to_csv(out_file.replace('.shp', '.csv'))
    print(out_file)


def add_records(d, n, l):

    for k, v in n.items():

        try:
            _ = int(v['fid'])
            int_name = True
        except ValueError:
            int_name = False

        if int_name:
            r = re.sub(r'\W+', '_', v['name'])[:10].upper()
            d[r] = {kk: vv for kk, vv in v.items() if kk != 'fid'}
            continue


        if v['fid'] not in d.keys():
            d[v['fid']] = {kk: vv for kk, vv in v.items() if kk != 'fid'}
        elif v['name'] == d[v['fid']]['name']:
            continue
        elif d[v['fid']]['latitude'] - v['latitude'] < 0.2:
            continue
        else:
            r = re.sub(r'\W+', '_', v['name'])[:10].upper()
            d[r] = {kk: vv for kk, vv in v.items() if kk != 'fid'}
            l.append(v['fid'])


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    sno_list = os.path.join(d, 'climate', 'snotel', 'snotel_list.csv')
    meso_list = os.path.join(d, 'climate', 'madis', 'mesonet_sites.csv')
    agrim_list = os.path.join(d, 'dads', 'met', 'stations', 'openet_gridwxcomp_input.csv')
    stations = os.path.join(d, 'dads', 'met', 'stations', 'dads_stations_renamed_elev.shp')
    join_stations(sno_list, meso_list, agrim_list, stations, bounds=None, fill_elevation=True)

# ========================= EOF ====================================================================
