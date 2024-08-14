import os

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

    agrimet_data = agrimet_data.rename(columns={'Index': 'fid',
                                                'original_network_id': 'name',
                                                'ELEV_FT': 'elevation',
                                                'STATION_LAT': 'latitude',
                                                'STATION_LON': 'longitude'})

    mesonet_data = mesonet_data.rename(columns={'index': 'fid', 'NAME': 'name', 'ELEV': 'elevation'})
    mesonet_data['fid'] = mesonet_data['fid'].astype(str)

    snotel_data = snotel_data.rename(columns={'ID': 'fid', 'Elevation_ft': 'elevation',
                                              'Name': 'name', 'Latitude': 'latitude',
                                              'Longitude': 'longitude'})
    snotel_data['elevation'] /= 3.28084
    agrimet_data['elevation'] /= 3.28084

    snotel_data = snotel_data[['fid', 'name', 'elevation', 'latitude', 'longitude', 'source']]
    snotel_data['fid'] = [f.strip() for f in snotel_data['fid']]
    mesonet_data = mesonet_data[['fid', 'name', 'elevation', 'latitude', 'longitude', 'source']]
    agrimet_data = agrimet_data[['fid', 'name', 'elevation', 'latitude', 'longitude', 'source']]

    agrimet_names = [f.lower() for f in agrimet_data['name']]
    dup_gwx_madis = [r['fid'] for i, r in mesonet_data.iterrows() if r['fid'].lower().strip() in agrimet_names]
    idx = [i for i, r in mesonet_data['fid'].items() if r not in dup_gwx_madis]
    mesonet_data = mesonet_data.loc[idx]

    comb_df = pd.concat([snotel_data, mesonet_data, agrimet_data])

    if bounds:
        w, s, e, n = bounds
        comb_df = comb_df[(comb_df['latitude'] < n) & (comb_df['latitude'] >= s)]
        comb_df = comb_df[(comb_df['longitude'] < e) & (comb_df['longitude'] >= w)]

    if fill_elevation:
        for i, r in comb_df.iterrows():
            try:
                if isinstance(r['elevation'], type(None)):
                    comb_df.loc[i, 'elevation'] = elevation_from_coordinate(r['latitude'], r['longitude'])
                    print(r['fid'], i, 'of', comb_df.shape[0])
                elif np.isnan(r['elevation']):
                    comb_df.loc[i, 'elevation'] = elevation_from_coordinate(r['latitude'], r['longitude'])
                    print(r['fid'], i, 'of', comb_df.shape[0])
                else:
                    pass
            except KeyError:
                print('Elevation error at {}'.format(r['fid']))
                continue
            except requests.exceptions.JSONDecodeError:
                print('Elevation error at {}'.format(r['fid']))
                continue
            except Exception as e:
                print('Elevation error {} at {}'.format(e, r['fid']))
                continue

    geometry = [Point(xy) for xy in zip(comb_df['longitude'], comb_df['latitude'])]
    gdf = gpd.GeoDataFrame(comb_df, geometry=geometry)

    gdf.to_file(out_file, crs='EPSG:4326', engine='fiona')
    gdf.drop(columns=['geometry'], inplace=True)
    df = pd.DataFrame(gdf)
    df.to_csv(out_file.replace('.shp', '.csv'))
    print(out_file)


def fill_out_elevation(infile, outfile):
    """"""
    gdf = gpd.read_file(infile)

    for i, r in gdf.iterrows():
        try:
            if isinstance(r['elevation'], type(None)):
                el = elevation_from_coordinate(r['latitude'], r['longitude'])
                gdf.loc[i, 'elevation'] = el
                print(r['fid'], '{:.2f} {} of {}'.format(el, i, gdf.shape[0]))
            elif np.isnan(r['elevation']):
                el = elevation_from_coordinate(r['latitude'], r['longitude'])
                gdf.loc[i, 'elevation'] = el
                print(r['fid'], '{:.2f} {} of {}'.format(el, i, gdf.shape[0]))
            else:
                pass
        except KeyError:
            print('Elevation error at {}'.format(r['fid']))
            continue
        except requests.exceptions.JSONDecodeError:
            print('Elevation error at {}'.format(r['fid']))
            continue
        except Exception as e:
            print('Elevation error {} at {}'.format(e, r['fid']))
            continue

    gdf.to_file(outfile)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'
    sno_list = os.path.join(d, 'climate', 'snotel', 'snotel_list.csv')
    meso_list = os.path.join(d, 'climate', 'madis', 'mesonet_sites.csv')
    agrim_list = os.path.join(d, 'dads', 'met', 'stations', 'openet_gridwxcomp_input.csv')

    stations = os.path.join(d, 'dads', 'met', 'stations', 'dads_stations.shp')
    join_stations(sno_list, meso_list, agrim_list, stations, bounds=None, fill_elevation=True)
    station_elev = os.path.join(d, 'dads', 'met', 'stations', 'dads_stations_elev_.shp')
    # fill_out_elevation(stations, station_elev)

# ========================= EOF ====================================================================
