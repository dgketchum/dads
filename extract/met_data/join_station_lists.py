import os

import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from utils.elevation import elevation_from_coordinate

SNOTEL_DATA = pd.DataFrame(
    {
        'Name': ['Abbey'],
        'ID': ['ABY'],
        'State': ['California'],
        'Network': ['Snow Course/Aerial Marker'],
        'County': ['Plumas'],
        'Elevation_ft': [5650],
        'Latitude': [39.955],
        'Longitude': [-120.538],
        'HUC': [180201220103],
        'Date_of_Data': [''],
        'Date_Report_Created': ['10-11-2021, 04:26 PM MDT'],
    }
)

MESONET_DATA = pd.DataFrame(
    {
        'index': ['QURC2'],
        'latitude': [39.638999938964837],
        'longitude': [-104.768997192382812],
        'ELEV': [1752.599999999999909],
        'NET': [''],
        'NAME': ['Quincy_Res'],
    }
)

AGRIMET_DATA = pd.DataFrame(
    {
        'Index': [1],
        'original_station_name': ['Stuttgart'],
        'STATION_ID': ['001_AR'],
        'in_UCR_basin': [''],
        'in_UCR_project': ['TRUE'],
        'included': ['C:\\Users\\dunkerly\\Desktop\\may_runs\\001_AR_data.xlsx'],
        'cleaned_station_name': ['stuttgart'],
        'STATION_LAT': [34.46867],
        'STATION_LON': [-91.4204],
        'original_network_id': ['USDA_AR_1'],
        'STATION_ELEV_M': [61],
        'record_start': ['5/4/2008 0:00'],
        'record_end': ['12/31/2018 0:00'],
        'record_length': [10.65867198],
        'etr_obs_count': [3523],
        'Anemom_height_m': [3.65],
        'gridmet_comparison_notes': [''],
        'Source': ['USDA'],
        'State': ['AR'],
        'original_file_name': ['usda_stuttgart_output.xlsx'],
        'run_count': [2],
        'correction_notes': ['standard corrections'],
        'Lat_Long_Precision': [''],
        'wind_instrumentation': [''],
        'GRIDMET_ID': [314036],
        'LAT': [34.48333333],
        'LON': [-91.4333333],
        'ELEV_M': [61.84],
        'ELEV_FT': [202.8871456],
        'FIPS_C': [5001],
        'STPO': ['AR'],
        'COUNTYNAME': ['Arkansas'],
        'CNTYCATEGO': ['County'],
        'STATENAME': ['Arkansas'],
        'HUC8': [8020303],
        'GRID_FILE_PATH': ['C:\\Users\\dunkerly\\Desktop\\may_runs\\gridmet_data\\gridmet_historical_314036.csv'],
    }
)


def join_stations(snotel, mesonet, agrimet, out_file, fill_elevation=False, bounds=None):
    ''''''
    snotel_data = pd.read_csv(snotel)
    snotel_data['source'] = 'snotel'
    mesonet_data = pd.read_csv(mesonet)
    mesonet_data['source'] = 'madis'
    agrimet_data = pd.read_csv(agrimet)
    agrimet_data['source'] = 'gwx'

    snotel_data = snotel_data.rename(columns={'ID': 'fid', 'Elevation_ft': 'elevation',
                                              'Name': 'name', 'Latitude': 'latitude',
                                              'Longitude': 'longitude'})
    mesonet_data = mesonet_data.rename(
        columns={'index': 'fid', 'NAME': 'name', 'ELEV': 'elevation'}
    )

    agrimet_data = agrimet_data.rename(
        columns={
            'Index': 'fid',
            'cleaned_station_name': 'name',
            'ELEV_FT': 'elevation',
            'STATION_LAT': 'latitude',
            'STATION_LON': 'longitude',
        }
    )

    snotel_data = snotel_data[['fid', 'name', 'elevation', 'latitude', 'longitude', 'source']]
    mesonet_data = mesonet_data[['fid', 'name', 'elevation', 'latitude', 'longitude', 'source']]
    agrimet_data = agrimet_data[['fid', 'name', 'elevation', 'latitude', 'longitude', 'source']]

    comb_df = pd.concat([snotel_data, mesonet_data, agrimet_data])

    if bounds:
        w, s, e, n = bounds
        comb_df = comb_df[(comb_df['latitude'] < n) & (comb_df['latitude'] >= s)]
        comb_df = comb_df[(comb_df['longitude'] < e) & (comb_df['longitude'] >= w)]

    if fill_elevation:
        for i, r in comb_df.iterrows():
            try:
                if isinstance(r['elevation'], type(None)):
                    r['elevation'] = elevation_from_coordinate(r['latitude'], r['longitude'])
                    print(r['fid'], r['elevation'])
                elif np.isnan(r['elevation']):
                    r['elevation'] = elevation_from_coordinate(r['latitude'], r['longitude'])
                    print(r['fid'], r['elevation'])
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
    print(out_file)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'
    sno_list = os.path.join(d, 'climate', 'snotel', 'snotel_list.csv')
    meso_list = os.path.join(d, 'climate', 'madis', 'mesonet_sites.csv')
    agrim_list = os.path.join(d, 'dads', 'met', 'stations', 'openet_gridwxcomp_input.csv')
    out_file_ = os.path.join(d, 'dads', 'met', 'stations', 'dads_stations.shp')
    join_stations(sno_list, meso_list, agrim_list, out_file_, bounds=(-116., 45., -109., 49.),
                  fill_elevation=True)
    join_stations(sno_list, meso_list, agrim_list, out_file_, bounds=None, fill_elevation=True)


# ========================= EOF ====================================================================
