import json
import os
import glob
import time

import geopandas as gpd
import requests
import urllib3

import xarray as xr
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

from utils.state_county_names_codes import state_county_code, state_fips_code

AQS_PARAMETERS = {'88101': 'pm2.5',
                  '81102': 'pm10',
                  '42602': 'no2',
                  '44201': 'ozone',
                  '42401': 'so2'}

TARGET_STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']


def download_county_air_quality_data(key_file, state_fips, county_fips, start_date, end_date, data_dst,
                                     overwrite=False):
    """"""
    st_fips = {v: k for k, v in state_fips_code().items()}

    URL = ('https://aqs.epa.gov/data/api/dailyData/byCounty?email={a}&'
           'key={b}&param={p}&bdate={c}&edate={d}&state={e}&county={f}')

    with open(key_file, 'r') as f:
        key = json.load(f)

    # st/co/site/code/yr
    exists = ['_'.join(f.split('.')[0].split('_')[1:3]) for f in os.listdir(data_dst)]

    years = list(range(start_date, end_date))
    years.reverse()

    for code, obsname in AQS_PARAMETERS.items():

        for yr in years:

            st_co = '{}{}'.format(state_fips, county_fips)
            if st_co in exists and not overwrite:
                continue

            start, end = f'{yr}0101', f'{yr}1231'
            url = URL.format(a=key['email'], b=key['key'], p=code,
                             c=start, d=end, e=state_fips, f=county_fips)

            try:
                resp = requests.get(url)
                data_dict = json.loads(resp.content.decode('utf-8'))

                if data_dict['Header'][0]['status'] == 'No data matched your selection':
                    continue

                df = pd.DataFrame(data_dict['Data'])
                if df.empty:
                    continue

                sites = np.unique(df['site_number'])

                for site_no in sites:
                    file_ = os.path.join(data_dst, '{}_{}{}{}_{}_{}.csv'.format(st_fips[state_fips], state_fips,
                                                                                county_fips, site_no, code, yr))
                    c = df[df['site_number'] == site_no].copy()
                    c = c[['date_local', 'first_max_value', 'latitude', 'longitude']]
                    c.columns = ['date', obsname, 'latitude', 'longitude']
                    c = c.groupby('date').agg({obsname: 'first', 'latitude': 'first', 'longitude': 'first'})
                    c.to_csv(file_)
                    print(os.path.basename(file_))

            except requests.exceptions.ConnectionError:
                continue

            except json.decoder.JSONDecodeError:
                continue

            except urllib3.exceptions.ReadTimeoutError:
                time.sleep(60)
                continue


def create_daily_idw_raster(csv_files_pattern):
    dfs = []
    for file in glob.glob(csv_files_pattern):
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        dfs.append(df)

    df = pd.concat(dfs)
    ds = xr.Dataset.from_dataframe(df.set_index(['timestamp', 'latitude', 'longitude']))

    grid_lon = np.arange(-130, -60, 0.1)
    grid_lat = np.arange(25, 50, 0.1)

    def compute_idw(data):
        station_coords = data[['longitude', 'latitude']].values
        tree = cKDTree(station_coords)
        grid_coords = np.array(np.meshgrid(grid_lon, grid_lat)).T.reshape(-1, 2)
        distances, indices = tree.query(grid_coords, k=5)
        weights = 1.0 / distances**2
        weights /= weights.sum(axis=1, keepdims=True)
        aqi_values = data['aqi'].values[indices]
        interpolated_aqi = np.sum(weights * aqi_values, axis=1)
        return interpolated_aqi.reshape(len(grid_lat), len(grid_lon))

    daily_idw = ds.groupby('timestamp.date').apply(compute_idw)
    daily_idw_da = xr.DataArray(daily_idw.values, coords=[daily_idw.index, grid_lat, grid_lon], dims=['time', 'lat', 'lon'])
    daily_idw_da.to_netcdf('daily_aqi_idw.nc')


def write_aqs_shapefile(meta_js, shapefile_out):
    with open(meta_js, 'r') as fp:
        meta = json.load(fp)

    pop = []
    for k, v in meta.items():
        if v == 'nodata':
            pop.append(k)
    [meta.pop(k) for k in pop]
    gdf = pd.DataFrame().from_dict(meta).T
    gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf['lon'], gdf['lat'], crs='EPSG:4326'))
    gdf.to_file(shapefile_out)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/dads'

    aq_d = os.path.join(root, 'aq')
    aq_data = os.path.join(root, 'aq', 'data')
    aq_meta = os.path.join(aq_d, 'aqs_meta.json')
    aq_shp = os.path.join(aq_d, 'aqs.shp')

    js = os.path.join(aq_d, 'aqs_key.json')

    fips = state_county_code()
    states = list(fips.keys())
    # states.reverse()
    for state in states:

        if state not in TARGET_STATES:
            continue

        state_dct = fips[state]
        counties = {v['GEOID']: v['NAME'] for k, v in state_dct.items()}

        for geoid, name_ in counties.items():
            st_code, co_code = geoid[:2], geoid[2:]
            # download_county_air_quality_data(js, st_code, co_code, 2000, 2024, data_dst=aq_data)

        write_aqs_shapefile(meta_js=aq_meta, shapefile_out=aq_shp)

# ========================= EOF ====================================================================
