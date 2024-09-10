import json
import os
import glob
import time
from tqdm import tqdm

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

    years = list(range(start_date, end_date + 1))
    years.reverse()

    for code, obsname in AQS_PARAMETERS.items():

        for yr in years:

            if state_fips == '45' and int(county_fips) < 23:
                continue

            st_co = '{}{}'.format(state_fips, county_fips)
            if st_co in exists and not overwrite:
                continue

            start, end = f'{yr}0101', f'{yr}1231'
            url = URL.format(a=key['email'], b=key['key'], p=code,
                             c=start, d=end, e=state_fips, f=county_fips)

            try:
                try:
                    resp = requests.get(url)
                except TimeoutError:
                    time.sleep(600)
                    try:
                        resp = requests.get(url)
                    except Exception as e:
                        continue

                data_dict = json.loads(resp.content.decode('utf-8'))

                if isinstance(data_dict, str):
                    continue

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


def join_aq_data(data_src, data_dst, out_csv):
    """"""

    metadata = {}

    dt = pd.date_range('2000-01-01', '2024-06-30', freq='D')
    blank = pd.DataFrame(columns=[v for k, v in AQS_PARAMETERS.items()], index=pd.DatetimeIndex(dt))

    files_ = [f for f in os.listdir(data_src) if f.endswith('.csv')]

    records = []
    for f in files_:
        parts = f.split('_')
        state_fips, county_fips, site_no = parts[1][:2], parts[1][2:5], parts[1][5:]
        site_ = f'{state_fips}{county_fips}{site_no}'
        records.append(site_)

    sites = sorted(list(set(records)))

    for site in tqdm(sites, total=len(sites)):

        state_name = None

        df = blank.copy()

        filepaths = [os.path.join(data_src, f) for f in os.listdir(data_src) if site in f and f.endswith('.csv')]

        state_fips, county_fips, site_no = site[:2], site[2:5], site[5:]

        first = True
        for file_ in filepaths:

            splt = os.path.basename(file_).split('_')
            state_name, code = splt[0], splt[2]
            param = AQS_PARAMETERS[code]

            if site in metadata.keys():
                continue

            try:
                c = pd.read_csv(file_, index_col='date', parse_dates=True)
                c = c.groupby(c.index).agg('first')
            except Exception as e:
                print(e, os.path.basename(file_))
                continue

            if first:
                df['latitude'] = c.iloc[0]['latitude']
                df['longitude'] = c.iloc[0]['longitude']
                first = False

            idx = [i for i in c.index if i in df.index]
            df.loc[idx, param] = c.loc[idx, param].astype(float)

        if site not in metadata:
            metadata[site] = {
                'state': state_name,
                'st_fips': state_fips,
                'co_fips': county_fips,
                'site_no': site_no,
                'latitude': df['latitude'].iloc[0],
                'longitude': df['longitude'].iloc[0],
            }
            for param_code, param_name in AQS_PARAMETERS.items():
                metadata[site][param_name] = 0

        out_file = os.path.join(data_dst, f'{site}.csv')
        df.to_csv(out_file)

        for k, v in AQS_PARAMETERS.items():
            metadata[site][v] = np.count_nonzero(~pd.isna(df[v]))

    metadata_df = pd.DataFrame(metadata.values())
    metadata_df.to_csv(os.path.join(out_csv), index=False)

    sums = metadata_df[[v for k, v in AQS_PARAMETERS.items()]].sum(axis=0)
    [print(k, v) for k, v in sums.items()]

    gdf = gpd.GeoDataFrame(metadata_df, geometry=gpd.points_from_xy(metadata_df.longitude, metadata_df.latitude))
    shp = out_csv.replace('.csv', '.shp')
    gdf.to_file(shp, driver='ESRI Shapefile', crs='EPSG:4326', engine='fiona')


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/dads'

    aq_d = os.path.join(root, 'aq')
    aq_data_src = os.path.join(root, 'aq', 'data')
    aq_data_dst = os.path.join(root, 'aq', 'joined_data')
    aq_meta = os.path.join(aq_d, 'aqs_meta.json')
    aq_csv = os.path.join(aq_d, 'aqs.csv')
    missing_csv = os.path.join(aq_d, 'missing_aqs.csv')

    js = os.path.join(aq_d, 'aqs_key.json')

    fips = state_county_code()
    states = list(fips.keys())
    missing = ['NE']

    for state in missing:

        state_dct = fips[state]
        counties = {v['GEOID']: v['NAME'] for k, v in state_dct.items()}

        for geoid, name_ in counties.items():
            st_code, co_code = geoid[:2], geoid[2:]
            # download_county_air_quality_data(js, st_code, co_code, 2000, 2024, data_dst=aq_data_src)

    join_aq_data(aq_data_src, aq_data_dst, aq_csv)

# ========================= EOF ====================================================================
