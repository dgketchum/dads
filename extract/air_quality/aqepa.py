import json
import os

import requests
import numpy as np
import pandas as pd
import geopandas as gpd

from utils.state_county_names_codes import state_county_code, state_fips_code

AQS_PARAMETERS = {'88101': 'pm2.5',
                  '81102': 'pm10',
                  '42602': 'no2',
                  '44201': 'ozone',
                  '42401': 'so2'}

TARGET_STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']


def download_county_air_quality_data(key_file, name, state_fips, county_fips, start_date, end_date, data_dst, meta_js):
    ''''''

    stco_code = '{}{}'.format(state_fips, county_fips)
    st_fips = {v: k for k, v in state_fips_code().items()}
    if os.path.exists(meta_js):
        with open(meta_js, 'r') as fp:
            meta = json.load(fp)
    else:
        meta = {}

    URL = ('https://aqs.epa.gov/data/api/dailyData/byCounty?email={a}&'
           'key={b}&param={p}&bdate={c}&edate={d}&state={e}&county={f}')

    with open(key_file, 'r') as f:
        key = json.load(f)

    years = list(range(start_date, end_date))
    years.reverse()

    lat, lon, site_no, df = None, None, None, None

    for code, obsname in AQS_PARAMETERS.items():

        first, code_df = True, None
        for yr in years:
            start, end = f'{yr}0101', f'{yr}1231'

            url = URL.format(a=key['email'], b=key['key'], p=code,
                             c=start, d=end, e=state_fips, f=county_fips)

            try:
                resp = requests.get(url)
                data_dict = json.loads(resp.content.decode('utf-8'))

                if data_dict['Header'][0]['status'] == 'No data matched your selection':
                    continue

                c = pd.DataFrame(data_dict['Data'])
                if c.empty:
                    continue

                if first:
                    lat, lon = c.iloc[0]['latitude'], c.iloc[0]['longitude']
                    site_no = c.iloc[0]['site_number']

                c = c[['date_local', 'first_max_value']]
                c.columns = ['date', obsname]
                c = c.groupby('date').agg({obsname: 'first'})

                if first:
                    code_df = c
                    first = False

                else:
                    code_df = pd.concat([code_df, c], axis=0, ignore_index=False)

            except requests.exceptions.ConnectionError:
                continue

        if code_df is None:
            continue

        elif not isinstance(df, pd.DataFrame):
            df = code_df.copy()

        else:
            df = pd.concat([df, code_df], axis=1, ignore_index=False)

    if not isinstance(df, pd.DataFrame):
        meta[stco_code] = 'nodata'
        pass
    else:
        df.index = pd.DatetimeIndex(df.index)
        df.sort_index(inplace=True)
        meta[stco_code] = {'state': st_fips[state_fips]}
        meta[stco_code].update({'county': name})
        meta[stco_code].update({'site_no': site_no})
        [meta[stco_code].update({'{}_len'.format(v): np.isfinite(df[v]).sum(axis=0).item()
        if v in df.columns else 0 for k, v in AQS_PARAMETERS.items()})]
        meta[stco_code].update({'start': df.index[0].strftime('%Y-%m-%d')})
        meta[stco_code].update({'end': df.index[-1].strftime('%Y-%m-%d')})
        meta[stco_code].update({'lat': lat.item()})
        meta[stco_code].update({'lon': lon.item()})
        file_ = os.path.join(data_dst, '{}{}{}.csv'.format(state_fips, county_fips, site_no))
        df.to_csv(file_)
        print('write {}, {}'.format(name, st_fips[state_fips]))

    with open(meta_js, 'w') as fp:
        json.dump(meta, fp)


def write_aqs_shapefile(meta_js, shapefile_out):
    with open(meta_js, 'r') as fp:
        meta = json.load(fp)

    pop = []
    for k, v in meta.items():
        if v == 'nodata':
            pop.append(k)
    [meta.pop(k) for k in pop]
    gdf = pd.DataFrame().from_dict(meta).T
    gdf = gpd.GeoDataFrame(gdf)
    gdf.geometry = gpd.points_from_xy(gdf['lon'], gdf['lat'], crs='EPSG:4326')
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
    for state in fips.keys():

        if state not in TARGET_STATES:
            continue

        state_dct = fips[state]
        counties = {v['GEOID']: v['NAME'] for k, v in state_dct.items()}

        for geoid, name_ in counties.items():
            st_code, co_code = geoid[:2], geoid[2:]
            download_county_air_quality_data(js, name_, st_code, co_code, 2022, 2024, data_dst=aq_data, meta_js=aq_meta)

        write_aqs_shapefile(meta_js=aq_meta, shapefile_out=aq_shp)

# ========================= EOF ====================================================================
