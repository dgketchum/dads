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


def download_county_air_quality_data(key_file, state_fips, county_fips, start_date, end_date, data_dst):
    """"""
    st_fips = {v: k for k, v in state_fips_code().items()}

    URL = ('https://aqs.epa.gov/data/api/dailyData/byCounty?email={a}&'
           'key={b}&param={p}&bdate={c}&edate={d}&state={e}&county={f}')

    with open(key_file, 'r') as f:
        key = json.load(f)

    years = list(range(start_date, end_date))
    years.reverse()

    for code, obsname in AQS_PARAMETERS.items():

        for yr in years:
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


def build_aq_metadata(d, outshp, out_json):
    state_fips, county_fips = None, None
    stco_code = '{}{}'.format(state_fips, county_fips)

    meta = {}

    if not isinstance(df, pd.DataFrame):
        meta[stco_code] = 'nodata'
        pass
    else:
        df.index = pd.DatetimeIndex(df.index)
        df.sort_index(inplace=True)
        meta[stco_code] = {'state': st_fips[state_fips]}
        meta[stco_code].update({'county': name})
        meta[stco_code].update({'site_no': site_no})
        _ = [meta[stco_code].update({'{}_len'.format(v): np.isfinite(df[v]).sum(axis=0).item()
        if v in df.columns else 0 for k, v in AQS_PARAMETERS.items()})]
        meta[stco_code].update({'start': df.index[0].strftime('%Y-%m-%d')})
        meta[stco_code].update({'end': df.index[-1].strftime('%Y-%m-%d')})
        meta[stco_code].update({'lat': lat.item()})
        meta[stco_code].update({'lon': lon.item()})
        df.to_csv(filoutshpe_)
        print('write {}, {}'.format(name, st_fips[state_fips]))

    with open(meta_js, 'w') as fp:
        json.dump(meta, fp, indent=4)


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
    states = list(fips.keys())[13:]
    states.reverse()
    for state in states:

        state_dct = fips[state]
        counties = {v['GEOID']: v['NAME'] for k, v in state_dct.items()}

        for geoid, name_ in counties.items():
            st_code, co_code = geoid[:2], geoid[2:]
            download_county_air_quality_data(js, st_code, co_code, 1990, 2024, data_dst=aq_data)

        write_aqs_shapefile(meta_js=aq_meta, shapefile_out=aq_shp)

# ========================= EOF ====================================================================
