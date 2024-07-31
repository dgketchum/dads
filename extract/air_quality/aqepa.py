import json
import os

import pandas as pd
import requests


def download_county_air_quality_data(key_file, state_fips, county_fips, start_date, end_date):
    ''''''

    URL = ('https://aqs.epa.gov/data/api/dailyData/byCounty?email={a}&'
           'key={b}&param={p}&bdate={c}&edate={d}&state={e}&county={f}')

    with open(key_file, 'r') as f:
        key = json.load(f)

    for yr in range(start_date, end_date):
        s, e = f'{yr}0101', f'{yr}1231'
        aqs_param_codes = ["88101", "81102", "42602", "44201", "42401"]
        first = True

        for code in aqs_param_codes:

            url = URL.format(a=key['email'], b=key['key'], p=code,
                             c=s, d=e, e=state_fips, f=county_fips)

            try:
                resp = requests.get(url)
                data_dict = json.loads(resp.content.decode('utf-8'))
                c = pd.DataFrame(data_dict['Data'])
                param = c.iloc[1]['parameter'].split(' - ')[0]
                # read in site numbber for file save and lat/lon for shapefile
                c = c[['date_local', 'first_max_value']]
                c.columns = ['date', param]
                c = c.groupby('date').agg({param: 'first'})
                if first:
                    df = c
                    first = False
                else:
                    df = pd.concat([df, c], axis=1)

            except Exception as e:
                print(e)
                continue

        df.to_csv()


if __name__ == '__main__':
    root = '/media/nvm/IrrigationGIS/dads'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/dads'

    aq_d = os.path.join(root, 'aq')
    js = os.path.join(root, 'aq', 'aqs_key.json')

    download_county_air_quality_data(js, '30', '031', 2000, 2024)
# ========================= EOF ====================================================================
