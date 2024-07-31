import os
import json
import requests

import pandas as pd
import pyaqsapi
import pyaqsapi.listfunctions as listfunctions
from pyaqsapi.helperfunctions import aqs_credentials

root = '/media/nvm/IrrigationGIS/dads'
if not os.path.exists(root):
    root = '/home/dgketchum/data/IrrigationGIS/dads'

aq_d = os.path.join(root, 'aq')
js = os.path.join(root, 'aq', 'aqs_key.json')

with open(js, 'r') as f:
    key = json.load(f)

URL = ('https://aqs.epa.gov/data/api/dailyData/byCounty?email={a}&'
       'key={b}&param=88101&bdate={c}&edate={d}&state={e}&county={f}')


def download_county_air_quality_data(key_file, state_fips, county_fips, start_date, end_date):
    for yr in range(start_date, end_date):
        s, e = f'{yr}0101', f'{yr}1231'
        url = URL.format(a=key['email'], b=key['key'],
                         c=s, d=e, e=state_fips, f=county_fips)

        try:
            resp = requests.get(url)
            data_dict = json.loads(resp.content.decode('utf-8'))
            df = pd.DataFrame(data_dict['Data'])
            param = df.iloc[1]['parameter'].split(' - ')[0]
            df = df[['date_local', 'first_max_value']]
            df.columns = ['date', param]
            df = df.groupby('date').agg({param: 'first'})
            return df

        except pyaqsapi.PyAQSAPIError as e:
            print(f"Error downloading data: {e}")
            return None


if __name__ == '__main__':
    download_county_air_quality_data(js, '30', '031', 2000, 2024)
# ========================= EOF ====================================================================
