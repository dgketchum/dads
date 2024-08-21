import os
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
import requests
from pprint import pprint
from urllib.parse import urlunparse

from bs4 import BeautifulSoup

CATALOG = {
    2000: 'AVHRR-Land_v005_AVH09C1_NOAA-14',
    2001: 'AVHRR-Land_v005_AVH09C1_NOAA-16',
    2002: 'AVHRR-Land_v005_AVH09C1_NOAA-16',
    2003: 'AVHRR-Land_v005_AVH09C1_NOAA-16',
    2004: 'AVHRR-Land_v005_AVH09C1_NOAA-16',
    2005: 'AVHRR-Land_v005_AVH09C1_NOAA-16',
    2006: 'AVHRR-Land_v005_AVH09C1_NOAA-18',
    2007: 'AVHRR-Land_v005_AVH09C1_NOAA-18',
    2008: 'AVHRR-Land_v005_AVH09C1_NOAA-18',
    2009: 'AVHRR-Land_v005_AVH09C1_NOAA-18',
    2010: 'AVHRR-Land_v005_AVH09C1_NOAA-19',
    2011: 'AVHRR-Land_v005_AVH09C1_NOAA-19',
    2012: 'AVHRR-Land_v005_AVH09C1_NOAA-19',
    2013: 'AVHRR-Land_v005_AVH09C1_NOAA-19',
    2014: 'VIIRS-Land_v001_NPP09C1_S-NPP',
    2015: 'VIIRS-Land_v001_NPP09C1_S-NPP',
    2016: 'VIIRS-Land_v001_NPP09C1_S-NPP',
    2017: 'VIIRS-Land_v001_NPP09C1_S-NPP',
    2018: 'VIIRS-Land_v001_NPP09C1_S-NPP',
    2019: 'VIIRS-Land_v001_NPP09C1_S-NPP',
    2020: 'VIIRS-Land_v001_NPP09C1_S-NPP',
    2021: 'VIIRS-Land_v001_NPP09C1_S-NPP',
    2022: 'VIIRS-Land_v001_NPP09C1_S-NPP',
    2023: 'VIIRS-Land_v001_JP109C1_NOAA-20',
    2024: 'VIIRS-Land_v001_JP109C1_NOAA-20',
}

BASE_URL = 'https://www.ncei.noaa.gov/thredds/fileServer/cdr/surface-reflectance'

CATALOG_URL = 'https://www.ncei.noaa.gov/thredds/catalog/cdr/surface-reflectance/{}/catalog.html'


def get_catalog(out_meta):
    catalog = {}
    for year in range(2000, 2025):
        catalog[year] = {}
        catalog_url = CATALOG_URL.format(year)
        response = requests.get(catalog_url)

        soup = BeautifulSoup(response.content, 'html.parser')

        table = soup.find('table')
        rows = table.find_all('tr')
        headers = [th.text.strip() for th in rows[0].find_all('th')]

        data = []
        for row in rows[1:]:
            cols = row.find_all('td')
            cols = [col.text.strip() if col.find('a') is None else col.find('a').text.strip() for col in cols]
            data.append({header: col for header, col in zip(headers, cols)})

        for item in data:
            if item['Size'] == '':
                continue
            dataset = item['Dataset']
            dt_str = dataset.split('_')[-2]
            catalog[year][dt_str] = [dataset, '/'.join([BASE_URL, str(year), dataset])]
        print(year)

    with open(out_meta, 'w') as fp:
        json.dump(catalog, fp, indent=4)


def download(data_dir, meta):
    """Download and extract data for a given dataset."""

    with open(meta, 'r') as f:
        catalog = json.load(f)

    local_dir = os.path.join(data_dir, 'nc')
    download_count = 0

    for year in range(2000, 2025):
        start = time.perf_counter()
        urls = [v[1] for k, v in catalog[str(year)].items()]
        targets = [os.path.join(local_dir, v[0]) for k, v in catalog[str(year)].items()]

        if not all([os.path.exists(f) for f in targets]):
            try:
                input_file = "aria2c_input.txt"
                with open(input_file, "w") as f:
                    for file_name in urls:
                        f.write('{}\n'.format(file_name))
                aria2c_cmd = [
                    "aria2c", "-x", "16", "-s", "16", "--http-user",
                    "--allow-overwrite=true", "--dir", local_dir, "--input-file", input_file,
                ]
                subprocess.run(aria2c_cmd, check=True)

            except subprocess.CalledProcessError as e:
                print(f"Failed download to {local_dir} for {year} {e}")

            if download_count % 10 == 0:
                print('sleep wait')
                time.sleep(0.5)

            end = time.perf_counter()
            print(f"Download {year} took {end - start:0.2f} seconds")


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    usr, pswd = 'usr', 'pswd'
    cdr_data = os.path.join(d, 'dads', 'rs', 'cdr')
    cdr_meta = os.path.join(d, 'dads', 'rs', 'cdr_catalog.json')

    # get_catalog(cdr_meta)
    download(cdr_data, cdr_meta)

# ========================= EOF ====================================================================
