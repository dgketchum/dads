# =============================================================================================
# Copyright 2017 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================================
from __future__ import print_function, absolute_import

import io
import json
from pprint import pprint
from copy import deepcopy

import requests
from requests.compat import urlencode, OrderedDict
from datetime import datetime
from fiona import collection
from fiona.crs import from_epsg
from pandas import read_table, to_datetime, date_range, to_numeric, DataFrame

STATION_INFO_URL = 'https://www.usbr.gov/pn/agrimet/agrimetmap/usbr_map.json'
AGRIMET_MET_REQ_SCRIPT_PN = 'https://www.usbr.gov/pn-bin/agrimet.pl'
AGRIMET_CROP_REQ_SCRIPT_PN = 'https://www.usbr.gov/pn/agrimet/chart/{}{}et.txt'
AGRIMET_MET_REQ_SCRIPT_GP = 'https://www.usbr.gov/gp-bin/agrimet_archives.pl'
AGRIMET_CROP_REQ_SCRIPT_GP = 'https://www.usbr.gov/gp-bin/et_summaries.pl?station={}&year={}&submit2=++Submit++'
# in km
EARTH_RADIUS = 6371.

WEATHER_PARAMETRS_UNCONVERTED = [
    ('DATETIME', 'Date - [YYYY-MM-DD]'),
    ('ET', 'Evapotranspiration Kimberly-Penman - [in]'),
    ('MM', 'Mean Daily Air Temperature - [F]'),
    ('MN', 'Minimum Daily Air Temperature - [F]'),
    ('MX', 'Maximum Daily Air Temperature - [F]'),
    ('PC', 'Accumulated Precipitation Since Recharge/Reset - [in]'),
    ('PP', 'Daily (24 hour) Precipitation - [in]'),
    ('PU', 'Accumulated Water Year Precipitation - [in]'),
    ('SR', 'Daily Global Solar Radiation - [langleys]'),
    ('TA', 'Mean Daily Humidity - [%]'),
    ('TG', 'Growing Degree Days - [base 50F]'),
    ('YM', 'Mean Daily Dewpoint Temperature - [F]'),
    ('UA', 'Daily Average Wind Speed - [mph]'),
    ('UD', 'Daily Average Wind Direction - [deg az]'),
    ('WG', 'Daily Peak Wind Gust - [mph]'),
    ('WR', 'Daily Wind Run - [miles]'),
]

STANDARD_PARAMS = ['et', 'mm', 'mn',
                   'mx', 'pp', 'pu', 'sr', 'ta', 'tg',
                   'ua', 'ud', 'wg', 'wr', 'ym']


class Agrimet(object):
    def __init__(self, start_date=None, end_date=None, station=None,
                 interval=None, lat=None, lon=None, sat_image=None,
                 write_stations=False, region=None):

        self.station_info_url = STATION_INFO_URL
        self.station = station
        self.distance_from_station = None
        self.station_coords = None
        self.distances = None
        self.region = region

        self.empty_df = True

        if not station and not write_stations:
            if not lat and not sat_image:
                raise ValueError('Must initialize agrimet with a station, '
                                 'an Image, or some coordinates.')
            if not sat_image:
                self.station = self.find_closest_station(lat, lon)
            else:

                lat = (sat_image.corner_ll_lat_product + sat_image.corner_ul_lat_product) / 2
                lon = (sat_image.corner_ll_lon_product + sat_image.corner_lr_lon_product) / 2
                self.station = self.find_closest_station(lat, lon)

        if station:
            self.find_station_coords()

        self.interval = interval

        if start_date and end_date:
            self.start = datetime.strptime(start_date, '%Y-%m-%d')
            self.end = datetime.strptime(end_date, '%Y-%m-%d')
            self.today = datetime.now()
            self.start_index = (self.today - self.start).days - 1

        self.rank = 0

    @property
    def params(self):
        return urlencode(OrderedDict([
            ('cbtt', self.station),
            ('interval', self.interval),
            ('format', 2),
            ('back', self.start_index)
        ]))

    def find_station_coords(self):
        station_data = load_stations()
        sta_ = station_data[self.station]
        self.station_coords = sta_['geometry']['coordinates'][1], sta_['geometry']['coordinates'][0]

    def fetch_met_data(self, return_raw=False, out_csv_file=None, long_names=False):

        if self.region == 'pnro':
            url = '{}?{}'.format(AGRIMET_MET_REQ_SCRIPT_PN, self.params)
            r = requests.get(url)
            txt = r.text.split('\n')
            s_idx, e_idx = txt.index('BEGIN DATA\r'), txt.index('END DATA\r')

        if self.region == 'great_plains':
            pairs = ','.join(['{} {}'.format(self.station.upper(), x.upper()) for x in STANDARD_PARAMS])
            url = "https://www.usbr.gov/gp-bin/webarccsv.pl?parameter={0}&syer={1}&smnth={2}&sdy={3}&" \
                  "eyer={4}&emnth={5}&edy={6}&format=2".format(pairs,
                                                               self.start.year,
                                                               self.start.month,
                                                               self.start.day,
                                                               self.end.year,
                                                               self.end.month,
                                                               self.end.day)

            r = requests.get(url)
            txt = r.text.split('\r\n')
            s_idx, e_idx = txt.index('BEGIN DATA'), txt.index('END DATA')

        content = txt[s_idx + 1: e_idx]
        names = [c.strip() for c in content[0].split(',')]
        data = {name: [x.split(',')[i].strip() for x in content[1:]] for i, name in enumerate(names)}
        df = DataFrame(data)
        return df

    @staticmethod
    def write_agrimet_sation_shp(json_data, epsg, out):
        agri_schema = {'geometry': 'Point',
                       'properties': {
                           'program': 'str',
                           'url': 'str',
                           'siteid': 'str',
                           'title': 'str',
                           'state': 'str',
                           'type': 'str',
                           'region': 'str',
                           'install': 'str'}}

        cord_ref = from_epsg(epsg)
        shp_driver = 'ESRI Shapefile'

        with collection(out, mode='w', driver=shp_driver, schema=agri_schema,
                        crs=cord_ref) as output:
            for rec in json_data['features']:
                try:
                    output.write({'geometry': {'type': 'Point',
                                               'coordinates':
                                                   (rec['geometry']['coordinates'][0],
                                                    rec['geometry']['coordinates'][1])},
                                  'properties': {
                                      'program': rec['properties']['program'],
                                      'url': rec['properties']['url'],
                                      'siteid': rec['properties']['siteid'],
                                      'title': rec['properties']['title'],
                                      'state': rec['properties']['state'],
                                      'type': rec['properties']['type'],
                                      'region': rec['properties']['region'],
                                      'install': rec['properties']['install']}})
                except KeyError:
                    pass


def load_stations():
    r = requests.get(STATION_INFO_URL)
    stations = json.loads(r.text)
    stations = stations['features']
    stations = {s['properties']['siteid']: s for s in stations}
    return stations


if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================
