import os
import requests
import urllib
import geopandas as gpd

url = 'https://epqs.nationalmap.gov/v1/json?x={}&y={}&units=Meters&wkid=4326&includeDate=False'


def elevation_from_coordinate(lat, lon):
    result = requests.get(url.format(lon, lat))
    elev = float(result.json()['value'])
    return elev


if __name__ == '__main__':
    r = '/media/research/IrrigationGIS'
    if not os.path.isdir(r):
        home = os.path.expanduser('~')
        r = os.path.join(home, 'data', 'IrrigationGIS')

    fields = os.path.join(r, 'climate', 'agrimet', 'agrimet_mt_aea.shp')
    out_fields = os.path.join(r, 'climate', 'agrimet', 'agrimet_mt_aea_elev.shp')

    gdf = gpd.read_file(fields)
    elev = elevation_from_coordinate(gdf.loc[0, 'lat'], gdf.loc[0, 'lon'])
    gdf['elev'] = gdf[['lat', 'lon']].apply(lambda r: elevation_from_coordinate(r['lat'], r['lon']), axis=1)
    gdf.to_file(out_fields)
# ========================= EOF ====================================================================
