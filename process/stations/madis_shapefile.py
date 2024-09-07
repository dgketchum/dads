import os

import geopandas as gpd
import pandas as pd

from utils.elevation import elevation_from_coordinate


def madis_station_shapefile(mesonet_dir, meta_file, outfile):
    """"""
    meta = pd.read_csv(meta_file, index_col='STAID')
    meta = meta.groupby(meta.index).first()
    unique_ids = set()
    unique_id_gdf = gpd.GeoDataFrame()

    shapefiles = [os.path.join(mesonet_dir, f) for f in os.listdir(mesonet_dir) if f.endswith('.shp')]
    for shapefile in shapefiles:
        print(os.path.basename(shapefile))
        gdf = gpd.read_file(shapefile)
        if 'index' in gdf.columns:
            unique_rows = gdf[~gdf['index'].isin(unique_ids)].copy()
            unique_rows.index = unique_rows['index']
            idx = [i for i in meta.index if i in unique_rows.index]

            unique_rows.loc[idx, 'ELEV'] = meta.loc[idx, 'ELEV']
            for i, r in unique_rows.iterrows():
                if isinstance(r['ELEV'], type(None)):
                    r['ELEV'] = elevation_from_coordinate(r['longitude'], r['latitude'])

            unique_rows['ELEV'] = unique_rows['ELEV'].astype(float)
            unique_rows.loc[idx, 'NET'] = meta.loc[idx, 'NET']
            unique_rows.loc[idx, 'NAME'] = meta.loc[idx, 'NAME']
            unique_ids.update(unique_rows['index'].unique())

            if unique_id_gdf.empty:
                unique_id_gdf = unique_rows
            else:
                unique_id_gdf = pd.concat([unique_id_gdf, unique_rows])

    unique_id_gdf.drop(columns=['index'], inplace=True)
    unique_id_gdf.to_file(outfile)
    print(outfile)


if __name__ == "__main__":

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    mesonet_dir = os.path.join(d, 'climate', 'madis', 'LDAD', 'mesonet')
    netcdf = os.path.join(mesonet_dir, 'netCDF')
    out_dir_ = os.path.join(mesonet_dir, 'csv')
    shapes = os.path.join(mesonet_dir, 'shapes')


    sites = os.path.join(mesonet_dir, 'mesonet_sites.shp')
    madis_station_shapefile(shapes, stn_meta, sites)

# ========================= EOF ====================================================================

