def station_par_map(station_type):
    if station_type == 'openet':
        return {'index': 'STATION_ID',
                'lat': 'LAT',
                'lon': 'LON',
                'elev': 'ELEV_M',
                'start': 'START DATE',
                'end': 'END DATE'}
    elif station_type == 'agri':
        return {'index': 'id',
                'lat': 'lat',
                'lon': 'lon',
                'elev': 'elev',
                'start': 'record_start',
                'end': 'record_end'}
    if station_type == 'ghcn':
        return {'index': 'STAID',
                'lat': 'LAT',
                'lon': 'LON',
                'elev': 'ELEV',
                'start': 'START DATE',
                'end': 'END DATE'}

    if station_type == 'madis':
        return {'index': 'index',
                'lat': 'latitude',
                'lon': 'longitude',
                'elev': 'ELEV'}
    if station_type == 'dads':
        return {'index': 'fid',
                'lat': 'latitude',
                'lon': 'longitude',
                'elev': 'elevation'}
    else:
        raise NotImplementedError


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
