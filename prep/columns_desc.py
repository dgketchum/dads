CDR_FEATURES = ['SR1', 'SR2', 'SR3', 'BT1', 'BT2', 'BT3']
LANDSAT_FEATURES = ['B10', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
TERRAIN_FEATURES = ['aspect', 'elevation', 'slope', 'tpi_10000', 'tpi_22500', 'tpi_2500', 'tpi_500']

GEO_FEATURES = ['lat', 'lon', 'rsun', 'doy'] + LANDSAT_FEATURES + CDR_FEATURES + TERRAIN_FEATURES

COMPARISON_FEATURES = ['prcp_gm', 'tmax_gm', 'tmin_gm', 'vpd_gm', 'rsds_gm', 'u2_gm']

# Canonical observed target columns aligned with pipeline usage
# Keep only actual observation targets here so row filtering in
# sequence building behaves correctly.
TARGETS = [
    'rsds_obs',
    'tmax_obs',
    'tmin_obs',
    'ea_obs',
    'prcp_obs',
    'wind_obs',
]

ADDED_FEATURES = [
    'doy_sin',
    'doy_cos',
    'hour_sin',
    'hour_cos',
    'time_diff'
]

if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
