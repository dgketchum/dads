MET_FEATURES = [
    'CAPE_nl_hr',
    'CRainf_frac_nl_hr',
    'LWdown_nl_hr',
    'PSurf_nl_hr',
    'PotEvap_nl_hr',
    'Qair_nl_hr',
    'Rainf_nl_hr',
    'SWdown_nl_hr',
    'Tair_nl_hr',
    'Wind_E_nl_hr',
    'Wind_N_nl_hr',
    'dt_nl_hr',
    'lat_nl_hr',
    'lon_nl_hr',
]

CDR_FEATURES = ['SR1', 'SR2', 'SR3', 'BT1', 'BT2', 'BT3']
LANDSAT_FEATURES = ['B10', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
TERRAIN_FEATURES = ['aspect', 'elevation', 'slope', 'tpi_10000', 'tpi_22500', 'tpi_2500', 'tpi_500']

GEO_FEATURES = ['lat', 'lon', 'rsun', 'doy'] + LANDSAT_FEATURES + CDR_FEATURES + TERRAIN_FEATURES

COMPARISON_FEATURES = ['mean_temp_dm', 'rsds_dm', 'vpd_dm']

TARGETS = ['rsds_obs', 'mean_temp_obs', 'min_temp_obs', 'max_temp_obs', 'vpd_obs', 'prcp_obs',
           'rn_obs', 'u2_obs', 'doy_obs']

ADDED_FEATURES = [
    'doy_sin',
    'doy_cos',
    'hour_sin',
    'hour_cos',
]

PTH_COLUMNS = TARGETS + COMPARISON_FEATURES + MET_FEATURES + GEO_FEATURES + ADDED_FEATURES

if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
