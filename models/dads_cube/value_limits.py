FEATURE_LIMITS = {
    "lat": (-90, 90),
    "lon": (-180, 180),
    "rsun": (1.0, 40.0),
    "doy": (0, 366),
    "B10": (0, 500),
    "B2": (-0.25, 1.6),
    "B3": (-0.25, 1.6),  # likely error if Landsat bands not [0,1]
    "B4": (-0.25, 1.6),
    "B5": (-0.25, 1.6),
    "B6": (-0.25, 1.6),
    "B7": (-0.25, 1.6),
    "SR1": (-1000, 15000),
    "SR2": (-1000, 15000),
    "SR3": (-1000, 15000),
    "BT1": (1500, 3500),
    "BT2": (1500, 3500),
    "BT3": (1500, 3500),
    "aspect": (0, 365),
    "elevation": (-100, 5000),
    "slope": (0, 90),
    "tpi_10000": (-1500, 1500),
    "tpi_22500": (-2000, 2000),
    "tpi_2500": (-600, 600),
    "tpi_500": (-500, 500),
}


TARGET_LIMITS = {
    "tmax": (-60, 50),
    "tmin": None,  # will raise if used
    "rsds": None,  # will raise if used
    "ea": None,  # will raise if used
    "wind": None,  # will raise if used
    "prcp": None,  # will raise if used
}


# ========================= EOF ====================================================================
