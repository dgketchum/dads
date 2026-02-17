"""
1 km EPSG:5070 PNW grid constants for the RTMA humidity-correction MVP.

All prep scripts and the training dataset import bounds, transform, and shape
from here so that every raster shares an identical pixel grid.
"""

from __future__ import annotations

import math

from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import transform_bounds

# Source PNW bounding box (EPSG:4326)
_PNW_LON_WEST, _PNW_LAT_SOUTH = -125.0, 42.0
_PNW_LON_EAST, _PNW_LAT_NORTH = -104.0, 49.0

PNW_1KM_CRS = CRS.from_epsg(5070)
PNW_1KM_RES = 1000.0  # metres

# Transform lat/lon bounds → EPSG:5070
_left, _bottom, _right, _top = transform_bounds(
    CRS.from_epsg(4326),
    PNW_1KM_CRS,
    _PNW_LON_WEST,
    _PNW_LAT_SOUTH,
    _PNW_LON_EAST,
    _PNW_LAT_NORTH,
)

# Snap outward to 1 km alignment
_left = math.floor(_left / PNW_1KM_RES) * PNW_1KM_RES
_bottom = math.floor(_bottom / PNW_1KM_RES) * PNW_1KM_RES
_right = math.ceil(_right / PNW_1KM_RES) * PNW_1KM_RES
_top = math.ceil(_top / PNW_1KM_RES) * PNW_1KM_RES

PNW_1KM_BOUNDS = (_left, _bottom, _right, _top)

_width = int((_right - _left) / PNW_1KM_RES)
_height = int((_top - _bottom) / PNW_1KM_RES)
PNW_1KM_SHAPE = (_height, _width)  # (rows, cols)

# North-up affine: origin is top-left corner
PNW_1KM_TRANSFORM = Affine(
    PNW_1KM_RES,
    0.0,
    _left,
    0.0,
    -PNW_1KM_RES,
    _top,
)

if __name__ == "__main__":
    print(f"CRS:       {PNW_1KM_CRS}")
    print(f"Bounds:    {PNW_1KM_BOUNDS}")
    print(f"Shape:     {PNW_1KM_SHAPE}  (H, W)")
    print(f"Transform: {PNW_1KM_TRANSFORM}")
    print(f"Pixels:    {_width} x {_height} = {_width * _height:,}")
