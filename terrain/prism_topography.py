"""PRISM terrain preprocessing algorithms — pure numerical library.

Implements the multi-scale topographic facet builder and effective terrain height
algorithm from Daly et al. (2008) and Daly et al. (2002), using a circular Gaussian
(Barnes 1964) filter, Horn's slope/aspect method, connected-component labeling, and
FFT convolution throughout.

No DADS-specific path handling.  All inputs and outputs are numpy arrays.

Target grid: PNW 1 km (1076 rows x 1758 cols, 1000 m cell, EPSG:5070), but the
functions are grid-agnostic — they accept arbitrary (H, W) float32/float64 arrays and
a cell_size parameter.

References
----------
Daly, C. et al. (2008). Physiographically sensitive mapping of climatological
    temperature and precipitation across the conterminous United States.
    Int. J. Climatol., 28, 2031-2064.
Daly, C. et al. (2002). A knowledge-based approach to the statistical mapping of
    climate. Climate Research, 22, 99-113.
Barnes, S.L. (1964). A technique for maximizing details in numerical weather map
    analysis. J. Appl. Meteor., 3, 396-409.
Horn, B.K.P. (1981). Hill shading and the reflectance map. Proc. IEEE, 69, 14-47.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label, minimum_filter
from scipy.signal import fftconvolve

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FACET_SCALES_KM: list[float] = [0.8, 12.0, 24.0, 36.0, 48.0, 60.0]

# Aspect orientation labels (0=flat, 1-8 = N, NE, E, SE, S, SW, W, NW)
FLAT = 0
NORTH = 1
NORTHEAST = 2
EAST = 3
SOUTHEAST = 4
SOUTH = 5
SOUTHWEST = 6
WEST = 7
NORTHWEST = 8

# ---------------------------------------------------------------------------
# Internal helpers (not part of public API)
# ---------------------------------------------------------------------------


def _radius_pixels(radius_m: float, cell_size_m: float) -> int:
    """Convert a metric radius to a pixel radius (minimum 1)."""
    return max(1, int(round(radius_m / cell_size_m)))


def circular_footprint(radius_m: float, cell_size: float) -> np.ndarray:
    """Boolean circular footprint for morphological operations.

    Parameters
    ----------
    radius_m:
        Radius in metres.
    cell_size:
        Grid cell size in metres.

    Returns
    -------
    np.ndarray
        2-D bool array of odd side length ``2*r_cells + 1``.
    """
    r_cells = int(np.ceil(radius_m / cell_size))
    side = 2 * r_cells + 1
    cy, cx = r_cells, r_cells
    y, x = np.ogrid[:side, :side]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) * cell_size
    return dist <= radius_m


def gaussian_kernel(radius_m: float, cell_size: float) -> np.ndarray:
    """Circular Gaussian (Barnes) kernel.

    Weight at distance d: ``exp(-d^2 / (2 * sigma^2))``, where
    ``sigma = radius_m / 2``.  Cells beyond ``radius_m`` receive zero weight.
    The kernel is normalized to sum to 1.

    Parameters
    ----------
    radius_m:
        Radius of influence in metres.
    cell_size:
        Grid cell size in metres.

    Returns
    -------
    np.ndarray
        2-D float64 kernel array of odd side length ``2*r_cells + 1``.
    """
    r_cells = int(np.ceil(radius_m / cell_size))
    side = 2 * r_cells + 1
    sigma = radius_m / 2.0
    cy, cx = r_cells, r_cells
    y, x = np.ogrid[:side, :side]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) * cell_size
    kernel = np.exp(-(dist**2) / (2.0 * sigma**2))
    kernel[dist > radius_m] = 0.0
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return kernel


def uniform_kernel(radius_m: float, cell_size: float) -> np.ndarray:
    """Circular uniform (box) kernel.

    All cells within *radius_m* get equal weight.  Normalized to sum to 1.

    Parameters
    ----------
    radius_m:
        Radius of influence in metres.
    cell_size:
        Grid cell size in metres.

    Returns
    -------
    np.ndarray
        2-D float64 kernel array of odd side length.
    """
    r_cells = int(np.ceil(radius_m / cell_size))
    side = 2 * r_cells + 1
    cy, cx = r_cells, r_cells
    y, x = np.ogrid[:side, :side]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) * cell_size
    kernel = (dist <= radius_m).astype(np.float64)
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return kernel


# ---------------------------------------------------------------------------
# 2. NaN-aware filter functions
# ---------------------------------------------------------------------------

_NAN_MASK_THRESHOLD = 1e-6  # minimum valid-weight sum to trust a result


def _nan_aware_fft_smooth(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """NaN-aware FFT convolution for any pre-built kernel.

    Replace NaN cells with 0, convolve data; create a binary valid-mask,
    convolve mask; divide.  Result is NaN where the accumulated valid weight
    falls below the threshold.

    Parameters
    ----------
    arr:
        Input 2-D array (may contain NaN).
    kernel:
        Pre-normalized 2-D convolution kernel (sums to 1).

    Returns
    -------
    np.ndarray
        Smoothed float64 array, NaN where data coverage is insufficient.
    """
    arr = arr.astype(np.float64)
    kernel = kernel.astype(np.float64)
    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0)
    data_conv = fftconvolve(filled, kernel, mode="same")
    mask_conv = fftconvolve(valid.astype(np.float64), kernel, mode="same")
    return np.where(
        mask_conv >= _NAN_MASK_THRESHOLD,
        data_conv / mask_conv,
        np.nan,
    )


def gaussian_smooth(arr: np.ndarray, radius_m: float, cell_size: float) -> np.ndarray:
    """NaN-aware circular Gaussian smoothing via FFT convolution.

    Convolve both filled-data and valid-mask, divide for proper normalization.

    Parameters
    ----------
    arr:
        Input 2-D array (may contain NaN).
    radius_m:
        Gaussian radius of influence in metres.
    cell_size:
        Grid cell size in metres.

    Returns
    -------
    np.ndarray
        Smoothed float64 array.
    """
    return _nan_aware_fft_smooth(arr, gaussian_kernel(radius_m, cell_size))


def uniform_smooth(arr: np.ndarray, radius_m: float, cell_size: float) -> np.ndarray:
    """NaN-aware circular uniform smoothing via FFT convolution.

    Parameters
    ----------
    arr:
        Input 2-D array (may contain NaN).
    radius_m:
        Radius of the uniform averaging window in metres.
    cell_size:
        Grid cell size in metres.

    Returns
    -------
    np.ndarray
        Smoothed float64 array.
    """
    return _nan_aware_fft_smooth(arr, uniform_kernel(radius_m, cell_size))


def circular_minimum(arr: np.ndarray, radius_m: float, cell_size: float) -> np.ndarray:
    """Circular minimum filter using ``scipy.ndimage.minimum_filter``.

    NaN cells are set to ``+inf`` before filtering so they do not contaminate
    neighbours; they are restored to NaN in the output.

    Parameters
    ----------
    arr:
        Input 2-D array (may contain NaN).
    radius_m:
        Radius of the circular footprint in metres.
    cell_size:
        Grid cell size in metres.

    Returns
    -------
    np.ndarray
        Float64 array of local circular minima.
    """
    nan_mask = ~np.isfinite(arr)
    filled = arr.astype(np.float64)
    filled[nan_mask] = np.inf
    fp = circular_footprint(radius_m, cell_size)
    result = minimum_filter(filled, footprint=fp, mode="nearest")
    result[nan_mask] = np.nan
    return result


# ---------------------------------------------------------------------------
# Slope and aspect
# ---------------------------------------------------------------------------


def horns_slope_aspect(
    dem: np.ndarray, cell_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute slope and geographic aspect using Horn (1981).

    Aspect is degrees clockwise from north (0 = N, 90 = E, 180 = S, 270 = W).
    Flat cells (dzdx == dzdy == 0) get aspect = 0.

    Parameters
    ----------
    dem : np.ndarray, float32
        2-D elevation array (metres).
    cell_size_m : float
        Pixel size in metres.

    Returns
    -------
    slope_deg : np.ndarray, float32
        Slope in degrees [0, 90].
    aspect_deg : np.ndarray, float32
        Geographic aspect in degrees [0, 360).
    """
    # Horn's 3x3 gradient weights
    # dzdx = ((z3+2z6+z9) - (z1+2z4+z7)) / (8 * cs)
    # dzdy = ((z7+2z8+z9) - (z1+2z2+z3)) / (8 * cs)
    # In array indexing: row increases downward (= southward for north-up raster)
    cs = cell_size_m

    # Pad with edge reflection
    p = np.pad(dem.astype(np.float64), 1, mode="edge")

    z1 = p[:-2, :-2]
    z2 = p[:-2, 1:-1]
    z3 = p[:-2, 2:]
    z4 = p[1:-1, :-2]
    # z5 = p[1:-1, 1:-1]  # centre, not used
    z6 = p[1:-1, 2:]
    z7 = p[2:, :-2]
    z8 = p[2:, 1:-1]
    z9 = p[2:, 2:]

    dzdx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8.0 * cs)
    dzdy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8.0 * cs)

    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # Geographic aspect: atan2(-dzdx, dzdy) gives CCW from north;
    # negate to get CW from north, then map to [0, 360).
    # Equivalent to: aspect = 180 - degrees(atan2(dzdy, -dzdx))
    # Standard GIS formulation (ESRI / GRASS geographic aspect):
    #   aspect_geo = atan2(-dzdx, dzdy)  (CW from N)
    aspect_rad = np.arctan2(-dzdx, dzdy)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = aspect_deg % 360.0
    aspect_deg = aspect_deg.astype(np.float32)

    return slope_deg, aspect_deg


# ---------------------------------------------------------------------------
# Aspect classification
# ---------------------------------------------------------------------------


def classify_aspect(
    aspect_deg: np.ndarray,
    slope_deg: np.ndarray,
    flat_slope_threshold: float = 2.0,
) -> np.ndarray:
    """Classify aspect into 8 cardinal/intercardinal bins + flat class.

    Classes:
        0 = flat (slope < threshold)
        1 = N   (337.5 – 22.5°)
        2 = NE  (22.5 – 67.5°)
        3 = E   (67.5 – 112.5°)
        4 = SE  (112.5 – 157.5°)
        5 = S   (157.5 – 202.5°)
        6 = SW  (202.5 – 247.5°)
        7 = W   (247.5 – 292.5°)
        8 = NW  (292.5 – 337.5°)

    Parameters
    ----------
    aspect_deg : np.ndarray
        Geographic aspect in degrees [0, 360).
    slope_deg : np.ndarray
        Slope in degrees.
    flat_slope_threshold : float
        Cells with slope < this value are classified as flat (0).

    Returns
    -------
    np.ndarray, uint8
        Orientation class array.
    """
    # Bin index: 0-based bin from (aspect + 22.5) / 45, wrapped mod 8, then +1
    # gives 1-8 for non-flat.
    bin_idx = ((aspect_deg + 22.5) % 360.0 / 45.0).astype(np.int32)  # 0-7
    classes = (bin_idx % 8 + 1).astype(np.uint8)  # 1-8
    classes[slope_deg < flat_slope_threshold] = 0
    return classes


# ---------------------------------------------------------------------------
# Connected-component facet labelling
# ---------------------------------------------------------------------------


def label_facets(orientation_class: np.ndarray) -> np.ndarray:
    """Label connected regions of the same orientation class.

    Uses 4-connectivity (PRISM convention: facets must share an edge).

    Parameters
    ----------
    orientation_class : np.ndarray, uint8
        Grid of orientation classes (0–8). Class 0 (flat) cells are labelled
        separately from each other and from non-flat classes.

    Returns
    -------
    np.ndarray, int32
        Unique integer ID for each connected component (1-based; 0 = nodata
        if orientation_class had nodata, but here all cells get an ID).
    """
    out = np.zeros(orientation_class.shape, dtype=np.int32)
    offset = 0
    for cls in np.unique(orientation_class):
        mask = orientation_class == cls
        labeled, n = label(mask, structure=np.ones((3, 3), dtype=np.int32))
        out[mask] = labeled[mask] + offset
        offset += n
    return out


# ---------------------------------------------------------------------------
# PRISM terrain index I3c
# ---------------------------------------------------------------------------


def terrain_index_i3c(height: np.ndarray) -> np.ndarray:
    """Piecewise linear terrain index I3c from PRISM documentation.

    Maps effective terrain height (metres) to [0, 1]:
        height <= 75 m   → 0
        75 < height < 250 m → linearly 0 → 1
        height >= 250 m  → 1

    Parameters
    ----------
    height : np.ndarray, float32
        Effective terrain height in metres.

    Returns
    -------
    np.ndarray, float32
        I3c terrain index in [0, 1].
    """
    h = np.asarray(height, dtype=np.float32)
    result = np.clip((h - 75.0) / 175.0, 0.0, 1.0)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Effective terrain height
# ---------------------------------------------------------------------------


def effective_terrain_height(
    dem: np.ndarray,
    cell_size_m: float,
    base_radius_m: float = 12_000.0,
) -> np.ndarray:
    """Elevation above the local topographic base (uniform-smoothed DEM).

    The "base elevation" is estimated as a large-radius uniform circular mean
    of the DEM.  Effective terrain height = DEM - base.

    Parameters
    ----------
    dem : np.ndarray, float32
        2-D elevation array (metres).
    cell_size_m : float
        Pixel size in metres.
    base_radius_m : float
        Radius for base-elevation smoothing (default 12 km, ~PRISM scale).

    Returns
    -------
    np.ndarray, float32
        Effective terrain height in metres.
    """
    base = uniform_smooth(dem.astype(np.float32), base_radius_m, cell_size_m)
    return (dem.astype(np.float32) - base).astype(np.float32)


# ---------------------------------------------------------------------------
# PRISM DEM products  (Daly et al. 2008, Table II)
# ---------------------------------------------------------------------------

#: Temperature DEM: Barnes-averaged raw DEM with radius 1.3 km.
TEMPERATURE_DEM_RADIUS_M: float = 1_300.0
#: Precipitation DEM: additional Gaussian smooth with radius ~7 km.
PRECIPITATION_DEM_RADIUS_M: float = 7_000.0


def build_temperature_dem(
    dem: np.ndarray,
    cell_size_m: float = 1_000.0,
    radius_m: float = TEMPERATURE_DEM_RADIUS_M,
) -> np.ndarray:
    """Gaussian-smooth the raw DEM to produce the PRISM temperature DEM.

    Applies a Barnes (1964) circular Gaussian with radius of influence
    1.3 km (Daly et al. 2008, Table II).

    Parameters
    ----------
    dem : np.ndarray, float32
        Raw elevation array (metres).  NaN where nodata.
    cell_size_m : float
        Pixel size in metres.
    radius_m : float
        Gaussian radius of influence (default 1 300 m).

    Returns
    -------
    np.ndarray, float32
    """
    return gaussian_smooth(dem.astype(np.float32), radius_m, cell_size_m)


def build_precipitation_dem(
    temperature_dem: np.ndarray,
    cell_size_m: float = 1_000.0,
    radius_m: float = PRECIPITATION_DEM_RADIUS_M,
) -> np.ndarray:
    """Apply additional Gaussian smooth to produce the PRISM precipitation DEM.

    Removes terrain features smaller than ~7 km, reflecting the scale at
    which orographic precipitation responds to topography.

    Parameters
    ----------
    temperature_dem : np.ndarray, float32
        Temperature DEM from :func:`build_temperature_dem`.
    cell_size_m : float
        Pixel size in metres.
    radius_m : float
        Gaussian radius of influence (default 7 000 m).

    Returns
    -------
    np.ndarray, float32
    """
    return gaussian_smooth(temperature_dem.astype(np.float32), radius_m, cell_size_m)


# ---------------------------------------------------------------------------
# Full effective terrain height pipeline  (Daly et al. 2008, Section 4.2.6)
# ---------------------------------------------------------------------------

#: Radii for the four-step ETH pipeline (metres).
ETH_MIN_RADIUS_M: float = 22_000.0
ETH_BASE_SMOOTH_RADIUS_M: float = 11_000.0
ETH_FINAL_SMOOTH_RADIUS_M: float = 15_000.0

#: Default 2D/3D thresholds (metres).
H2_DEFAULT: float = 75.0
H3_DEFAULT: float = 250.0

#: I3a inverse-distance support radius (metres).
I3A_RADIUS_M: float = 100_000.0


def build_effective_terrain_height(
    precip_dem: np.ndarray,
    cell_size_m: float = 1_000.0,
    min_radius_m: float = ETH_MIN_RADIUS_M,
    base_smooth_radius_m: float = ETH_BASE_SMOOTH_RADIUS_M,
    final_smooth_radius_m: float = ETH_FINAL_SMOOTH_RADIUS_M,
    verbose: bool = True,
) -> np.ndarray:
    """Four-step PRISM effective terrain height pipeline.

    Steps (Daly et al. 2008, Section 4.2.6 and Table II):

    1. Circular minimum filter (22 km) on the precipitation DEM.
    2. Circular uniform mean (11 km) of the step-1 base.
    3. ``precip_dem - smoothed_base``.
    4. Circular uniform mean (15 km) of the raw profile.

    Parameters
    ----------
    precip_dem : np.ndarray, float32
        Precipitation DEM.
    cell_size_m : float
        Pixel size in metres.
    min_radius_m : float
        Radius for step-1 minimum filter (default 22 000 m).
    base_smooth_radius_m : float
        Radius for step-2 uniform smooth (default 11 000 m).
    final_smooth_radius_m : float
        Radius for step-4 uniform smooth (default 15 000 m).
    verbose : bool
        Print progress messages.

    Returns
    -------
    np.ndarray, float32
        Effective terrain height (metres).
    """
    if verbose:
        print("  ETH step 1: circular minimum filter …")
    base_min = circular_minimum(
        precip_dem.astype(np.float32), min_radius_m, cell_size_m
    )
    if verbose:
        print("  ETH step 2: uniform smooth of base elevation …")
    base_smooth = uniform_smooth(base_min, base_smooth_radius_m, cell_size_m)
    if verbose:
        print("  ETH step 3: subtract smoothed base …")
    raw_profile = (precip_dem.astype(np.float32) - base_smooth).astype(np.float32)
    if verbose:
        print("  ETH step 4: uniform smooth of raw profile …")
    eth = uniform_smooth(raw_profile, final_smooth_radius_m, cell_size_m)
    return eth.astype(np.float32)


# ---------------------------------------------------------------------------
# 2D/3D terrain coverage indices  (Daly et al. 2008, Appendix C)
# ---------------------------------------------------------------------------


def build_i3c(
    eth: np.ndarray,
    h2: float = H2_DEFAULT,
    h3: float = H3_DEFAULT,
) -> np.ndarray:
    """Cell-local terrain coverage index I3c (Equation C1).

    ::

        I3c = 0                          if h_c <= h2
        I3c = (h_c - h2) / (h3 - h2)   if h2 < h_c < h3
        I3c = 1                          if h_c >= h3

    Parameters
    ----------
    eth : np.ndarray, float32
        Effective terrain height (metres).
    h2 : float
        2D threshold (default 75 m, Daly et al. 2008).
    h3 : float
        3D threshold (default 250 m, Daly et al. 2008).

    Returns
    -------
    np.ndarray, float32, values in [0, 1].
    """
    h = np.asarray(eth, dtype=np.float32)
    return np.clip((h - h2) / (h3 - h2), 0.0, 1.0).astype(np.float32)


def build_i3a(
    eth: np.ndarray,
    cell_size_m: float = 1_000.0,
    support_radius_m: float = I3A_RADIUS_M,
    h2: float = H2_DEFAULT,
    h3: float = H3_DEFAULT,
) -> np.ndarray:
    """Areal 3D support index I3a (Equation C2).

    Inverse-distance-weighted mean of I3c within *support_radius_m*.
    Pixels at the origin receive a distance of ``cell_size_m / 2`` to avoid
    division by zero.

    Parameters
    ----------
    eth : np.ndarray, float32
        Effective terrain height (metres).
    cell_size_m : float
        Pixel size in metres.
    support_radius_m : float
        Support radius in metres (default 100 000 m).
    h2 : float
        2D threshold passed to :func:`build_i3c`.
    h3 : float
        3D threshold passed to :func:`build_i3c`.

    Returns
    -------
    np.ndarray, float32, values in [0, 1].
    """
    from scipy.signal import fftconvolve

    i3c = build_i3c(eth, h2=h2, h3=h3).astype(np.float64)

    r_pix = _radius_pixels(support_radius_m, cell_size_m)
    y, x = np.ogrid[-r_pix : r_pix + 1, -r_pix : r_pix + 1]
    d2 = (x * cell_size_m) ** 2 + (y * cell_size_m) ** 2
    d = np.sqrt(d2)
    d_safe = np.where(d == 0.0, cell_size_m / 2.0, d)
    k = np.where(d <= support_radius_m, 1.0 / d_safe, 0.0).astype(np.float64)

    valid = np.isfinite(i3c).astype(np.float64)
    filled = np.where(np.isfinite(i3c), i3c, 0.0)

    data_conv = fftconvolve(filled, k, mode="same")
    mask_conv = fftconvolve(valid, k, mode="same")

    with np.errstate(invalid="ignore", divide="ignore"):
        i3a = np.where(mask_conv > 0.0, data_conv / mask_conv, np.nan)

    return np.clip(i3a, 0.0, 1.0).astype(np.float32)


def build_i3d(i3c: np.ndarray, i3a: np.ndarray) -> np.ndarray:
    """Final terrain coverage index I3d = max(I3c, I3a).

    Parameters
    ----------
    i3c : np.ndarray, float32
    i3a : np.ndarray, float32

    Returns
    -------
    np.ndarray, float32, values in [0, 1].
    """
    return np.fmax(i3c, i3a).astype(np.float32)


# ---------------------------------------------------------------------------
# Facet hierarchy  (6 scales, Daly et al. 2008, Table II)
# ---------------------------------------------------------------------------

#: Default scale-proportional minimum facet sizes (cells).
FACET_MIN_CELLS: list[int] = [5, 10, 15, 20, 30, 40]


def build_facet_scale(
    precip_dem: np.ndarray,
    radius_m: float,
    flat_slope_deg: float = 2.0,
    min_facet_cells: int = 20,
    cell_size_m: float = 1_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build facet ID and orientation grids for one spatial scale.

    Pipeline:

    1. Gaussian-smooth the precipitation DEM at *radius_m*.
    2. Compute slope and aspect (Horn's method).
    3. Classify aspect into 8-direction + flat.
    4. Connected-component label contiguous same-orientation cells.
    5. Merge facets smaller than *min_facet_cells* into adjacent facets.

    Parameters
    ----------
    precip_dem : np.ndarray, float32
        Precipitation DEM.
    radius_m : float
        Gaussian smoothing radius for this scale (metres).
    flat_slope_deg : float
        Flat threshold (degrees).
    min_facet_cells : int
        Merge facets smaller than this into neighbours.
    cell_size_m : float
        Pixel size in metres.

    Returns
    -------
    facet_id : np.ndarray, int32  (H, W)
    facet_orient : np.ndarray, uint8  (H, W)
    """
    smoothed = gaussian_smooth(precip_dem.astype(np.float32), radius_m, cell_size_m)
    slope_deg, aspect_deg = horns_slope_aspect(smoothed, cell_size_m)
    orient = classify_aspect(aspect_deg, slope_deg, flat_slope_deg)
    facet_id = label_facets(orient)
    if min_facet_cells > 0:
        facet_id, orient = _merge_small_facets(facet_id, orient, min_facet_cells)
    return facet_id.astype(np.int32), orient.astype(np.uint8)


def _merge_small_facets(
    facet_id: np.ndarray,
    orient: np.ndarray,
    min_facet_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge facets smaller than *min_facet_cells* into the best adjacent facet.

    "Best" = longest shared border; ties broken by smallest orientation
    angular difference.
    """
    from collections import defaultdict

    facet_id = facet_id.copy()
    orient = orient.copy()

    n_total = int(facet_id.max()) + 1
    counts = np.bincount(facet_id.ravel(), minlength=n_total)

    fid_orient = np.zeros(n_total, dtype=np.uint8)
    for fid in range(1, n_total):
        mask = facet_id == fid
        if mask.any():
            codes, c = np.unique(orient[mask], return_counts=True)
            fid_orient[fid] = int(codes[c.argmax()])

    adj: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for dr, dc in ((0, 1), (1, 0)):
        a = facet_id[: facet_id.shape[0] - dr, : facet_id.shape[1] - dc].ravel()
        b = facet_id[dr:, dc:].ravel()
        for ai, bi in zip(a, b):
            if ai != bi and ai > 0 and bi > 0:
                adj[ai][bi] += 1
                adj[bi][ai] += 1

    small_ids = sorted(
        [fid for fid in range(1, n_total) if 0 < counts[fid] < min_facet_cells],
        key=lambda fid: counts[fid],
    )

    def _angle_diff(c1: int, c2: int) -> int:
        if c1 == 0 or c2 == 0:
            return 4
        a_deg, b_deg = (c1 - 1) * 45, (c2 - 1) * 45
        diff = abs(a_deg - b_deg)
        return min(diff, 360 - diff) // 45

    for fid in small_ids:
        if counts[fid] == 0:
            continue
        if not adj.get(fid):
            continue
        best_nbr = max(
            adj[fid],
            key=lambda nbr: (
                adj[fid][nbr],
                -_angle_diff(int(fid_orient[fid]), int(fid_orient[nbr])),
            ),
        )
        mask = facet_id == fid
        facet_id[mask] = best_nbr
        orient[mask] = fid_orient[best_nbr]
        counts[best_nbr] += counts[fid]
        counts[fid] = 0
        for nbr, border in list(adj[fid].items()):
            if nbr == best_nbr:
                continue
            adj[best_nbr][nbr] = adj[best_nbr].get(nbr, 0) + border
            if fid in adj.get(nbr, {}):
                adj[nbr][best_nbr] = adj[nbr].get(best_nbr, 0) + border
                del adj[nbr][fid]
        del adj[fid]

    return facet_id, orient


def build_facet_hierarchy(
    precip_dem: np.ndarray,
    flat_slope_deg: float = 2.0,
    min_facet_cells: list[int] | None = None,
    scales_km: list[float] | None = None,
    cell_size_m: float = 1_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the full 6-scale PRISM facet hierarchy.

    Parameters
    ----------
    precip_dem : np.ndarray, float32
        Precipitation DEM.
    flat_slope_deg : float
        Flat threshold in degrees.
    min_facet_cells : list[int] or None
        Per-scale minimum facet size.  Defaults to :data:`FACET_MIN_CELLS`.
    scales_km : list[float] or None
        Smoothing radii in km.  Defaults to :data:`FACET_SCALES_KM`.
    cell_size_m : float
        Pixel size in metres.

    Returns
    -------
    facet_ids : np.ndarray, int32, shape (6, H, W)
    facet_orients : np.ndarray, uint8, shape (6, H, W)
    """
    if scales_km is None:
        scales_km = FACET_SCALES_KM
    if min_facet_cells is None:
        min_facet_cells = FACET_MIN_CELLS

    H, W = precip_dem.shape
    n = len(scales_km)
    facet_ids = np.zeros((n, H, W), dtype=np.int32)
    facet_orients = np.zeros((n, H, W), dtype=np.uint8)

    for i, (scale_km, min_cells) in enumerate(zip(scales_km, min_facet_cells)):
        radius_m = scale_km * 1_000.0
        print(f"  facet scale {i + 1}/{n}: radius={scale_km} km, min_cells={min_cells}")
        fid, fori = build_facet_scale(
            precip_dem,
            radius_m=radius_m,
            flat_slope_deg=flat_slope_deg,
            min_facet_cells=min_cells,
            cell_size_m=cell_size_m,
        )
        facet_ids[i] = fid
        facet_orients[i] = fori

    return facet_ids, facet_orients


def facet_counts_per_scale(
    facet_ids: np.ndarray,
) -> list[int]:
    """Return the number of distinct facet IDs at each scale band.

    Parameters
    ----------
    facet_ids : np.ndarray, int32, shape (N, H, W)

    Returns
    -------
    list[int]  Length N.
    """
    counts = []
    for i in range(facet_ids.shape[0]):
        band = facet_ids[i]
        n = int(np.unique(band[band > 0]).size)
        counts.append(n)
    return counts
