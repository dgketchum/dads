"""Unit tests for PRISM terrain numerics in terrain.prism_topography."""

from __future__ import annotations

import numpy as np
import pytest

from terrain.prism_topography import (
    build_effective_terrain_height,
    build_i3a,
    build_i3c,
    build_i3d,
    circular_footprint,
    circular_minimum,
    classify_aspect,
    effective_terrain_height,
    gaussian_kernel,
    gaussian_smooth,
    horns_slope_aspect,
    label_facets,
    terrain_index_i3c,
    uniform_kernel,
    uniform_smooth,
)


# ---------------------------------------------------------------------------
# Kernel tests
# ---------------------------------------------------------------------------


def test_gaussian_kernel_sums_to_one():
    """Gaussian kernel with any radius should sum to ~1.0."""
    for radius_m in [1000.0, 5000.0, 12000.0]:
        k = gaussian_kernel(radius_m, 1000.0)
        assert k.sum() == pytest.approx(1.0, abs=1e-5), (
            f"Gaussian kernel (r={radius_m}) sum = {k.sum()}, expected ~1.0"
        )


def test_uniform_kernel_sums_to_one():
    """Uniform kernel should sum to ~1.0."""
    for radius_m in [1000.0, 3000.0, 8000.0]:
        k = uniform_kernel(radius_m, 1000.0)
        assert k.sum() == pytest.approx(1.0, abs=1e-5), (
            f"Uniform kernel (r={radius_m}) sum = {k.sum()}, expected ~1.0"
        )


def test_circular_footprint_shape():
    """Footprint should be boolean and roughly circular."""
    fp = circular_footprint(3000.0, 1000.0)
    assert fp.dtype == bool
    # Shape must be square and odd-sized
    assert fp.ndim == 2
    assert fp.shape[0] == fp.shape[1]
    assert fp.shape[0] % 2 == 1
    # Centre pixel must be True
    cy, cx = fp.shape[0] // 2, fp.shape[1] // 2
    assert fp[cy, cx]
    # Corners of the bounding box should be False (circular, not square)
    assert not fp[0, 0]
    assert not fp[0, -1]
    assert not fp[-1, 0]
    assert not fp[-1, -1]


def test_kernels_are_symmetric():
    """Both kernel types should be symmetric around the centre."""
    for radius_m in [2000.0, 5000.0]:
        gk = gaussian_kernel(radius_m, 1000.0)
        uk = uniform_kernel(radius_m, 1000.0)
        # Horizontal, vertical, and diagonal symmetry
        np.testing.assert_allclose(
            gk, gk[::-1, :], atol=1e-7, err_msg="Gaussian not vertically symmetric"
        )
        np.testing.assert_allclose(
            gk, gk[:, ::-1], atol=1e-7, err_msg="Gaussian not horizontally symmetric"
        )
        np.testing.assert_allclose(
            uk, uk[::-1, :], atol=1e-7, err_msg="Uniform not vertically symmetric"
        )
        np.testing.assert_allclose(
            uk, uk[:, ::-1], atol=1e-7, err_msg="Uniform not horizontally symmetric"
        )


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------


def test_gaussian_smooth_preserves_constant():
    """Smoothing a constant field should return the same constant."""
    arr = np.full((100, 100), 42.0, dtype="float32")
    result = gaussian_smooth(arr, 5000.0, 1000.0)
    np.testing.assert_allclose(result[20:-20, 20:-20], 42.0, atol=0.01)


def test_uniform_smooth_preserves_constant():
    """Smoothing a constant field with uniform kernel should return the same constant."""
    arr = np.full((100, 100), 7.5, dtype="float32")
    result = uniform_smooth(arr, 4000.0, 1000.0)
    np.testing.assert_allclose(result[10:-10, 10:-10], 7.5, atol=0.01)


def test_gaussian_smooth_handles_nan():
    """NaN cells should not poison the output at distant pixels."""
    arr = np.ones((50, 50), dtype="float32")
    arr[25, 25] = np.nan
    result = gaussian_smooth(arr, 3000.0, 1000.0)
    assert np.isfinite(result[0, 0])


def test_circular_minimum_finds_valley():
    """On a V-shaped valley, min filter should recover the valley floor."""
    H, W = 100, 100
    _, x = np.mgrid[:H, :W]
    elev = np.abs(x - 50).astype("float32") * 10  # V-shape, min at col 50
    result = circular_minimum(elev, 5000.0, 1000.0)
    # Centre column should be 0.0 (the valley floor)
    assert result[50, 50] == 0.0


# ---------------------------------------------------------------------------
# Slope / aspect tests
# ---------------------------------------------------------------------------


def test_aspect_north_facing():
    """Elevation increases southward → steepest descent is north → aspect ~0."""
    H, W = 50, 50
    y, x = np.mgrid[:H, :W]
    # row 0 = north edge; elev increases with row = increases southward
    # steepest descent toward row 0 = north → aspect = 0
    elev = (y * 10.0).astype("float32")
    slope_deg, aspect_deg = horns_slope_aspect(elev, 1000.0)
    center_aspect = aspect_deg[20:30, 20:30]
    # Allow wrap-around: either < 10° or > 350°
    assert np.all((center_aspect < 10.0) | (center_aspect > 350.0)), (
        f"Expected aspect ~0 (N), got min={center_aspect.min():.1f} max={center_aspect.max():.1f}"
    )


def test_aspect_east_facing():
    """Elevation decreases eastward → steepest descent is east → aspect ~90."""
    H, W = 50, 50
    _, x = np.mgrid[:H, :W]
    # col 0 is highest; elev decreases going east → steepest descent = east
    elev = ((W - 1 - x) * 10.0).astype("float32")
    slope_deg, aspect_deg = horns_slope_aspect(elev, 1000.0)
    center = aspect_deg[20:30, 20:30]
    np.testing.assert_allclose(center, 90.0, atol=5.0)


def test_slope_flat_field():
    """A flat elevation field should produce zero slope everywhere."""
    arr = np.full((30, 30), 500.0, dtype="float32")
    slope_deg, _ = horns_slope_aspect(arr, 1000.0)
    np.testing.assert_allclose(slope_deg, 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Aspect classification tests
# ---------------------------------------------------------------------------


def test_classify_flat():
    """Cells with slope below threshold should be class 0 regardless of aspect."""
    aspect = np.full((10, 10), 45.0, dtype="float32")
    slope = np.full((10, 10), 1.0, dtype="float32")  # below default 2.0 threshold
    result = classify_aspect(aspect, slope)
    assert np.all(result == 0)


def test_classify_cardinal_directions():
    """Known aspect angles should map to the correct cardinal-direction bins."""
    # N=0→1, NE=45→2, E=90→3, SE=135→4, S=180→5, SW=225→6, W=270→7, NW=315→8
    aspects = np.array([[0, 45, 90, 135, 180, 225, 270, 315]], dtype="float32")
    slopes = np.full_like(aspects, 10.0)
    result = classify_aspect(aspects, slopes)
    expected = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype="uint8")
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Connected-component facet labelling
# ---------------------------------------------------------------------------


def test_label_facets_separates_regions():
    """Two disconnected patches of the same class should get different IDs."""
    grid = np.array(
        [
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 3, 0, 1, 1],
            [3, 3, 0, 1, 1],
        ],
        dtype="uint8",
    )
    ids = label_facets(grid)
    # Class 1 patch top-left vs class 3 bottom-left: different classes → different IDs
    assert ids[0, 0] != ids[3, 3]
    # Class 1 top-left vs class 1 bottom-right: same class but not connected
    assert ids[0, 0] != ids[3, 4]
    # Within the top-left class-1 patch: same connected component
    assert ids[0, 0] == ids[1, 1]


def test_label_facets_all_nonzero():
    """Every cell in a fully non-zero grid should receive a positive label."""
    grid = np.ones((5, 5), dtype="uint8")
    ids = label_facets(grid)
    assert np.all(ids > 0)


# ---------------------------------------------------------------------------
# I3c terrain index
# ---------------------------------------------------------------------------


def test_i3c_thresholds():
    """Piecewise function should match PRISM specification at key breakpoints."""
    heights = np.array([0, 50, 75, 100, 162.5, 250, 500], dtype="float32")
    result = terrain_index_i3c(heights)
    expected = np.array([0.0, 0.0, 0.0, 0.142857, 0.5, 1.0, 1.0], dtype="float32")
    np.testing.assert_allclose(result, expected, atol=0.01)


def test_i3c_output_range():
    """I3c must always be in [0, 1]."""
    rng = np.random.default_rng(0)
    heights = rng.uniform(-100, 1000, size=500).astype("float32")
    result = terrain_index_i3c(heights)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


# ---------------------------------------------------------------------------
# Effective terrain height
# ---------------------------------------------------------------------------


def test_effective_terrain_plateau():
    """A flat plateau should have ~0 effective terrain height in the interior."""
    arr = np.full((200, 200), 1000.0, dtype="float32")
    result = effective_terrain_height(arr, 1000.0)
    assert np.abs(result[80:120, 80:120]).max() < 10.0


@pytest.mark.slow
def test_effective_terrain_mountain():
    """A single conical mountain should have large positive height at the peak."""
    H, W = 200, 200
    y, x = np.mgrid[:H, :W]
    r = np.sqrt((x - 100) ** 2 + (y - 100) ** 2).astype("float32")
    elev = np.maximum(0, 2000 - r * 40).astype("float32")
    result = effective_terrain_height(elev, 1000.0)
    assert result[100, 100] > 200, (
        f"Peak effective height = {result[100, 100]:.1f} m, expected > 200 m"
    )


# ---------------------------------------------------------------------------
# Production pipeline: build_effective_terrain_height, build_i3c, build_i3a, build_i3d
# ---------------------------------------------------------------------------


def test_build_eth_plateau():
    """Production ETH on a flat plateau should be near zero."""
    arr = np.full((200, 200), 500.0, dtype="float32")
    eth = build_effective_terrain_height(arr, cell_size_m=1000.0)
    assert np.abs(eth[60:140, 60:140]).max() < 15.0


def test_build_i3c_matches_spec():
    """build_i3c should match the piecewise I3c function."""
    h = np.array([0, 50, 75, 162.5, 250, 500], dtype="float32")
    result = build_i3c(h)
    expected = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0], dtype="float32")
    np.testing.assert_allclose(result, expected, atol=0.01)


def test_build_i3a_uses_eth_not_i3c():
    """I3a should be derived from IDW-smoothed ETH, not smoothed I3c.

    Place a tall mountain (ETH=500m) surrounded by flat terrain (ETH=0).
    A cell 50 km away should get I3a > 0 because the IDW-averaged ETH
    around it includes the tall mountain. If I3a were incorrectly computed
    from smoothed I3c, this test might still pass, but the values would
    differ — this test verifies the sign and rough magnitude.
    """
    H, W = 250, 250
    eth = np.zeros((H, W), dtype="float32")
    # Place a 20-pixel-radius mountain in the center with ETH=500m
    y, x = np.mgrid[:H, :W]
    r = np.sqrt((x - 125) ** 2 + (y - 125) ** 2)
    eth[r < 20] = 500.0

    i3a = build_i3a(eth, cell_size_m=1000.0, support_radius_m=100_000.0)
    # Center of mountain: IDW mean pulls toward 500m but the 100km
    # neighborhood is mostly zero, so h_a is modest. After thresholding
    # at (h_a - 75)/175, expect a positive but not saturated value.
    assert i3a[125, 125] > 0.05, f"I3a at mountain center = {i3a[125, 125]}"
    # I3a should decrease with distance from the mountain
    assert i3a[125, 125] > i3a[125, 200], "I3a should decay with distance"
    # Far corner: should be 0 (beyond support or h_a < h2)
    assert i3a[0, 0] == 0.0


def test_build_i3d_is_max():
    """I3d = max(I3c, I3a)."""
    i3c = np.array([[0.0, 0.5], [1.0, 0.3]], dtype="float32")
    i3a = np.array([[0.2, 0.4], [0.8, 0.6]], dtype="float32")
    i3d = build_i3d(i3c, i3a)
    expected = np.array([[0.2, 0.5], [1.0, 0.6]], dtype="float32")
    np.testing.assert_allclose(i3d, expected)
