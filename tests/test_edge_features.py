"""Tests for DA edge-v1 feature computations: upwind_cos, facet_align_12km, delta_eth."""

from __future__ import annotations

import numpy as np
import pytest


# ── facet_align_12km ──────────────────────────────────────────────────────


def _facet_align(q_sin: float, q_cos: float, s_sin: float, s_cos: float) -> float:
    """Reproduce the facet alignment computation from build_da_graphs.py."""
    dot = q_sin * s_sin + q_cos * s_cos
    q_mag = np.sqrt(q_sin**2 + q_cos**2)
    s_mag = np.sqrt(s_sin**2 + s_cos**2)
    denom = q_mag * s_mag
    return float(dot / denom) if denom > 1e-8 else 0.0


def test_facet_align_identical():
    """Same facet orientation → cosine similarity = 1."""
    assert _facet_align(0.707, 0.707, 0.707, 0.707) == pytest.approx(1.0, abs=0.01)


def test_facet_align_opposite():
    """Opposite facet orientations → cosine similarity = -1."""
    # N (sin=0, cos=1) vs S (sin=0, cos=-1)
    assert _facet_align(0.0, 1.0, 0.0, -1.0) == pytest.approx(-1.0, abs=0.01)


def test_facet_align_orthogonal():
    """Orthogonal facet orientations → cosine similarity = 0."""
    # N (sin=0, cos=1) vs E (sin=1, cos=0)
    assert _facet_align(0.0, 1.0, 1.0, 0.0) == pytest.approx(0.0, abs=0.01)


def test_facet_align_flat_query():
    """Flat query (sin=0, cos=0) → alignment = 0."""
    assert _facet_align(0.0, 0.0, 0.707, 0.707) == 0.0


def test_facet_align_flat_source():
    """Flat source (sin=0, cos=0) → alignment = 0."""
    assert _facet_align(0.707, 0.707, 0.0, 0.0) == 0.0


def test_facet_align_both_flat():
    """Both flat → alignment = 0."""
    assert _facet_align(0.0, 0.0, 0.0, 0.0) == 0.0


# ── upwind_cos ────────────────────────────────────────────────────────────


def _upwind_cos(wdir_deg: float, bearing_rad: float) -> float:
    """Reproduce the upwind_cos computation from build_da_graphs.py."""
    wind_from_rad = np.radians(wdir_deg)
    return float(np.cos(wind_from_rad - bearing_rad))


def test_upwind_cos_aligned():
    """Source directly upwind of query → upwind_cos = 1.

    Wind from north (wdir=0°), source is due north of query (bearing=0°).
    cos(0 - 0) = 1.
    """
    assert _upwind_cos(0.0, 0.0) == pytest.approx(1.0, abs=0.01)


def test_upwind_cos_crosswind():
    """Source perpendicular to wind → upwind_cos = 0.

    Wind from north (wdir=0°), source is due east (bearing=π/2).
    cos(0 - π/2) = 0.
    """
    assert _upwind_cos(0.0, np.pi / 2) == pytest.approx(0.0, abs=0.01)


def test_upwind_cos_downwind():
    """Source directly downwind → upwind_cos = -1.

    Wind from north (wdir=0°), source is due south (bearing=π).
    cos(0 - π) = -1.
    """
    assert _upwind_cos(0.0, np.pi) == pytest.approx(-1.0, abs=0.01)


def test_upwind_cos_sw_wind():
    """Wind from SW (225°), source to NE (bearing=45°=π/4).

    Wind-from is 225° = 5π/4 rad. Bearing is π/4.
    cos(5π/4 - π/4) = cos(π) = -1. Source is downwind.
    """
    assert _upwind_cos(225.0, np.pi / 4) == pytest.approx(-1.0, abs=0.01)


def test_upwind_cos_range():
    """upwind_cos should always be in [-1, 1]."""
    for wdir in range(0, 360, 30):
        for bearing in np.linspace(-np.pi, np.pi, 12):
            val = _upwind_cos(float(wdir), bearing)
            assert -1.0 <= val <= 1.0


# ── delta_effective_terrain_height ────────────────────────────────────────


def test_delta_eth_sign():
    """delta_eth = query_eth - source_eth. If query is on a taller barrier,
    delta should be positive."""
    q_eth, s_eth = 500.0, 100.0
    delta = q_eth - s_eth
    assert delta == 400.0


def test_delta_eth_normalization():
    """Normalized delta should be (raw - mean) / std."""
    raw = 400.0
    mean, std = 30.0, 200.0
    normed = (raw - mean) / std
    assert normed == pytest.approx(1.85, abs=0.01)
