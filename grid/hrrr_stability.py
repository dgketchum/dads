"""
Monin-Obukhov Similarity Theory (MOST) wind height correction.

Pure numpy functions — no I/O.  Given skin temperature (from HRRR),
2-m air temperature, 10-m wind speed, roughness length, and sensor
height, compute stability-corrected wind adjustment factors using the
Businger-Dyer formulation.

References
----------
Paulson (1970) — unstable psi_m
Businger et al. (1971) — flux-profile relationships
Dyer (1974) — universal functions
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VONKARMAN = 0.4
GRAVITY = 9.81
Z0_FLOOR = 0.001  # m — minimum roughness length
U_FLOOR = 0.1  # m/s — minimum wind speed (stabilises Ri_b)
FACTOR_LO = 0.5  # clamp bounds for the correction factor
FACTOR_HI = 3.0
MIN_VALID_HOURS = 4  # need at least this many hours for daily mean


# ---------------------------------------------------------------------------
# Bulk Richardson number
# ---------------------------------------------------------------------------


def bulk_richardson(
    t_2m_k: np.ndarray,
    t_skin_k: np.ndarray,
    u_10m: np.ndarray,
    z_ref: float = 10.0,
) -> np.ndarray:
    """Bulk Richardson number from 2-m temperature, skin temperature, and 10-m wind.

    Ri_b = g * z_ref * (t_2m - t_skin) / (t_mean * u^2)

    Positive Ri_b → stable, negative → unstable.
    """
    t_2m_k = np.asarray(t_2m_k, dtype=np.float64)
    t_skin_k = np.asarray(t_skin_k, dtype=np.float64)
    u_10m = np.asarray(u_10m, dtype=np.float64)

    u_safe = np.maximum(u_10m, U_FLOOR)
    t_mean = 0.5 * (t_2m_k + t_skin_k)

    return GRAVITY * z_ref * (t_2m_k - t_skin_k) / (t_mean * u_safe**2)


# ---------------------------------------------------------------------------
# Ri_b → Obukhov length
# ---------------------------------------------------------------------------


def ri_to_obukhov_length(
    ri_b: np.ndarray,
    z_ref: float = 10.0,
) -> np.ndarray:
    """Approximate Obukhov length L from bulk Richardson number.

    Uses the Li (2010) approximation:
      stable   (Ri_b > 0):  L = z_ref / Ri_b  (capped at Ri_b < 0.2)
      unstable (Ri_b < 0):  L = z_ref / Ri_b
      neutral  (Ri_b ≈ 0):  L = very large
    """
    ri_b = np.asarray(ri_b, dtype=np.float64)
    # Cap stable Ri to avoid singularity
    ri_safe = np.where(ri_b > 0.19, 0.19, ri_b)
    # Avoid division by zero
    ri_safe = np.where(np.abs(ri_safe) < 1e-6, 1e-6, ri_safe)
    return z_ref / ri_safe


# ---------------------------------------------------------------------------
# Stability correction function ψ_m
# ---------------------------------------------------------------------------


def psi_m_businger_dyer(zeta: np.ndarray) -> np.ndarray:
    """Stability correction for momentum ψ_m(ζ) where ζ = z/L.

    Unstable (ζ < 0): Paulson (1970)
      x = (1 - 16ζ)^0.25
      ψ_m = 2*ln((1+x)/2) + ln((1+x²)/2) - 2*arctan(x) + π/2

    Stable (ζ > 0): Businger-Dyer
      ψ_m = -5ζ  (capped at ζ = 1)
    """
    zeta = np.asarray(zeta, dtype=np.float64)
    psi = np.zeros_like(zeta)

    # Stable
    stable = zeta > 0
    zeta_s = np.minimum(zeta, 1.0)  # cap at ζ = 1
    psi = np.where(stable, -5.0 * zeta_s, psi)

    # Unstable
    unstable = zeta < 0
    x = np.where(unstable, (1.0 - 16.0 * zeta) ** 0.25, 1.0)
    psi_u = (
        2.0 * np.log((1.0 + x) / 2.0)
        + np.log((1.0 + x**2) / 2.0)
        - 2.0 * np.arctan(x)
        + np.pi / 2.0
    )
    psi = np.where(unstable, psi_u, psi)

    return psi


# ---------------------------------------------------------------------------
# Full MOST wind factor
# ---------------------------------------------------------------------------


def most_wind_factor(
    t_2m_k: np.ndarray,
    t_skin_k: np.ndarray,
    u_10m: np.ndarray,
    z0: np.ndarray,
    zw: np.ndarray,
) -> np.ndarray:
    """MOST stability-corrected wind height factor.

    factor = [ln(10/z0) - ψ_m(10/L)] / [ln(zw/z0) - ψ_m(zw/L)]

    Multiply observed wind at height zw by this factor to get
    estimated 10-m wind.

    Parameters
    ----------
    t_2m_k : 2-m air temperature [K]
    t_skin_k : skin temperature [K]
    u_10m : 10-m wind speed [m/s] (from RTMA or first guess)
    z0 : roughness length [m]
    zw : sensor height [m]
    """
    z0 = np.maximum(np.asarray(z0, dtype=np.float64), Z0_FLOOR)
    zw = np.asarray(zw, dtype=np.float64)

    # Skip correction for 10-m sensors
    is_10m = np.abs(zw - 10.0) < 0.1
    zw_safe = np.where(is_10m, 10.0, zw)

    ri_b = bulk_richardson(t_2m_k, t_skin_k, u_10m)
    L = ri_to_obukhov_length(ri_b)

    zeta_10 = 10.0 / L
    zeta_zw = zw_safe / L

    psi_10 = psi_m_businger_dyer(zeta_10)
    psi_zw = psi_m_businger_dyer(zeta_zw)

    ln_10_z0 = np.log(10.0 / z0)
    ln_zw_z0 = np.log(zw_safe / z0)

    numerator = ln_10_z0 - psi_10
    denominator = ln_zw_z0 - psi_zw

    # Guard against zero/negative denominator
    denom_safe = np.where(np.abs(denominator) < 0.01, 0.01, denominator)

    factor = numerator / denom_safe
    factor = np.clip(factor, FACTOR_LO, FACTOR_HI)

    # 10-m sensors → factor = 1.0
    factor = np.where(is_10m, 1.0, factor)

    return factor


# ---------------------------------------------------------------------------
# Daily mean factor
# ---------------------------------------------------------------------------


def daily_mean_most_factor(
    hourly_t_skin_k: np.ndarray,
    daily_t_2m_k: float,
    daily_u_10m: float,
    z0: float,
    zw: float,
) -> tuple[float, int]:
    """Compute daily-mean MOST correction factor from hourly skin temps.

    Parameters
    ----------
    hourly_t_skin_k : up to 24 hourly skin temperatures [K]; NaN = missing hour
    daily_t_2m_k : daily mean 2-m air temperature [K]
    daily_u_10m : daily mean 10-m wind speed [m/s]
    z0 : roughness length [m]
    zw : sensor height [m]

    Returns
    -------
    (factor, n_valid) : mean factor and number of valid hours.
        If n_valid < MIN_VALID_HOURS, falls back to FAO-56 neutral factor.
    """
    hourly = np.asarray(hourly_t_skin_k, dtype=np.float64)
    valid = np.isfinite(hourly)
    n_valid = int(valid.sum())

    if n_valid < MIN_VALID_HOURS:
        return _fao56_neutral_factor(zw), n_valid

    t_skin_valid = hourly[valid]
    t_2m = np.full(n_valid, daily_t_2m_k)
    u_10m = np.full(n_valid, max(daily_u_10m, U_FLOOR))
    z0_arr = np.full(n_valid, z0)
    zw_arr = np.full(n_valid, zw)

    factors = most_wind_factor(t_2m, t_skin_valid, u_10m, z0_arr, zw_arr)
    return float(np.mean(factors)), n_valid


def _fao56_neutral_factor(zw: float) -> float:
    """FAO-56 neutral log-law factor: ln(67.8*10 - 5.42) / ln(67.8*zw - 5.42)."""
    if abs(zw - 10.0) < 0.1:
        return 1.0
    ln_10m = np.log(67.8 * 10 - 5.42)
    return float(ln_10m / np.log(67.8 * zw - 5.42))
