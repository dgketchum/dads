"""Thin path-resolution layer for geostationary COGs (GridSat-B1 + GOES ABI).

Resolves which source to use for a given date and returns paths to COG files
on disk. No data processing — just path logic.
"""

from __future__ import annotations

import os
from datetime import date

from extract.rs.goes_abi.goes_abi_common import BAND_MAP, HOURS_3H

CHANNELS = list(BAND_MAP.keys())

GRIDSAT_ROOT = "/nas/dads/rs/gridsat_b1"
GOES_ROOT = "/nas/dads/rs/goes_abi"
GRIDSAT_END = date(2025, 8, 31)
GOES_START = date(2017, 7, 10)


def resolve_cog_path(
    d: date,
    hour: str,
    channel: str,
    gridsat_root: str = GRIDSAT_ROOT,
    goes_root: str = GOES_ROOT,
    prefer_gridsat: bool = True,
) -> str | None:
    """Return the path to a COG file for a given date/hour/channel, or None."""
    ds_label = d.strftime("%Y%m%d")
    year = d.strftime("%Y")

    gridsat_path = os.path.join(
        gridsat_root, year, f"gridsat_b1_{ds_label}_{hour}00_{channel}.tif"
    )
    goes_path = os.path.join(
        goes_root, year, f"goes_abi_{ds_label}_{hour}00_{channel}.tif"
    )

    if d <= GRIDSAT_END and prefer_gridsat:
        if os.path.exists(gridsat_path):
            return gridsat_path
        if os.path.exists(goes_path):
            return goes_path
        return None

    # Post-GridSat or prefer_gridsat=False: try GOES first
    if os.path.exists(goes_path):
        return goes_path
    if os.path.exists(gridsat_path):
        return gridsat_path
    return None


def resolve_day_paths(
    d: date,
    gridsat_root: str = GRIDSAT_ROOT,
    goes_root: str = GOES_ROOT,
) -> dict[tuple[str, str], str]:
    """Return available COG paths for all hour/channel slots in a day.

    Returns up to 24 entries (8 hours x 3 channels), omitting missing files.
    """
    paths: dict[tuple[str, str], str] = {}
    for hour in HOURS_3H:
        for channel in CHANNELS:
            p = resolve_cog_path(d, hour, channel, gridsat_root, goes_root)
            if p is not None:
                paths[(hour, channel)] = p
    return paths


def source_for_date(d: date) -> str:
    """Return which source covers a date: 'gridsat', 'goes', or 'overlap'."""
    if d >= GOES_START and d <= GRIDSAT_END:
        return "overlap"
    if d <= GRIDSAT_END:
        return "gridsat"
    return "goes"
