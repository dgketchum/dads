"""Shared constants and S3 helpers for GOES-R ABI MCMIP-C downloads.

Provides bucket names, band mapping (GOES ABI → GridSat canonical channel
names), clip bounds matching the GridSat/URMA+1° extent, and utilities for
listing S3 prefixes and selecting the nearest scan to a target hour.
"""

from __future__ import annotations

import re
from datetime import date, datetime

import boto3
from botocore import UNSIGNED
from botocore.config import Config

# ── S3 buckets ────────────────────────────────────────────────────────────────

BUCKETS: dict[str, str] = {
    "goes16": "noaa-goes16",
    "goes18": "noaa-goes18",
}

PRODUCT = "ABI-L2-MCMIPC"

# ── temporal ──────────────────────────────────────────────────────────────────

HOURS_3H = ["00", "03", "06", "09", "12", "15", "18", "21"]

# ── band mapping (GridSat canonical name → GOES ABI MCMIP-C variable) ────────

BAND_MAP: dict[str, str] = {
    "irwin_cdr": "CMI_C14",  # ~11 µm IR window
    "irwvp": "CMI_C09",  # ~6.7 µm water vapor
    "vschn": "CMI_C02",  # ~0.6 µm visible
}

# ── spatial ───────────────────────────────────────────────────────────────────

# URMA extent + 1° buffer (same as GridSat pipeline)
CLIP_BOUNDS = {
    "lat_min": 18.2,
    "lat_max": 55.4,
    "lon_min": -139.4,
    "lon_max": -58.0,
}

TARGET_RES_DEG = 0.07

# ── satellite selection ───────────────────────────────────────────────────────

GOES18_OPERATIONAL = date(2022, 9, 1)
SAT_LON_SPLIT = -105.0  # GOES-18 west, GOES-16 east


def satellites_for_date(d: date) -> list[str]:
    """Return list of satellite keys to use for a given date."""
    if d < GOES18_OPERATIONAL:
        return ["goes16"]
    return ["goes16", "goes18"]


# ── S3 helpers ────────────────────────────────────────────────────────────────

# Regex to extract scan start time from MCMIP-C filenames.
# Example: OR_ABI-L2-MCMIPC-M6_G16_s20251800300...
_SCAN_START_RE = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})")


def s3_prefix(satellite: str, dt: datetime) -> str:
    """Build an S3 listing prefix for MCMIP-C files at a given hour.

    The GOES S3 layout is:
        {Product}/{YYYY}/{DDD}/{HH}/OR_ABI-L2-MCMIPC-...
    where DDD is the day of year.
    """
    doy = dt.timetuple().tm_yday
    return f"{PRODUCT}/{dt.year}/{doy:03d}/{dt.hour:02d}/"


def _parse_scan_start(key: str) -> datetime | None:
    """Extract scan start datetime from an MCMIP-C S3 key."""
    m = _SCAN_START_RE.search(key)
    if m is None:
        return None
    year, doy, hour, minute = (
        int(m.group(1)),
        int(m.group(2)),
        int(m.group(3)),
        int(m.group(4)),
    )
    return datetime.strptime(f"{year}{doy:03d}{hour:02d}{minute:02d}", "%Y%j%H%M")


def pick_nearest_scan(
    s3_client: boto3.client,
    bucket: str,
    prefix: str,
    target_dt: datetime,
) -> str | None:
    """List MCMIP-C keys under *prefix* and return the one closest to *target_dt*.

    Returns the full S3 key, or None if no files are found.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    best_key: str | None = None
    best_delta: float = float("inf")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            scan_dt = _parse_scan_start(key)
            if scan_dt is None:
                continue
            delta = abs((scan_dt - target_dt).total_seconds())
            if delta < best_delta:
                best_delta = delta
                best_key = key

    return best_key


def make_s3_client() -> boto3.client:
    """Create a boto3 S3 client with unsigned (public) access."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))
