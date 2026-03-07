"""Shared HRRR S3 byte-range fetch and IDX parsing utilities.

Used by both ``download_hrrr_archive`` (bulk 9-var download) and
``download_hrrr_stability`` (skin-temperature / MOST factor extraction).
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from datetime import date

BUCKET = "noaa-hrrr-bdp-pds"
BASE_URL = f"https://{BUCKET}.s3.amazonaws.com"
ARCHIVE_START = date(2014, 9, 30)

MAX_RETRIES = 3

_VERSION_BOUNDARIES = [
    (date(2020, 12, 2), "v4"),
    (date(2018, 7, 12), "v3"),
    (date(2016, 8, 23), "v2"),
    (date(2014, 9, 30), "v1"),
]


def version_for_date(d: date) -> str:
    """Return the HRRR version string for a given date."""
    for cutoff, ver in _VERSION_BOUNDARIES:
        if d >= cutoff:
            return ver
    return "v1"


def s3_prefix(d: date, hour: int) -> str:
    return f"hrrr.{d:%Y%m%d}/conus/hrrr.t{hour:02d}z.wrfsfcf00.grib2"


def grib_url(d: date, hour: int) -> str:
    return f"{BASE_URL}/{s3_prefix(d, hour)}"


def idx_url(d: date, hour: int) -> str:
    return f"{BASE_URL}/{s3_prefix(d, hour)}.idx"


def fetch_idx(d: date, hour: int) -> str | None:
    """Fetch the .idx sidecar file content.  Returns None on 404."""
    url = idx_url(d, hour)
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(2**attempt)
        except Exception:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(2**attempt)
    return None


def parse_idx_for_var(idx_text: str, var: str, level: str) -> tuple[int, int] | None:
    """Parse IDX text to find byte range for a given variable and level.

    Returns (start_byte, end_byte) or None if not found.
    end_byte is -1 if it's the last message.
    """
    lines = [ln.strip() for ln in idx_text.strip().split("\n") if ln.strip()]
    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) >= 5 and parts[3] == var and parts[4] == level:
            start = int(parts[1])
            if i + 1 < len(lines):
                next_parts = lines[i + 1].split(":")
                end = int(next_parts[1]) - 1
            else:
                end = -1
            return start, end
    return None


def fetch_byte_range(url: str, start: int, end: int) -> bytes | None:
    """Fetch a byte range from a URL.  Returns None on failure."""
    if end == -1:
        range_header = f"bytes={start}-"
    else:
        range_header = f"bytes={start}-{end}"

    req = urllib.request.Request(url, headers={"Range": range_header})
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(2**attempt)
        except Exception:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(2**attempt)
    return None
