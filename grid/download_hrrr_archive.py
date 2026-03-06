"""
Bulk download HRRR surface fields (fxx=0) from AWS S3 via byte-range fetch.

Downloads 9 target surface variables per hour by parsing the IDX sidecar,
fetching individual GRIB messages via HTTP Range requests, and concatenating
them into a single multi-message GRIB2 file on disk (~15 MB/hour).

Layout
------
  {dest}/
    v1/{YYYY}/{YYYYMMDD}/hrrr.t{HH}z.9var.grib2   (2014-11-15 to 2016-08-22)
    v2/{YYYY}/{YYYYMMDD}/hrrr.t{HH}z.9var.grib2   (2016-08-23 to 2018-07-11)
    v3/{YYYY}/{YYYYMMDD}/hrrr.t{HH}z.9var.grib2   (2018-07-12 to 2020-12-01)
    v4/{YYYY}/{YYYYMMDD}/hrrr.t{HH}z.9var.grib2   (2020-12-02 to present)
    manifest.parquet
    README.md

Usage
-----
  uv run python -m grid.download_hrrr_archive \\
      --dest /mnt/mco_nas1/shared/hrrr_hourly \\
      --workers 20 \\
      --start-date 2014-11-15 --end-date 2025-01-08 \\
      --schedule "00:00-08:00" --weekend-free
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import pandas as pd

# ── constants ────────────────────────────────────────────────────────────────

BUCKET = "noaa-hrrr-bdp-pds"
BASE_URL = f"https://{BUCKET}.s3.amazonaws.com"
ARCHIVE_START = date(2014, 9, 30)

_HOURS = list(range(24))
_MAX_RETRIES = 3

TARGET_FIELDS = [
    ("TMP", "2 m above ground"),
    ("DPT", "2 m above ground"),
    ("UGRD", "10 m above ground"),
    ("VGRD", "10 m above ground"),
    ("DSWRF", "surface"),
    ("PRES", "surface"),
    ("TCDC", "entire atmosphere"),
    ("HPBL", "surface"),
    ("SPFH", "2 m above ground"),
]

_VERSION_BOUNDARIES = [
    (date(2020, 12, 2), "v4"),
    (date(2018, 7, 12), "v3"),
    (date(2016, 8, 23), "v2"),
    (date(2014, 9, 30), "v1"),
]


def _version_for_date(d: date) -> str:
    for cutoff, ver in _VERSION_BOUNDARIES:
        if d >= cutoff:
            return ver
    return "v1"


# ── URL helpers ──────────────────────────────────────────────────────────────


def _s3_prefix(d: date, hour: int) -> str:
    return f"hrrr.{d:%Y%m%d}/conus/hrrr.t{hour:02d}z.wrfsfcf00.grib2"


def _grib_url(d: date, hour: int) -> str:
    return f"{BASE_URL}/{_s3_prefix(d, hour)}"


def _idx_url(d: date, hour: int) -> str:
    return f"{BASE_URL}/{_s3_prefix(d, hour)}.idx"


def _local_path(dest: str, d: date, hour: int) -> str:
    ver = _version_for_date(d)
    return os.path.join(
        dest, ver, f"{d:%Y}", f"{d:%Y%m%d}", f"hrrr.t{hour:02d}z.9var.grib2"
    )


# ── IDX parsing + byte-range fetch ──────────────────────────────────────────


def _fetch_idx(d: date, hour: int) -> str | None:
    """Fetch the .idx sidecar file content.  Returns None on 404."""
    url = _idx_url(d, hour)
    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == _MAX_RETRIES - 1:
                return None
            time.sleep(2**attempt)
        except Exception:
            if attempt == _MAX_RETRIES - 1:
                return None
            time.sleep(2**attempt)
    return None


def _parse_idx_for_var(idx_text: str, var: str, level: str) -> tuple[int, int] | None:
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


def _fetch_byte_range(url: str, start: int, end: int) -> bytes | None:
    """Fetch a byte range from a URL.  Returns None on failure."""
    if end == -1:
        range_header = f"bytes={start}-"
    else:
        range_header = f"bytes={start}-{end}"

    req = urllib.request.Request(url, headers={"Range": range_header})
    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == _MAX_RETRIES - 1:
                return None
            time.sleep(2**attempt)
        except Exception:
            if attempt == _MAX_RETRIES - 1:
                return None
            time.sleep(2**attempt)
    return None


# ── scheduling ───────────────────────────────────────────────────────────────


def _parse_schedule(schedule: str) -> tuple[int, int]:
    """Parse 'HH:MM-HH:MM' into (start_minutes, end_minutes) from midnight."""
    start_s, end_s = schedule.split("-")
    sh, sm = start_s.strip().split(":")
    eh, em = end_s.strip().split(":")
    return int(sh) * 60 + int(sm), int(eh) * 60 + int(em)


def _in_window(schedule: str | None, weekend_free: bool) -> bool:
    if schedule is None:
        return True
    now = datetime.now()
    if weekend_free and now.weekday() >= 5:
        return True
    start_m, end_m = _parse_schedule(schedule)
    cur_m = now.hour * 60 + now.minute
    if start_m <= end_m:
        return start_m <= cur_m < end_m
    return cur_m >= start_m or cur_m < end_m


def _seconds_until_window(schedule: str, weekend_free: bool) -> float:
    """Seconds to sleep before the next download window opens."""
    now = datetime.now()
    start_m, _ = _parse_schedule(schedule)
    start_time = datetime.combine(now.date(), datetime.min.time()) + timedelta(
        minutes=start_m
    )
    if start_time <= now:
        start_time += timedelta(days=1)
    wait_weeknight = (start_time - now).total_seconds()

    if not weekend_free:
        return wait_weeknight

    for day_offset in range(1, 7):
        future = now + timedelta(days=day_offset)
        if future.weekday() >= 5:
            target = datetime.combine(future.date(), datetime.min.time())
            wait_weekend = (target - now).total_seconds()
            return min(wait_weeknight, wait_weekend)

    return wait_weeknight


def _wait_for_window(schedule: str | None, weekend_free: bool) -> None:
    if schedule is None or _in_window(schedule, weekend_free):
        return
    wait = _seconds_until_window(schedule, weekend_free)
    resume_at = datetime.now() + timedelta(seconds=wait)
    print(f"  Outside schedule — sleeping until {resume_at:%Y-%m-%d %H:%M}")
    sys.stdout.flush()
    time.sleep(wait)


# ── per-hour download ────────────────────────────────────────────────────────


def _download_hour(d: date, hour: int, dest: str) -> dict:
    """Fetch 9 target fields for one hour via byte-range, write concatenated GRIB2.

    Returns a manifest row dict.
    """
    grib_url = _grib_url(d, hour)
    dest_path = _local_path(dest, d, hour)
    ds = d.isoformat()
    ver = _version_for_date(d)

    base_row = {
        "model": "hrrr",
        "date": ds,
        "hour": hour,
        "product": "sfc9",
        "version": ver,
        "s3_url": grib_url,
        "local_path": dest_path,
    }

    # Fetch IDX
    idx_text = _fetch_idx(d, hour)
    if idx_text is None:
        return {
            **base_row,
            "n_fields": 0,
            "size_bytes": 0,
            "status": "missing",
            "downloaded_at": datetime.now().isoformat(),
        }

    # Fetch each target field
    chunks: list[bytes] = []
    n_found = 0
    for var, level in TARGET_FIELDS:
        rng = _parse_idx_for_var(idx_text, var, level)
        if rng is None:
            continue
        data = _fetch_byte_range(grib_url, rng[0], rng[1])
        if data is not None:
            chunks.append(data)
            n_found += 1

    if n_found == 0:
        return {
            **base_row,
            "n_fields": 0,
            "size_bytes": 0,
            "status": "error",
            "downloaded_at": datetime.now().isoformat(),
        }

    # Write concatenated GRIB2
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in chunks:
            f.write(chunk)

    size = os.path.getsize(dest_path)
    status = "done" if n_found == len(TARGET_FIELDS) else "partial"

    return {
        **base_row,
        "n_fields": n_found,
        "size_bytes": size,
        "status": status,
        "downloaded_at": datetime.now().isoformat(),
    }


# ── manifest ─────────────────────────────────────────────────────────────────

_MANIFEST_COLS = [
    "model",
    "date",
    "hour",
    "product",
    "version",
    "s3_url",
    "local_path",
    "n_fields",
    "size_bytes",
    "status",
    "downloaded_at",
]


def _load_manifest(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame(columns=_MANIFEST_COLS)


def _save_manifest(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)


def _done_keys(manifest: pd.DataFrame) -> set[tuple[str, int]]:
    """Set of (date, hour) already completed or missing."""
    if manifest.empty:
        return set()
    done = manifest[manifest["status"].isin(("done", "missing"))]
    return set(zip(done["date"], done["hour"]))


# ── signal handling ──────────────────────────────────────────────────────────

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n  Shutdown requested — finishing current batch ...")
    sys.stdout.flush()


# ── README ───────────────────────────────────────────────────────────────────

_README = """\
# HRRR Hourly Surface Fields Archive

Selected CONUS 3 km HRRR surface analysis fields (fxx=0) downloaded from the
NOAA public AWS bucket via byte-range fetch.

## Source

| Bucket | Archive start | Grid |
|--------|--------------|------|
| `noaa-hrrr-bdp-pds` | 2014-09-30 | CONUS 3 km Lambert Conformal |

## Layout

```
hrrr_hourly/
  v1/{YYYY}/{YYYYMMDD}/hrrr.t{HH}z.9var.grib2   (2014-11-15 to 2016-08-22)
  v2/{YYYY}/{YYYYMMDD}/hrrr.t{HH}z.9var.grib2   (2016-08-23 to 2018-07-11)
  v3/{YYYY}/{YYYYMMDD}/hrrr.t{HH}z.9var.grib2   (2018-07-12 to 2020-12-01)
  v4/{YYYY}/{YYYYMMDD}/hrrr.t{HH}z.9var.grib2   (2020-12-02 to present)
  manifest.parquet
  README.md
```

Each `.9var.grib2` file is the concatenation of 9 individual GRIB2 messages
fetched via HTTP Range requests from the full `wrfsfcf00.grib2` file.  This is
a valid multi-message GRIB2 file readable by eccodes and cfgrib.

24 files per day (hours 00-23), ~15 MB per file.

## HRRR Version History

| Version | Date Range | Key Changes | Sfc Fields |
|---------|-----------|-------------|------------|
| v1 | 2014-09-30 to 2016-08-22 | Initial 3 km CONUS HRRR | ~102 |
| v2 | 2016-08-23 to 2018-07-11 | HRRRv2: improved physics, smoke | ~102 |
| v3 | 2018-07-12 to 2020-12-01 | HRRRv3: Thompson aerosol-aware microphysics | 132-148 |
| v4 | 2020-12-02 to present | HRRRv4: RAP/HRRR unification, RRFS prep | ~170 |

## Extracted Variables (9)

| Variable | Level | Description |
|----------|-------|-------------|
| TMP | 2 m above ground | 2-m air temperature (K) |
| DPT | 2 m above ground | 2-m dewpoint temperature (K) |
| UGRD | 10 m above ground | 10-m U-wind component (m/s) |
| VGRD | 10 m above ground | 10-m V-wind component (m/s) |
| DSWRF | surface | Downward shortwave radiation flux (W/m2) |
| PRES | surface | Surface pressure (Pa) |
| TCDC | entire atmosphere | Total cloud cover (%) |
| HPBL | surface | Planetary boundary layer height (m) |
| SPFH | 2 m above ground | 2-m specific humidity (kg/kg) |

## Field Availability by Version

All 9 fields are present in v3 and v4.  In v1/v2, HPBL and SPFH may be absent
from some files, resulting in `status="partial"` in the manifest.

## Storage Estimate

| Scenario | Estimate |
|----------|----------|
| ~15 MB/hour x 24 hours x 365 days x 10 years | ~1.3 TB |

## Script

`grid/download_hrrr_archive.py` in the `dads-mvp` repo.
"""


# ── main ─────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bulk download HRRR surface fields from AWS S3."
    )
    p.add_argument("--dest", required=True, help="Root destination directory.")
    p.add_argument(
        "--workers", type=int, default=20, help="Concurrent download threads."
    )
    p.add_argument(
        "--start-date",
        default=None,
        help="Start date (YYYY-MM-DD). Default: archive start.",
    )
    p.add_argument(
        "--end-date", default=None, help="End date (YYYY-MM-DD). Default: yesterday."
    )
    p.add_argument(
        "--schedule",
        default=None,
        help='Download time window, e.g. "00:00-08:00". Unrestricted if omitted.',
    )
    p.add_argument(
        "--weekend-free",
        action="store_true",
        help="Allow unrestricted downloads on Saturday and Sunday.",
    )
    p.add_argument(
        "--oldest-first",
        action="store_true",
        help="Download oldest dates first (default: most recent first).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    yesterday = date.today() - timedelta(days=1)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Write README on first run
    readme_path = os.path.join(args.dest, "README.md")
    if not os.path.exists(readme_path):
        os.makedirs(args.dest, exist_ok=True)
        with open(readme_path, "w") as f:
            f.write(_README)

    manifest_path = os.path.join(args.dest, "manifest.parquet")
    manifest = _load_manifest(manifest_path)
    done = _done_keys(manifest)
    new_rows: list[dict] = []

    start = date.fromisoformat(args.start_date) if args.start_date else ARCHIVE_START
    end = date.fromisoformat(args.end_date) if args.end_date else yesterday

    # Build day list
    days: list[date] = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    if not args.oldest_first:
        days.reverse()

    total_downloaded = 0
    total_bytes = 0
    t_start = time.time()

    print(f"\n{'=' * 60}")
    print(f"  HRRR: {start} → {end}  ({len(days)} days)")
    print(f"  Workers: {args.workers}  Schedule: {args.schedule or 'unrestricted'}")
    print(f"  Already done: {len(done)} file-hours")
    print(f"{'=' * 60}\n")
    sys.stdout.flush()

    for day_i, day in enumerate(days):
        if _shutdown_requested:
            break

        _wait_for_window(args.schedule, args.weekend_free)
        if _shutdown_requested:
            break

        # Build hour tasks for this day (skip done keys)
        ds = day.isoformat()
        tasks: list[tuple[date, int]] = []
        for h in _HOURS:
            if (ds, h) in done:
                continue
            lp = _local_path(args.dest, day, h)
            if os.path.exists(lp) and os.path.getsize(lp) > 0:
                done.add((ds, h))
                continue
            tasks.append((day, h))

        if not tasks:
            continue

        day_done = 0
        day_bytes = 0
        day_t0 = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_download_hour, d, h, args.dest): (d, h) for d, h in tasks
            }
            for fut in as_completed(futures):
                result = fut.result()
                new_rows.append(result)
                done.add((result["date"], result["hour"]))
                if result["status"] == "done":
                    day_done += 1
                    day_bytes += result["size_bytes"]
                elif result["status"] == "partial":
                    day_done += 1
                    day_bytes += result["size_bytes"]

        total_downloaded += day_done
        total_bytes += day_bytes
        elapsed = time.time() - day_t0
        rate = day_bytes / elapsed / 1e6 if elapsed > 0 else 0
        total_elapsed = time.time() - t_start

        print(
            f"  {day}  {day_done}/{len(tasks)} hours  "
            f"{day_bytes / 1e6:.0f} MB  {rate:.0f} MB/s  "
            f"[total: {total_downloaded} hours, {total_bytes / 1e9:.1f} GB, "
            f"{total_elapsed / 3600:.1f}h]"
        )
        sys.stdout.flush()

        # Flush manifest every 10 days
        if (day_i + 1) % 10 == 0 and new_rows:
            manifest = pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True)
            new_rows.clear()
            _save_manifest(manifest, manifest_path)

    # Final manifest save
    if new_rows:
        manifest = pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True)
    _save_manifest(manifest, manifest_path)

    elapsed_h = (time.time() - t_start) / 3600
    print(
        f"\n  Done: {total_downloaded} hours, {total_bytes / 1e9:.1f} GB in {elapsed_h:.1f}h"
    )
    if _shutdown_requested:
        print("  (interrupted — resume by re-running the same command)")


if __name__ == "__main__":
    main()
