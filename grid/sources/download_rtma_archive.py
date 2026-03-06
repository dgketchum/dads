"""
Bulk download RTMA / URMA hourly analysis files from AWS S3.

Downloads full analysis GRIB2 files (all variables) with threaded I/O,
manifest-based resume, and time-window scheduling so downloads can be
restricted to off-hours.  Optionally downloads hourly precipitation
analysis files (``--include-pcp``).

Layout
------
  {dest}/{model}/{YYYY}/{YYYYMMDD}/{model}2p5.tHHz.2dvaranl_ndfd.{suffix}
  {dest}/{model}/{YYYY}/{YYYYMMDD}/{model}2p5.YYYYMMDDHH.pcp*.grb2   (if --include-pcp)
  {dest}/manifest.parquet

Usage
-----
  uv run python -m grid.sources.download_rtma_archive \
      --model urma \
      --dest /mnt/mco_nas1/shared/rtma_hourly \
      --workers 20 --include-pcp \
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

_BUCKETS: dict[str, str] = {
    "rtma": "noaa-rtma-pds",
    "urma": "noaa-urma-pds",
}

# Earliest known date on AWS (from probe results)
_ARCHIVE_START: dict[str, date] = {
    "rtma": date(2013, 3, 19),
    "urma": date(2014, 1, 28),
}

_HOURS = list(range(24))
_MAX_RETRIES = 3

# Earliest date with GRIB2-format precipitation files on S3.
_PCP_START: dict[str, date] = {
    "rtma": date(2020, 1, 1),
    "urma": date(2020, 1, 1),
}


# ── suffix / URL helpers ────────────────────────────────────────────────────


def _suffix_for_date(d: date) -> str:
    if d < date(2014, 2, 1):
        return ".grb2"
    if d < date(2017, 5, 2):
        return ".grb2_ext"
    return ".grb2_wexp"


def _s3_url(model: str, d: date, hour: int) -> str:
    suffix = _suffix_for_date(d)
    pfx = f"{model}2p5"
    return (
        f"https://{_BUCKETS[model]}.s3.amazonaws.com/"
        f"{pfx}.{d:%Y%m%d}/{pfx}.t{hour:02d}z.2dvaranl_ndfd{suffix}"
    )


def _local_path(dest: str, model: str, d: date, hour: int) -> str:
    suffix = _suffix_for_date(d)
    pfx = f"{model}2p5"
    fname = f"{pfx}.t{hour:02d}z.2dvaranl_ndfd{suffix}"
    return os.path.join(dest, model, f"{d:%Y}", f"{d:%Y%m%d}", fname)


def _pcp_s3_url(model: str, d: date, hour: int) -> str:
    """S3 URL for the hourly precipitation analysis GRIB2."""
    pfx = f"{model}2p5"
    bucket = _BUCKETS[model]
    dh = f"{d:%Y%m%d}{hour:02d}"
    if model == "rtma":
        fname = f"{pfx}.{dh}.pcp.184.grb2"
    else:
        fname = f"{pfx}.{dh}.pcp_01h.wexp.grb2"
    return f"https://{bucket}.s3.amazonaws.com/{pfx}.{d:%Y%m%d}/{fname}"


def _pcp_local_path(dest: str, model: str, d: date, hour: int) -> str:
    pfx = f"{model}2p5"
    dh = f"{d:%Y%m%d}{hour:02d}"
    if model == "rtma":
        fname = f"{pfx}.{dh}.pcp.184.grb2"
    else:
        fname = f"{pfx}.{dh}.pcp_01h.wexp.grb2"
    return os.path.join(dest, model, f"{d:%Y}", f"{d:%Y%m%d}", fname)


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
    # crosses midnight
    return cur_m >= start_m or cur_m < end_m


def _seconds_until_window(schedule: str, weekend_free: bool) -> float:
    """Seconds to sleep before the next download window opens."""
    now = datetime.now()

    # Next scheduled weeknight window
    start_m, _ = _parse_schedule(schedule)
    start_time = datetime.combine(now.date(), datetime.min.time()) + timedelta(
        minutes=start_m
    )
    if start_time <= now:
        start_time += timedelta(days=1)
    wait_weeknight = (start_time - now).total_seconds()

    if not weekend_free:
        return wait_weeknight

    # Next weekend midnight (entire weekend days are open)
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


# ── download ─────────────────────────────────────────────────────────────────


def _download_one(url: str, dest_path: str) -> dict:
    """Download a single file with retries.  Returns a manifest row dict."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    for attempt in range(_MAX_RETRIES):
        try:
            urllib.request.urlretrieve(url, dest_path)
            size = os.path.getsize(dest_path)
            return {"local_path": dest_path, "size_bytes": size, "status": "done"}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {"local_path": dest_path, "size_bytes": 0, "status": "missing"}
            if attempt == _MAX_RETRIES - 1:
                return {
                    "local_path": dest_path,
                    "size_bytes": 0,
                    "status": f"error_{e.code}",
                }
            time.sleep(2**attempt)
        except Exception:
            if attempt == _MAX_RETRIES - 1:
                return {"local_path": dest_path, "size_bytes": 0, "status": "error"}
            time.sleep(2**attempt)
    return {"local_path": dest_path, "size_bytes": 0, "status": "error"}


# ── manifest ─────────────────────────────────────────────────────────────────

_MANIFEST_COLS = [
    "model",
    "date",
    "hour",
    "product",
    "s3_url",
    "local_path",
    "size_bytes",
    "status",
    "downloaded_at",
]


def _load_manifest(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_parquet(path)
        if "product" not in df.columns:
            df["product"] = "anl"
        return df
    return pd.DataFrame(columns=_MANIFEST_COLS)


def _save_manifest(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)


def _done_keys(manifest: pd.DataFrame) -> set[tuple[str, str, int, str]]:
    """Set of (model, date, hour, product) already completed or missing."""
    if manifest.empty:
        return set()
    done = manifest[manifest["status"].isin(("done", "missing"))]
    return set(zip(done["model"], done["date"], done["hour"], done["product"]))


# ── signal handling ──────────────────────────────────────────────────────────

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n  Shutdown requested — finishing current batch ...")
    sys.stdout.flush()


# ── main ─────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bulk download RTMA/URMA hourly analysis files from AWS S3."
    )
    p.add_argument(
        "--model",
        choices=["rtma", "urma", "both"],
        default="urma",
        help="Product to download (default: urma).",
    )
    p.add_argument(
        "--dest",
        required=True,
        help="Root destination directory.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of concurrent download threads.",
    )
    p.add_argument(
        "--start-date",
        default=None,
        help="Override start date (YYYY-MM-DD).  Defaults to archive start.",
    )
    p.add_argument(
        "--end-date",
        default=None,
        help="End date (YYYY-MM-DD).  Defaults to yesterday.",
    )
    p.add_argument(
        "--schedule",
        default=None,
        help='Download time window, e.g. "00:00-08:00".  Unrestricted if omitted.',
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
    p.add_argument(
        "--include-pcp",
        action="store_true",
        help="Also download hourly precipitation analysis GRIBs.",
    )
    p.add_argument(
        "--pcp-only",
        action="store_true",
        help="Download ONLY precipitation files (skip analysis).",
    )
    return p.parse_args()


def _build_day_tasks(
    mdl: str,
    day: date,
    dest: str,
    done: set[tuple[str, str, int, str]],
    include_anl: bool,
    include_pcp: bool,
) -> list[tuple[str, str, str, int, str]]:
    """Build (url, local_path, date_str, hour, product) tasks for one day."""
    tasks: list[tuple[str, str, str, int, str]] = []
    ds = day.isoformat()
    for h in _HOURS:
        if include_anl and (mdl, ds, h, "anl") not in done:
            url = _s3_url(mdl, day, h)
            lp = _local_path(dest, mdl, day, h)
            if os.path.exists(lp) and os.path.getsize(lp) > 0:
                done.add((mdl, ds, h, "anl"))
            else:
                tasks.append((url, lp, ds, h, "anl"))

        if include_pcp and day >= _PCP_START.get(mdl, date.max):
            if (mdl, ds, h, "pcp") not in done:
                url = _pcp_s3_url(mdl, day, h)
                lp = _pcp_local_path(dest, mdl, day, h)
                if os.path.exists(lp) and os.path.getsize(lp) > 0:
                    done.add((mdl, ds, h, "pcp"))
                else:
                    tasks.append((url, lp, ds, h, "pcp"))
    return tasks


def main() -> None:
    args = _parse_args()
    models = ["rtma", "urma"] if args.model == "both" else [args.model]
    yesterday = date.today() - timedelta(days=1)

    include_anl = not args.pcp_only
    include_pcp = args.include_pcp or args.pcp_only

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    manifest_path = os.path.join(args.dest, "manifest.parquet")
    manifest = _load_manifest(manifest_path)
    done = _done_keys(manifest)
    new_rows: list[dict] = []

    total_downloaded = 0
    total_bytes = 0
    t_start = time.time()

    products_desc = []
    if include_anl:
        products_desc.append("analysis")
    if include_pcp:
        products_desc.append("precip")

    for mdl in models:
        start = (
            date.fromisoformat(args.start_date)
            if args.start_date
            else _ARCHIVE_START[mdl]
        )
        end = date.fromisoformat(args.end_date) if args.end_date else yesterday

        # Build day list
        days: list[date] = []
        d = start
        while d <= end:
            days.append(d)
            d += timedelta(days=1)
        if not args.oldest_first:
            days.reverse()

        print(f"\n{'=' * 60}")
        print(f"  {mdl.upper()}: {start} → {end}  ({len(days)} days)")
        print(f"  Products: {' + '.join(products_desc)}")
        print(f"  Workers: {args.workers}  Schedule: {args.schedule or 'unrestricted'}")
        print(f"  Already done: {len(done)} file-hours")
        print(f"{'=' * 60}\n")
        sys.stdout.flush()

        for day_i, day in enumerate(days):
            if _shutdown_requested:
                break

            # Check schedule
            _wait_for_window(args.schedule, args.weekend_free)
            if _shutdown_requested:
                break

            tasks = _build_day_tasks(
                mdl, day, args.dest, done, include_anl, include_pcp
            )

            if not tasks:
                continue

            # Download this day's files
            day_done = 0
            day_bytes = 0
            day_t0 = time.time()

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(_download_one, url, lp): (url, ds, h, prod)
                    for url, lp, ds, h, prod in tasks
                }
                for fut in as_completed(futures):
                    s3_url, ds, h, prod = futures[fut]
                    result = fut.result()
                    row = {
                        "model": mdl,
                        "date": ds,
                        "hour": h,
                        "product": prod,
                        "s3_url": s3_url,
                        **result,
                        "downloaded_at": datetime.now().isoformat(),
                    }
                    new_rows.append(row)
                    done.add((mdl, ds, h, prod))
                    if result["status"] == "done":
                        day_done += 1
                        day_bytes += result["size_bytes"]

            total_downloaded += day_done
            total_bytes += day_bytes
            elapsed = time.time() - day_t0
            rate = day_bytes / elapsed / 1e6 if elapsed > 0 else 0
            total_elapsed = time.time() - t_start

            print(
                f"  {day}  {day_done}/{len(tasks)} files  "
                f"{day_bytes / 1e6:.0f} MB  {rate:.0f} MB/s  "
                f"[total: {total_downloaded} files, {total_bytes / 1e9:.1f} GB, "
                f"{total_elapsed / 3600:.1f}h]"
            )
            sys.stdout.flush()

            # Flush manifest every 10 days
            if (day_i + 1) % 10 == 0 and new_rows:
                manifest = pd.concat(
                    [manifest, pd.DataFrame(new_rows)], ignore_index=True
                )
                new_rows.clear()
                _save_manifest(manifest, manifest_path)

    # Final manifest save
    if new_rows:
        manifest = pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True)
    _save_manifest(manifest, manifest_path)

    elapsed_h = (time.time() - t_start) / 3600
    print(
        f"\n  Done: {total_downloaded} files, {total_bytes / 1e9:.1f} GB in {elapsed_h:.1f}h"
    )
    if _shutdown_requested:
        print("  (interrupted — resume by re-running the same command)")


if __name__ == "__main__":
    main()
