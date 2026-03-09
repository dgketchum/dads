"""Apply quantile-mapping calibration retroactively to existing GOES ABI COGs.

Reads uncalibrated COGs, applies the qmap transform built by
``calibrate_goes_to_gridsat.py``, and overwrites them in-place.

Only processes files dated *after* the overlap period (2025-08-31) since
overlap-period COGs serve as qmap training data and should stay raw.

Usage
-----
  uv run python -m extract.rs.goes_abi.apply_qmap_retroactive \
      --goes-dir /nas/dads/rs/goes_abi \
      --qmap artifacts/goes_to_gridsat_qmap.json \
      --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date

import numpy as np
import rasterio

from extract.rs.goes_abi.goes_abi_common import BAND_MAP

# Files on or before this date are overlap-period training data — skip them.
_OVERLAP_END = date(2025, 8, 31)

_CHANNELS = set(BAND_MAP.keys())

_FNAME_RE = re.compile(r"goes_abi_(\d{8})_(\d{4})_(.+)\.tif$")


def _parse_filename(fname: str) -> tuple[date, str] | None:
    """Extract (date, channel) from a GOES ABI COG filename.

    Returns None if the filename doesn't match the expected pattern.
    """
    m = _FNAME_RE.search(fname)
    if m is None:
        return None
    d = date(int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:8]))
    channel = m.group(3)
    return d, channel


def _apply_qmap_to_file(path: str, channel: str, qmap: dict) -> str:
    """Read a COG, apply qmap, write back in-place. Returns status string."""
    entry = qmap.get(channel)
    if entry is None:
        return "no_qmap_entry"

    goes_bp = np.array(entry["goes_breakpoints"], dtype=np.float32)
    gridsat_bp = np.array(entry["gridsat_breakpoints"], dtype=np.float32)

    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile.copy()

    valid = ~np.isnan(data)
    if not valid.any():
        return "all_nan"

    data[valid] = np.interp(data[valid], goes_bp, gridsat_bp)

    # COG driver doesn't support update mode — rewrite the file
    profile.update(driver="GTiff")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)

    return "calibrated"


def _collect_files(goes_dir: str) -> list[tuple[str, str]]:
    """Walk goes_dir and return [(path, channel), ...] for post-overlap COGs."""
    targets: list[tuple[str, str]] = []
    for dirpath, _, filenames in os.walk(goes_dir):
        for fname in filenames:
            parsed = _parse_filename(fname)
            if parsed is None:
                continue
            d, channel = parsed
            if d <= _OVERLAP_END:
                continue
            if channel not in _CHANNELS:
                continue
            targets.append((os.path.join(dirpath, fname), channel))
    return targets


def _worker(args: tuple[str, str, dict]) -> tuple[str, str]:
    """Process pool target: apply qmap to one file."""
    path, channel, qmap = args
    status = _apply_qmap_to_file(path, channel, qmap)
    return path, status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply qmap calibration retroactively to GOES ABI COGs"
    )
    parser.add_argument(
        "--goes-dir",
        default="/nas/dads/rs/goes_abi",
        help="Root dir for GOES ABI COGs",
    )
    parser.add_argument(
        "--qmap",
        default="artifacts/goes_to_gridsat_qmap.json",
        help="Path to quantile-mapping calibration artifact",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Parallel worker processes"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report files that would be updated without modifying them",
    )
    args = parser.parse_args()

    with open(args.qmap) as f:
        qmap = json.load(f)

    targets = _collect_files(args.goes_dir)
    print(f"Found {len(targets)} post-overlap COGs to calibrate", flush=True)

    if not targets:
        print("Nothing to do.")
        return

    if args.dry_run:
        by_channel: dict[str, int] = {}
        for _, ch in targets:
            by_channel[ch] = by_channel.get(ch, 0) + 1
        for ch, n in sorted(by_channel.items()):
            print(f"  {ch}: {n} files")
        print(f"  Total: {len(targets)} files (dry run — no changes made)")
        return

    done = 0
    errors = 0
    work_items = [(path, ch, qmap) for path, ch in targets]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, item): item[0] for item in work_items}
        for future in as_completed(futures):
            path = futures[future]
            try:
                _, status = future.result()
                done += 1
                if done % 500 == 0:
                    print(f"  {done}/{len(targets)} files processed", flush=True)
            except Exception as e:
                errors += 1
                print(f"  ERROR {path}: {e}", flush=True)

    print(f"\nDone: {done} calibrated, {errors} errors out of {len(targets)} files")


if __name__ == "__main__":
    main()
