"""Build RTMA wind uselist/rejectlist filter for PNW stations.

Parses 4 GSI-fix files and MADIS provider metadata to determine which
station fids would be accepted by RTMA's mesonet wind QC.

Acceptance logic (from sfcobsqc.f90):
    accepted = (on_provider_uselist OR on_station_uselist OR on_wbin_uselist)
               AND NOT on_wind_rejectlist
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import pandas as pd


def parse_provider_uselist(path: str) -> set[str]:
    """Parse rtma_mesonet_uselist.txt — 3-line header (*** borders + title)."""
    providers = set()
    with open(path) as f:
        lines = f.readlines()
    for line in lines[3:]:
        provider = line[:8].strip()
        if provider:
            providers.add(provider)
    return providers


def parse_station_uselist(path: str) -> set[str]:
    """Parse nam_mesonet_stnuselist.txt — no header, ID in cols 0:8."""
    stations = set()
    with open(path) as f:
        for line in f:
            sid = line[:8].strip()
            if sid:
                stations.add(sid)
    return stations


def parse_wbin_uselist(path: str) -> set[str]:
    """Parse rtma_wbinuselist — 1-line header, ID in cols 0:8, deduplicate."""
    stations = set()
    with open(path) as f:
        lines = f.readlines()
    for line in lines[1:]:
        sid = line[:8].strip()
        if sid:
            stations.add(sid)
    return stations


def parse_wind_rejectlist(path: str) -> set[str]:
    """Parse rtma_w_rejectlist — 3-line header, ID in cols 1:9 (inside quotes)."""
    stations = set()
    with open(path) as f:
        lines = f.readlines()
    for line in lines[3:]:
        sid = line[1:9].strip()
        if sid:
            stations.add(sid)
    return stations


def build_fid_provider_map(madis_dir: str, n_dates: int = 6) -> dict[str, str]:
    """Read a few MADIS daily parquets to build fid -> dataProvider map."""
    files = sorted(os.listdir(madis_dir))
    # Sample from recent dates for good coverage
    files = [f for f in files if f.endswith(".parquet")]
    if len(files) > n_dates:
        step = max(1, len(files) // n_dates)
        sampled = files[-1:] + files[-step * n_dates :: step]
        sampled = list(dict.fromkeys(sampled))[:n_dates]
    else:
        sampled = files

    fid_provider: dict[str, str] = {}
    for fname in sampled:
        path = os.path.join(madis_dir, fname)
        df = pd.read_parquet(path, columns=["stationId", "dataProvider"])
        for sid, prov in zip(df["stationId"], df["dataProvider"]):
            if sid not in fid_provider:
                fid_provider[sid] = prov
    return fid_provider


def main() -> None:
    p = argparse.ArgumentParser(description="Build RTMA wind accepted fids list.")
    p.add_argument(
        "--gsi-fix-dir",
        default=os.path.expanduser("~/code/GSI-fix"),
        help="Path to GSI-fix repo",
    )
    p.add_argument(
        "--wind-table",
        default="/nas/dads/mvp/station_day_wind_pnw_2018_2024.parquet",
        help="Wind station-day parquet (for fid list)",
    )
    p.add_argument(
        "--madis-daily-dir",
        default="/data/ssd2/madis/daily_all",
        help="MADIS daily parquet directory (for provider lookup)",
    )
    p.add_argument(
        "--out",
        default="artifacts/rtma_wind_accepted_fids.json",
        help="Output JSON path",
    )
    args = p.parse_args()

    # Parse GSI-fix files
    provider_uselist = parse_provider_uselist(
        os.path.join(args.gsi_fix_dir, "rtma_mesonet_uselist.txt")
    )
    station_uselist = parse_station_uselist(
        os.path.join(args.gsi_fix_dir, "nam_mesonet_stnuselist.txt")
    )
    wbin_uselist = parse_wbin_uselist(
        os.path.join(args.gsi_fix_dir, "rtma_wbinuselist")
    )
    wind_rejectlist = parse_wind_rejectlist(
        os.path.join(args.gsi_fix_dir, "rtma_w_rejectlist")
    )

    print(f"Provider uselist: {len(provider_uselist)} providers")
    print(f"  Providers: {sorted(provider_uselist)}")
    print(f"Station uselist: {len(station_uselist)} stations")
    print(f"Wbin uselist: {len(wbin_uselist)} unique stations")
    print(f"Wind rejectlist: {len(wind_rejectlist)} stations")

    # Get fids from wind table (fid is in the MultiIndex)
    idx = pd.read_parquet(args.wind_table, columns=[]).index
    if isinstance(idx, pd.MultiIndex):
        all_fids = set(idx.get_level_values("fid").unique())
    else:
        all_fids = set(idx.unique())
    print(f"\nWind table fids: {len(all_fids)}")

    # Build fid -> provider map from MADIS
    print("Loading MADIS provider metadata...")
    fid_provider = build_fid_provider_map(args.madis_daily_dir)
    print(f"MADIS fid->provider map: {len(fid_provider)} stations")

    # Apply acceptance logic
    accepted = set()
    path_counts: dict[str, int] = defaultdict(int)
    provider_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "accepted": 0}
    )

    for fid in sorted(all_fids):
        provider = fid_provider.get(fid, "UNKNOWN")
        # Truncate to 8 chars to match GSI-fix format
        provider_8 = provider[:8]

        provider_stats[provider]["total"] += 1

        # Check acceptance paths
        on_provider = provider_8 in provider_uselist
        on_station = fid in station_uselist
        on_wbin = fid in wbin_uselist
        on_reject = fid in wind_rejectlist

        if (on_provider or on_station or on_wbin) and not on_reject:
            accepted.add(fid)
            provider_stats[provider]["accepted"] += 1
            if on_provider:
                path_counts["provider_uselist"] += 1
            if on_station:
                path_counts["station_uselist"] += 1
            if on_wbin:
                path_counts["wbin_uselist"] += 1
        elif on_reject:
            path_counts["rejected"] += 1

    # Report
    print("\n--- Acceptance Summary ---")
    print(f"Total fids: {len(all_fids)}")
    print(f"Accepted: {len(accepted)} ({100 * len(accepted) / len(all_fids):.1f}%)")
    print(f"Rejected: {len(all_fids) - len(accepted)}")

    print("\nAcceptance paths (stations may match multiple):")
    for path, count in sorted(path_counts.items()):
        print(f"  {path}: {count}")

    print("\nBy provider (top 15):")
    by_total = sorted(provider_stats.items(), key=lambda x: x[1]["total"], reverse=True)
    for prov, stats in by_total[:15]:
        pct = 100 * stats["accepted"] / stats["total"] if stats["total"] else 0
        print(
            f"  {prov:15s}: {stats['accepted']:4d}/{stats['total']:4d} accepted ({pct:.0f}%)"
        )

    # Write output
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(sorted(accepted), f, indent=2)
    print(f"\nWrote {len(accepted)} accepted fids to {args.out}")


if __name__ == "__main__":
    main()
