"""
Probe the NOAA RTMA / URMA AWS S3 archives to build a coverage manifest.

Scans the public S3 buckets (noaa-rtma-pds, noaa-urma-pds) using unsigned
ListObjectsV2 REST calls to inventory every analysis file for each day in the
requested range.  Outputs:

  1. A JSON summary with start/end dates, era boundaries, variable inventories,
     and storage estimates.
  2. A Parquet manifest with one row per (model, date, hour) recording file
     existence, size, suffix, and .idx availability.

No AWS credentials required — uses the public S3 REST XML API over HTTPS.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta

import pandas as pd

# ── constants ────────────────────────────────────────────────────────────────

_BUCKETS: dict[str, str] = {
    "rtma": "noaa-rtma-pds",
    "urma": "noaa-urma-pds",
}

_PREFIX_TEMPLATES: dict[str, str] = {
    "rtma": "rtma2p5.{ymd}/",
    "urma": "urma2p5.{ymd}/",
}

# File-stem pattern: e.g. rtma2p5.t00z.2dvaranl_ndfd
_ANL_PATTERN: dict[str, re.Pattern[str]] = {
    "rtma": re.compile(r"rtma2p5\.t(\d{2})z\.2dvar(anl|err|ges)_ndfd(.*)"),
    "urma": re.compile(r"urma2p5\.t(\d{2})z\.2dvar(anl|err|ges)_ndfd(.*)"),
}

# Known search bounds for binary-search of archive start
_SEARCH_LO: dict[str, date] = {
    "rtma": date(2011, 1, 1),
    "urma": date(2012, 1, 1),
}
_SEARCH_HI: dict[str, date] = {
    "rtma": date(2014, 1, 1),
    "urma": date(2015, 1, 1),
}

# Version boundary dates to probe for variable inventories
_VERSION_BOUNDARY_DATES: list[str] = [
    "2018-12-15",  # v2.7
    "2020-07-15",  # v2.8
    "2023-02-15",  # v2.10 (JEDI)
    "2024-01-15",  # v2.10.5
]


# ── S3 helpers ───────────────────────────────────────────────────────────────


def _s3_list_url(bucket: str, prefix: str, continuation: str | None = None) -> str:
    url = f"https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}"
    if continuation:
        url += f"&continuation-token={urllib.request.quote(continuation)}"
    return url


def list_s3_prefix(bucket: str, prefix: str) -> list[tuple[str, int]]:
    """List all objects under *prefix* in *bucket*.  Returns (key, size) pairs."""
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    results: list[tuple[str, int]] = []
    continuation: str | None = None
    while True:
        url = _s3_list_url(bucket, prefix, continuation)
        with urllib.request.urlopen(url, timeout=30) as resp:
            tree = ET.parse(resp)
        root = tree.getroot()
        for contents in root.findall(f"{ns}Contents"):
            key_el = contents.find(f"{ns}Key")
            size_el = contents.find(f"{ns}Size")
            if (
                key_el is not None
                and key_el.text
                and size_el is not None
                and size_el.text
            ):
                results.append((key_el.text, int(size_el.text)))
        is_trunc = root.findtext(f"{ns}IsTruncated", "false")
        if is_trunc.lower() != "true":
            break
        token_el = root.find(f"{ns}NextContinuationToken")
        if token_el is None or not token_el.text:
            break
        continuation = token_el.text
    return results


def _s3_head(bucket: str, key: str) -> int | None:
    """HEAD request; returns content-length or None if 404."""
    url = f"https://{bucket}.s3.amazonaws.com/{key}"
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return int(resp.headers.get("Content-Length", 0))
    except urllib.error.HTTPError:
        return None


def _s3_get_text(bucket: str, key: str) -> str | None:
    """GET a small text object; returns body or None if 404."""
    url = f"https://{bucket}.s3.amazonaws.com/{key}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError:
        return None


# ── scanning ─────────────────────────────────────────────────────────────────


def find_archive_start(model: str, lo: date, hi: date) -> date | None:
    """Binary-search for the earliest date that has at least one file."""
    bucket = _BUCKETS[model]
    tmpl = _PREFIX_TEMPLATES[model]

    # Verify hi has data
    objs = list_s3_prefix(bucket, tmpl.format(ymd=hi.strftime("%Y%m%d")))
    if not objs:
        return None

    while lo < hi:
        mid = lo + (hi - lo) // 2
        prefix = tmpl.format(ymd=mid.strftime("%Y%m%d"))
        objs = list_s3_prefix(bucket, prefix)
        if objs:
            hi = mid
        else:
            lo = mid + timedelta(days=1)
    return lo


def scan_day(model: str, d: date) -> list[dict]:
    """List all analysis files for a given day and return one record per file.

    The .idx check is done once per day (on the first anl file found) to avoid
    24+ HEAD requests per day that would make full scans prohibitively slow.
    """
    bucket = _BUCKETS[model]
    prefix = _PREFIX_TEMPLATES[model].format(ymd=d.strftime("%Y%m%d"))
    objs = list_s3_prefix(bucket, prefix)
    if not objs:
        return []

    # Build a set of all keys for fast .idx lookup within the listing itself
    all_keys = {k for k, _ in objs}

    pat = _ANL_PATTERN[model]
    rows: list[dict] = []
    day_has_idx: bool | None = None  # checked once per day

    for key, size in objs:
        fname = key.rsplit("/", 1)[-1]
        m = pat.match(fname)
        if not m:
            continue
        hour_str, product, suffix = m.group(1), m.group(2), m.group(3)

        # Check .idx once per day: first see if .idx keys appear in the listing,
        # then fall back to a single HEAD request.
        if day_has_idx is None and product == "anl":
            day_has_idx = (key + ".idx") in all_keys
            if not day_has_idx:
                day_has_idx = _s3_head(bucket, key + ".idx") is not None

        rows.append(
            {
                "model": model,
                "date": d.isoformat(),
                "hour": int(hour_str),
                "product": product,
                "suffix": suffix if suffix else ".grb2",
                "size_bytes": size,
                "has_idx": bool(day_has_idx) if day_has_idx is not None else False,
                "key": key,
            }
        )
    return rows


def parse_idx(bucket: str, key: str) -> list[dict] | None:
    """Fetch and parse a GRIB2 .idx file into a list of message records."""
    text = _s3_get_text(bucket, key)
    if text is None:
        return None
    records: list[dict] = []
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) < 6:
            continue
        byte_offset = int(parts[1])
        var = parts[3]
        level = parts[4]
        forecast = parts[5] if len(parts) > 5 else ""
        # Compute message size from next offset
        next_offset = None
        if i + 1 < len(lines):
            nparts = lines[i + 1].split(":")
            if len(nparts) >= 2:
                next_offset = int(nparts[1])
        msg_size = (next_offset - byte_offset) if next_offset else None
        records.append(
            {
                "msg_num": int(parts[0]),
                "byte_offset": byte_offset,
                "var": var,
                "level": level,
                "forecast": forecast,
                "msg_size": msg_size,
            }
        )
    return records


def probe_version_boundaries(
    model: str, boundary_dates: list[str]
) -> dict[str, list[dict]]:
    """At each boundary date, fetch an anl .idx and return the variable inventory."""
    bucket = _BUCKETS[model]
    tmpl = _PREFIX_TEMPLATES[model]
    inventories: dict[str, list[dict]] = {}

    for ds in boundary_dates:
        d = date.fromisoformat(ds)
        prefix = tmpl.format(ymd=d.strftime("%Y%m%d"))
        objs = list_s3_prefix(bucket, prefix)
        if not objs:
            # Try day+1 and day-1
            for delta in (1, -1, 2, -2):
                d2 = d + timedelta(days=delta)
                prefix = tmpl.format(ymd=d2.strftime("%Y%m%d"))
                objs = list_s3_prefix(bucket, prefix)
                if objs:
                    d = d2
                    break
        if not objs:
            inventories[ds] = []
            continue

        # Find an analysis file
        pat = _ANL_PATTERN[model]
        anl_key = None
        for key, _ in objs:
            fname = key.rsplit("/", 1)[-1]
            m = pat.match(fname)
            if m and m.group(2) == "anl":
                anl_key = key
                break
        if anl_key is None:
            inventories[ds] = []
            continue

        # Try .idx suffixes
        idx_records = None
        for suf in (".idx",):
            idx_records = parse_idx(bucket, anl_key + suf)
            if idx_records:
                break
        inventories[ds] = idx_records or []

    return inventories


# ── storage estimates ────────────────────────────────────────────────────────


def compute_storage_estimates(df: pd.DataFrame) -> dict:
    """Compute storage totals from the manifest DataFrame."""
    if df.empty:
        return {}

    anl = df[df["product"] == "anl"].copy()
    estimates: dict = {}

    for mdl in anl["model"].unique():
        sub = anl[anl["model"] == mdl]
        total_bytes = int(sub["size_bytes"].sum())
        n_files = len(sub)
        # Extrapolate: sampled days → full coverage
        n_unique_days = sub["date"].nunique()
        d_min = date.fromisoformat(sub["date"].min())
        d_max = date.fromisoformat(sub["date"].max())
        total_days_in_range = (d_max - d_min).days + 1
        if n_unique_days > 0:
            scale = total_days_in_range / n_unique_days
        else:
            scale = 1.0
        estimates[mdl] = {
            "sampled_files": n_files,
            "sampled_bytes": total_bytes,
            "sampled_days": n_unique_days,
            "total_days_in_range": total_days_in_range,
            "extrapolation_factor": round(scale, 2),
            "estimated_total_bytes": int(total_bytes * scale),
            "estimated_total_tb": round(total_bytes * scale / 1e12, 2),
        }
    return estimates


# ── main ─────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Probe RTMA/URMA AWS S3 archives and build a coverage manifest."
    )
    p.add_argument(
        "--model",
        choices=["rtma", "urma", "both"],
        default="both",
        help="Which product(s) to scan.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sample every Nth day (1=every day, 7=weekly, 30=monthly).",
    )
    p.add_argument("--start-date", default=None, help="Scan start date (YYYY-MM-DD).")
    p.add_argument("--end-date", default=None, help="Scan end date (YYYY-MM-DD).")
    p.add_argument(
        "--out-json",
        default="artifacts/rtma_urma_archive_probe.json",
        help="Output JSON summary path.",
    )
    p.add_argument(
        "--out-manifest",
        default="artifacts/rtma_archive_manifest.parquet",
        help="Output Parquet manifest path.",
    )
    p.add_argument(
        "--skip-boundary-probe",
        action="store_true",
        help="Skip version-boundary .idx probing (faster).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    models = ["rtma", "urma"] if args.model == "both" else [args.model]
    today = date.today()
    all_rows: list[dict] = []
    summary: dict = {"generated": datetime.now().isoformat(), "models": {}}

    for mdl in models:
        print(f"\n{'=' * 60}")
        print(f"  Probing {mdl.upper()} ({_BUCKETS[mdl]})")
        print(f"{'=' * 60}")

        # ── find archive start via binary search ──
        if args.start_date:
            start = date.fromisoformat(args.start_date)
            print(f"  Using user-specified start: {start}")
        else:
            print("  Binary-searching for archive start date ...")
            start = find_archive_start(
                mdl,
                lo=_SEARCH_LO[mdl],
                hi=_SEARCH_HI[mdl],
            )
            if start is None:
                print(f"  WARNING: no data found for {mdl} in search range")
                continue
            print(f"  Archive starts: {start}")

        end = date.fromisoformat(args.end_date) if args.end_date else today

        # ── scan days ──
        d = start
        n_scanned = 0
        n_with_data = 0
        while d <= end:
            rows = scan_day(mdl, d)
            if rows:
                all_rows.extend(rows)
                n_with_data += 1
            n_scanned += 1
            if n_scanned % 50 == 0:
                print(
                    f"  {d}  scanned={n_scanned}  with_data={n_with_data}  files_so_far={len(all_rows)}"
                )
            d += timedelta(days=args.stride)

        print(
            f"  Done: scanned {n_scanned} days, {n_with_data} had data, {len(all_rows)} total file records"
        )

        # ── version boundary probing ──
        inventories: dict[str, list[dict]] = {}
        if not args.skip_boundary_probe:
            print("  Probing version boundaries for variable inventories ...")
            inventories = probe_version_boundaries(mdl, _VERSION_BOUNDARY_DATES)
            for bd, recs in inventories.items():
                n_vars = len(recs)
                vars_str = ", ".join(r["var"] for r in recs) if recs else "(no .idx)"
                print(f"    {bd}: {n_vars} vars — {vars_str}")

        summary["models"][mdl] = {
            "bucket": _BUCKETS[mdl],
            "archive_start": start.isoformat(),
            "scan_end": end.isoformat(),
            "stride": args.stride,
            "days_scanned": n_scanned,
            "days_with_data": n_with_data,
            "version_inventories": {
                bd: [
                    {"var": r["var"], "level": r["level"], "msg_size": r["msg_size"]}
                    for r in recs
                ]
                for bd, recs in inventories.items()
            },
        }

    # ── build manifest DataFrame ──
    if not all_rows:
        print("\nNo data found. Exiting.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

    # ── storage estimates ──
    estimates = compute_storage_estimates(df)
    summary["storage_estimates"] = estimates

    # ── detect era boundaries (suffix transitions) ──
    anl = df[df["product"] == "anl"].copy()
    if not anl.empty:
        for mdl in anl["model"].unique():
            sub = anl[anl["model"] == mdl].sort_values("date")
            suffixes = sub.groupby("date")["suffix"].first()
            transitions = []
            prev = None
            for dt_str, suf in suffixes.items():
                if prev is not None and suf != prev:
                    transitions.append({"date": str(dt_str), "from": prev, "to": suf})
                prev = suf
            summary["models"][mdl]["era_transitions"] = transitions
            summary["models"][mdl]["suffixes_seen"] = sorted(
                sub["suffix"].unique().tolist()
            )

    # ── write outputs ──
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nJSON summary: {args.out_json}")

    os.makedirs(os.path.dirname(args.out_manifest) or ".", exist_ok=True)
    df.to_parquet(args.out_manifest, index=False)
    print(f"Parquet manifest: {args.out_manifest} ({len(df)} rows)")

    # ── console summary ──
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for mdl, info in summary["models"].items():
        print(f"\n  {mdl.upper()}:")
        print(f"    Archive start : {info['archive_start']}")
        print(f"    Scan end      : {info['scan_end']}")
        print(f"    Days scanned  : {info['days_scanned']}")
        print(f"    Days with data: {info['days_with_data']}")
        if "suffixes_seen" in info:
            print(f"    Suffixes      : {info['suffixes_seen']}")
        if "era_transitions" in info:
            for t in info["era_transitions"]:
                print(f"    Transition    : {t['date']}  {t['from']} → {t['to']}")
    if estimates:
        print("\n  Storage estimates (analysis files only):")
        for mdl, est in estimates.items():
            print(
                f"    {mdl.upper()}: ~{est['estimated_total_tb']} TB ({est['sampled_files']} sampled files, {est['extrapolation_factor']}x extrapolation)"
            )


if __name__ == "__main__":
    main()
