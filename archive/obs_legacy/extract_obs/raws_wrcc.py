"""Download RAWS daily summary data from the Western Regional Climate Center (WRCC).

Scrapes the WRCC Daily Summary Time Series endpoint (wea_dysimts2.pl) for RAWS
stations listed on per-region inventory pages. Saves per-station CSVs to disk.

Usage:
    # Step 1: build station inventory from WRCC listing pages
    python -m extract.met_data.obs.raws_wrcc inventory --out-dir /nas/climate/raws/wrcc

    # Step 2: download daily summaries for all stations
    python -m extract.met_data.obs.raws_wrcc download --out-dir /nas/climate/raws/wrcc \
        --start 1985-01-01 --end 2001-12-31

    # Step 3 (optional): download only PNW stations within bounding box
    python -m extract.met_data.obs.raws_wrcc download --out-dir /nas/climate/raws/wrcc \
        --start 1985-01-01 --end 2001-12-31 --bounds -125.0,42.0,-104.0,49.0
"""

import argparse
import logging
import re
import time
from html import unescape
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://raws.dri.edu"
DATA_URL = "https://wrcc.dri.edu/cgi-bin/wea_dysimts2.pl"
INFO_URL = f"{BASE_URL}/cgi-bin/wea_info.pl"

# Region listing pages covering the western US. Each maps to a *lst.html page.
# The image map at raws.dri.edu/wraws/usaarch.map defines these regions.
REGION_PAGES = {
    "wa": "walst.html",
    "or": "orlst.html",
    "sid": "sidlst.html",
    "nidwmt": "nidwmtlst.html",
    "emt": "emtlst.html",
    "wy": "wylst.html",
    "nca": "ncalst.html",
    "cca": "ccalst.html",
    "sca": "scalst.html",
    "nvut": "nvutlst.html",
    "co": "colst.html",
    "az": "azlst.html",
    "nm": "nmlst.html",
    "nd": "ndlst.html",
    "sd": "sdlst.html",
    "ne": "nelst.html",
    "ks": "kslst.html",
    "ok": "oklst.html",
    "tx": "txlst.html",
    "mn": "mnlst.html",
    "ia": "ialst.html",
    "mo": "molst.html",
    "ar": "arlst.html",
    "la": "lalst.html",
    "ut": "utlst.html",
    "ak": "aklst.html",
}

# Columns produced by the daily summary with qBasic=ON, unit=M
DAILY_COLUMNS = [
    "date",
    "year",
    "doy",
    "day_of_run",
    "srad_total_kwh_m2",
    "wspd_ave_ms",
    "wdir_vec_deg",
    "wspd_gust_ms",
    "tair_ave_c",
    "tair_max_c",
    "tair_min_c",
    "rh_ave_pct",
    "rh_max_pct",
    "rh_min_pct",
    "prcp_total_mm",
]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "DADS-MVP/1.0 (climate research)"})


def _parse_dms(text):
    """Parse a degree-minute-second string like '45° 27\\' 25\"' to decimal degrees."""
    text = unescape(text).strip()
    m = re.match(r"(\d+)\s*\u00b0\s*(\d+)\s*[\']\s*(\d+)\s*[\"]*", text)
    if not m:
        return None
    deg, mn, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return deg + mn / 60.0 + sec / 3600.0


def scrape_station_list(region_key):
    """Scrape a region listing page and return a list of (wrcc_id, name) tuples."""
    page = REGION_PAGES[region_key]
    url = f"{BASE_URL}/wraws/{page}"
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()

    # Links: <A HREF="...rawMAIN.pl?orOTIL" onMouseOver="...update('Name (RAWS)',...
    # The onMouseOver contains embedded HTML that breaks simple regex, so extract
    # the station name from the update() JavaScript call which is always clean.
    href_pat = re.compile(r'rawMAIN\.pl\?(\w+)"', re.IGNORECASE)
    name_pat = re.compile(r"update\('([^']+?)\s*\(RAWS\)'")
    stations = []
    for line in resp.text.splitlines():
        m = href_pat.search(line)
        if not m:
            continue
        wrcc_id = m.group(1)
        nm = name_pat.search(line)
        if nm:
            name = re.sub(r"\s+", " ", nm.group(1)).strip()
        else:
            name = wrcc_id
        stations.append((wrcc_id, name))

    return stations


def scrape_station_metadata(wrcc_id):
    """Fetch station metadata (lat, lon, elevation, NESS ID, NWS ID, agency)."""
    url = f"{INFO_URL}?{wrcc_id}"
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    html = unescape(resp.text)

    info = {"wrcc_id": wrcc_id}

    loc = re.search(
        r"<b>Location\s*</b>\s*</td>\s*<td>\s*(.+?)\s*\n", html, re.IGNORECASE
    )
    if loc:
        info["location"] = loc.group(1).strip()

    lat = re.search(
        r"<b>Latitude\s*</b>\s*</td>\s*<td>\s*(.+?)\s*\n", html, re.IGNORECASE
    )
    if lat:
        info["latitude"] = _parse_dms(lat.group(1))

    lon = re.search(
        r"<b>Longitude\s*</b>\s*</td>\s*<td>\s*(.+?)\s*\n", html, re.IGNORECASE
    )
    if lon:
        val = _parse_dms(lon.group(1))
        if val is not None:
            info["longitude"] = -val  # western hemisphere

    elev = re.search(
        r"<b>Elevation\s*</b>\s*</td>\s*<td>\s*(\d+)\s*ft", html, re.IGNORECASE
    )
    if elev:
        info["elevation_m"] = round(int(elev.group(1)) * 0.3048, 1)

    ness = re.search(r"<b>NESS ID\s*</b>\s*</td>\s*<td>\s*(\w+)", html, re.IGNORECASE)
    if ness:
        info["ness_id"] = ness.group(1)

    nws = re.search(r"<b>NWS ID\s*</b>\s*</td>\s*<td>\s*(\w+)", html, re.IGNORECASE)
    if nws:
        info["nws_id"] = nws.group(1)

    agency = re.search(
        r"<b>Agency\s*</b>\s*</td>\s*<td>\s*(.+?)\s*\n", html, re.IGNORECASE
    )
    if agency:
        info["agency"] = agency.group(1).strip()

    return info


def scrape_station_por(wrcc_id):
    """Fetch station period of record from the monthly data inventory page.

    Returns (start_year, end_year) or (None, None) if unavailable.
    """
    url = f"{BASE_URL}/cgi-bin/wea_inventoryA.pl?{wrcc_id}"
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    years = re.findall(r"<B>\s*(\d{4})\s*</B>", resp.text)
    if years:
        return int(min(years)), int(max(years))
    return None, None


def build_inventory(regions, out_dir, delay=0.3):
    """Build station inventory CSV from WRCC listing and metadata pages."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stations = []
    for region in regions:
        if region not in REGION_PAGES:
            log.warning("Unknown region %s, skipping", region)
            continue
        log.info("Scraping station list for region: %s", region)
        stations = scrape_station_list(region)
        log.info("  Found %d stations", len(stations))
        for wrcc_id, name in stations:
            all_stations.append(
                {"wrcc_id": wrcc_id, "name": name, "region_page": region}
            )

    # Deduplicate (same station can appear on multiple region pages)
    df = pd.DataFrame(all_stations).drop_duplicates(subset="wrcc_id")
    log.info("Total unique stations: %d", len(df))

    # Fetch metadata and POR for each station (2 requests per station)
    records = []
    for i, row in enumerate(df.itertuples()):
        if i % 50 == 0:
            log.info("  Fetching metadata %d / %d ...", i, len(df))
        try:
            info = scrape_station_metadata(row.wrcc_id)
            info["name"] = row.name
            info["region_page"] = row.region_page
            time.sleep(delay)
            start_year, end_year = scrape_station_por(row.wrcc_id)
            info["por_start"] = start_year
            info["por_end"] = end_year
            records.append(info)
        except Exception as e:
            log.warning("  Failed metadata for %s: %s", row.wrcc_id, e)
            records.append(
                {
                    "wrcc_id": row.wrcc_id,
                    "name": row.name,
                    "region_page": row.region_page,
                }
            )
        time.sleep(delay)

    inv = pd.DataFrame(records)
    inv_path = out_dir / "raws_inventory.csv"
    inv.to_csv(inv_path, index=False)
    log.info("Wrote inventory to %s (%d stations)", inv_path, len(inv))
    return inv


# WRCC server silently returns empty data for requests spanning >5 years.
# Use 4-year offset so each chunk covers at most 5 calendar years (e.g. 1997-2001).
_MAX_CHUNK_YEARS = 4


def _fetch_chunk(stn_code, start, end):
    """Fetch a single <=5-year chunk of daily data. Returns list of parsed row lists."""
    payload = {
        "stn": stn_code,
        "smon": start.strftime("%m"),
        "sday": start.strftime("%d"),
        "syea": start.strftime("%y"),
        "emon": end.strftime("%m"),
        "eday": end.strftime("%d"),
        "eyea": end.strftime("%y"),
        "qBasic": "ON",
        "unit": "M",
        "Ofor": "A",
        "Datareq": "A",
        "qc": "Y",
        "miss": "08",
        "obs": "N",
        "WsMon": "01",
        "WsDay": "01",
        "WeMon": "12",
        "WeDay": "31",
        "Submit Info": "Submit Info",
    }

    try:
        resp = SESSION.post(DATA_URL, data=payload, timeout=5)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        log.debug("Timeout for %s %s-%s", stn_code, start.date(), end.date())
        return []
    except requests.exceptions.RequestException as e:
        log.debug("Request error for %s: %s", stn_code, e)
        return []

    text = resp.text
    pre_match = re.search(r"<PRE>(.*?)</PRE>", text, re.DOTALL | re.IGNORECASE)
    if pre_match:
        text = pre_match.group(1)

    if "Improper program call" in text or "Access to WRCC" in text:
        return []

    rows = []
    for line in text.strip().splitlines():
        stripped = line.strip()
        if re.match(r"\d{2}/\d{2}/\d{4}", stripped):
            parts = stripped.split()
            if len(parts) < 15:
                parts.extend(["-9999"] * (15 - len(parts)))
            rows.append(parts[:15])
    return rows


def download_daily(stn_code, start, end):
    """Download daily summary for a station, chunking into 5-year windows.

    Returns parsed DataFrame or None.
    """
    all_rows = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(
            pd.Timestamp(year=chunk_start.year + _MAX_CHUNK_YEARS, month=12, day=31),
            end,
        )
        rows = _fetch_chunk(stn_code, chunk_start, chunk_end)
        all_rows.extend(rows)
        chunk_start = pd.Timestamp(year=chunk_end.year + 1, month=1, day=1)

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows, columns=DAILY_COLUMNS)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")

    for col in DAILY_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace(-9999.0, float("nan"))
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

    return df


def download_all(
    out_dir,
    start=None,
    end=None,
    bounds=None,
    delay=0.5,
    overwrite=False,
    full_por=False,
):
    """Download daily summaries for all stations in inventory.

    If full_por=True, use each station's por_start/por_end from the inventory
    instead of global start/end dates.
    """
    out_dir = Path(out_dir)
    inv_path = out_dir / "raws_inventory.csv"
    if not inv_path.exists():
        raise FileNotFoundError(f"No inventory at {inv_path}. Run 'inventory' first.")

    inv = pd.read_csv(inv_path)
    data_dir = out_dir / "station_data"
    data_dir.mkdir(exist_ok=True)

    if bounds:
        w, s, e, n = bounds
        inv = inv.dropna(subset=["latitude", "longitude"])
        inv = inv[
            (inv["latitude"] >= s)
            & (inv["latitude"] <= n)
            & (inv["longitude"] >= w)
            & (inv["longitude"] <= e)
        ]
        log.info("Filtered to %d stations within bounds %s", len(inv), bounds)

    if not full_por:
        if not start or not end:
            raise ValueError("Must provide --start and --end, or use --full-por")

    success, fail, skip, no_por = 0, 0, 0, 0
    for i, row in enumerate(inv.itertuples()):
        wrcc_id = row.wrcc_id
        stn_code = wrcc_id[2:]  # strip 2-char region prefix

        out_file = data_dir / f"{wrcc_id}.csv"
        if out_file.exists() and not overwrite:
            skip += 1
            continue

        if full_por:
            por_s = getattr(row, "por_start", None)
            por_e = getattr(row, "por_end", None)
            if pd.isna(por_s) or pd.isna(por_e):
                no_por += 1
                continue
            start_dt = pd.Timestamp(f"{int(por_s)}-01-01")
            end_dt = pd.Timestamp(f"{int(por_e)}-12-31")
        else:
            start_dt = pd.Timestamp(start)
            end_dt = pd.Timestamp(end)

        if i % 20 == 0:
            log.info(
                "Progress: %d / %d (ok=%d, fail=%d, skip=%d, no_por=%d)",
                i,
                len(inv),
                success,
                fail,
                skip,
                no_por,
            )

        try:
            df = download_daily(stn_code, start_dt, end_dt)
            if df is not None and len(df) > 0:
                df.to_csv(out_file, index=False)
                success += 1
            else:
                fail += 1
                log.debug("No data for %s (%s)", wrcc_id, getattr(row, "name", ""))
        except Exception as e:
            fail += 1
            log.warning("Error downloading %s: %s", wrcc_id, e)

        time.sleep(delay)

    log.info(
        "Done: %d downloaded, %d failed, %d skipped, %d no_por",
        success,
        fail,
        skip,
        no_por,
    )


def main():
    parser = argparse.ArgumentParser(description="Download RAWS daily data from WRCC")
    sub = parser.add_subparsers(dest="command", required=True)

    # inventory subcommand
    p_inv = sub.add_parser(
        "inventory", help="Build station inventory from WRCC listing pages"
    )
    p_inv.add_argument("--out-dir", required=True, help="Output directory")
    p_inv.add_argument(
        "--regions",
        nargs="*",
        default=None,
        help="Region keys to scrape (default: all)",
    )
    p_inv.add_argument(
        "--delay", type=float, default=0.3, help="Seconds between metadata requests"
    )

    # download subcommand
    p_dl = sub.add_parser(
        "download", help="Download daily summaries for stations in inventory"
    )
    p_dl.add_argument(
        "--out-dir",
        required=True,
        help="Output directory (must contain raws_inventory.csv)",
    )
    p_dl.add_argument(
        "--full-por",
        action="store_true",
        help="Download each station's full period of record from inventory",
    )
    p_dl.add_argument(
        "--start", default=None, help="Start date YYYY-MM-DD (ignored with --full-por)"
    )
    p_dl.add_argument(
        "--end", default=None, help="End date YYYY-MM-DD (ignored with --full-por)"
    )
    p_dl.add_argument(
        "--bounds",
        default=None,
        help="Bounding box W,S,E,N (e.g. -125.0,42.0,-104.0,49.0)",
    )
    p_dl.add_argument(
        "--delay", type=float, default=0.5, help="Seconds between data requests"
    )
    p_dl.add_argument(
        "--overwrite", action="store_true", help="Re-download existing files"
    )

    args = parser.parse_args()

    if args.command == "inventory":
        regions = args.regions or list(REGION_PAGES.keys())
        build_inventory(regions, args.out_dir, delay=args.delay)

    elif args.command == "download":
        bounds = None
        if args.bounds:
            bounds = tuple(float(x) for x in args.bounds.split(","))
        download_all(
            args.out_dir,
            start=args.start,
            end=args.end,
            bounds=bounds,
            delay=args.delay,
            overwrite=args.overwrite,
            full_por=args.full_por,
        )


if __name__ == "__main__":
    main()
