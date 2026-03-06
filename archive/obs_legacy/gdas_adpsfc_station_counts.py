#!/usr/bin/env python
"""Extract ADPSFC station observation counts from GDAS PrepBUFR files (GDEX d337000).

Reads daily PrepBUFR tar.gz files, iterates all ADPSFC subsets, and writes a
shapefile with one point per station containing the total observation count
for the requested period.
"""

import os
import sys
import glob
import tarfile
import tempfile

import numpy as np
import ncepbufr
import geopandas as gpd
from shapely.geometry import Point


MISSING = 1e10  # ncepbufr default missing value


def extract_and_read(tar_path, tmpdir):
    """Extract prepbufr files from a daily tar.gz and return their paths."""
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(tmpdir)
    return sorted(glob.glob(os.path.join(tmpdir, "prepbufr.gdas.*")))


def read_adpsfc(bufr_path, stations):
    """Read ADPSFC subsets from a single PrepBUFR file, accumulate into stations dict."""
    bufr = ncepbufr.open(bufr_path)
    n = 0
    while bufr.advance() == 0:
        if bufr.msg_type != "ADPSFC":
            continue
        while bufr.load_subset() == 0:
            hdr = bufr.read_subset("SID XOB YOB ELV TYP")
            obs = bufr.read_subset("POB TOB QOB UOB VOB")

            sid_float = hdr[0, 0]
            # SID is packed as a float64 representing 8 ASCII chars
            sid = sid_float.tobytes().decode("ascii", errors="ignore").strip()
            lon = float(hdr[1, 0])
            lat = float(hdr[2, 0])
            sid = (
                "".join(c for c in sid if c.isprintable()) or f"UNK_{lon:.2f}_{lat:.2f}"
            )
            elv = float(hdr[3, 0])
            typ = int(hdr[4, 0])

            if sid not in stations:
                stations[sid] = {
                    "lat": lat,
                    "lon": lon,
                    "elv": elv if elv < MISSING else np.nan,
                    "typ": typ,
                    "count": 0,
                    "has_t": 0,
                    "has_q": 0,
                    "has_wind": 0,
                    "has_p": 0,
                }

            stations[sid]["count"] += 1

            # Track which variables are present (not missing)
            pob = float(obs[0, 0])
            tob = float(obs[1, 0])
            qob = float(obs[2, 0])
            uob = float(obs[3, 0])

            if pob < MISSING:
                stations[sid]["has_p"] += 1
            if tob < MISSING:
                stations[sid]["has_t"] += 1
            if qob < MISSING:
                stations[sid]["has_q"] += 1
            if uob < MISSING:
                stations[sid]["has_wind"] += 1

            n += 1
    bufr.close()
    return n


def main(prepbufr_dir, out_shapefile):
    tar_files = sorted(glob.glob(os.path.join(prepbufr_dir, "prepbufr.*.nr.tar.gz")))
    if not tar_files:
        print(f"No prepbufr tar.gz files found in {prepbufr_dir}")
        sys.exit(1)

    print(f"Found {len(tar_files)} daily tar files")
    stations = {}
    total_obs = 0

    for i, tar_path in enumerate(tar_files):
        day = os.path.basename(tar_path).split(".")[1]
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cycle_files = extract_and_read(tar_path, tmpdir)
                day_obs = 0
                for cf in cycle_files:
                    n = read_adpsfc(cf, stations)
                    day_obs += n
                total_obs += day_obs
            print(
                f"  [{i + 1}/{len(tar_files)}] {day}: {day_obs:,} obs, "
                f"{len(stations):,} stations cumulative"
            )
        except Exception as e:
            print(f"  [{i + 1}/{len(tar_files)}] {day}: ERROR - {e}")

    print(f"\nTotal: {total_obs:,} observations from {len(stations):,} stations")

    # Build GeoDataFrame
    rows = []
    for sid, info in stations.items():
        rows.append(
            {
                "station_id": sid,
                "latitude": info["lat"],
                "longitude": info["lon"],
                "elevation": info["elv"],
                "obs_type": info["typ"],
                "obs_count": info["count"],
                "has_p": info["has_p"],
                "has_t": info["has_t"],
                "has_q": info["has_q"],
                "has_wind": info["has_wind"],
                "geometry": Point(info["lon"], info["lat"]),
            }
        )

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf.to_file(out_shapefile, engine="fiona")
    print(f"Wrote {len(gdf)} stations to {out_shapefile}")


if __name__ == "__main__":
    prepbufr_dir = (
        sys.argv[1] if len(sys.argv) > 1 else "/nas/climate/gdas/prepbufr/2024"
    )
    out_shp = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "/nas/climate/gdas/prepbufr/adpsfc_stations_202407.shp"
    )
    main(prepbufr_dir, out_shp)
