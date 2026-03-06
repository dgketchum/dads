"""Download GridSat-B1 for a date range and export clipped COG GeoTIFFs.

Downloads all 8 daily timesteps, subsets to URMA extent (+ 1° buffer),
writes per-channel COGs, and deletes the raw NetCDF.  Uses a process pool
to avoid HDF5/NetCDF4 thread-safety issues.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import boto3
import numpy as np
import rasterio
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config
from rasterio.transform import from_bounds

BUCKET = "noaa-cdr-gridsat-b1-pds"
HOURS = ["00", "03", "06", "09", "12", "15", "18", "21"]
CHANNELS = ["vschn", "irwin_cdr", "irwvp"]
# URMA extent + 1° buffer
CLIP_BOUNDS = {"lat_min": 18.2, "lat_max": 55.4, "lon_min": -139.4, "lon_max": -58.0}


def download_and_clip(date_str, hh, out_dir, tmp_dir):
    """Download one timestep, clip to bounds, write COGs, return # written.

    Args use strings/primitives for pickling across process boundary.
    """
    date = datetime.strptime(date_str, "%Y-%m-%d")
    ds_label = date.strftime("%Y%m%d")

    # Skip if all 3 channel COGs already exist
    if all(
        os.path.exists(os.path.join(out_dir, f"gridsat_b1_{ds_label}_{hh}00_{ch}.tif"))
        for ch in CHANNELS
    ):
        return 0

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    date_dot = date.strftime("%Y.%m.%d")
    year = date.strftime("%Y")
    key = f"data/{year}/GRIDSAT-B1.{date_dot}.{hh}.v02r01.nc"
    local = os.path.join(tmp_dir, f"p{os.getpid()}_{date_dot}.{hh}.nc")

    try:
        s3.download_file(BUCKET, key, local)
    except Exception as e:
        if "404" in str(e) or "NoSuchKey" in str(e):
            return 0
        raise

    ds = xr.open_dataset(local, decode_times=True)
    sub = ds.sel(
        lat=slice(CLIP_BOUNDS["lat_min"], CLIP_BOUNDS["lat_max"]),
        lon=slice(CLIP_BOUNDS["lon_min"], CLIP_BOUNDS["lon_max"]),
    )

    lat = sub["lat"].values
    lon = sub["lon"].values

    written = 0
    for ch in CHANNELS:
        if ch not in sub.data_vars:
            continue

        data = sub[ch].values[0].astype(np.float32)

        if lat[0] < lat[-1]:
            data = data[::-1]
            lat_ordered = lat[::-1]
        else:
            lat_ordered = lat

        nrows, ncols = data.shape
        dlat = abs(lat_ordered[0] - lat_ordered[1])
        dlon = abs(lon[1] - lon[0])
        transform = from_bounds(
            lon[0] - dlon / 2,
            lat_ordered[-1] - dlat / 2,
            lon[-1] + dlon / 2,
            lat_ordered[0] + dlat / 2,
            ncols,
            nrows,
        )

        out_name = f"gridsat_b1_{ds_label}_{hh}00_{ch}.tif"
        out_path = os.path.join(out_dir, out_name)
        with rasterio.open(
            out_path,
            "w",
            driver="COG",
            height=nrows,
            width=ncols,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data, 1)
        written += 1

    ds.close()
    os.remove(local)

    return written


def main():
    parser = argparse.ArgumentParser(description="Download GridSat-B1 clipped COGs")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--out", required=True, help="Output root directory for COGs")
    parser.add_argument("--tmp", default="/tmp/gridsat_nc", help="Temp dir for raw NCs")
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel download processes"
    )
    args = parser.parse_args()

    os.makedirs(args.tmp, exist_ok=True)

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    # Process one month at a time
    cur = start.replace(day=1)
    grand_total = 0
    while cur <= end:
        year_dir = os.path.join(args.out, cur.strftime("%Y"))
        os.makedirs(year_dir, exist_ok=True)

        tasks = []
        date = max(cur, start)
        if cur.month == 12:
            next_month = cur.replace(year=cur.year + 1, month=1)
        else:
            next_month = cur.replace(month=cur.month + 1)
        while date < next_month and date <= end:
            for hh in HOURS:
                tasks.append((date.strftime("%Y-%m-%d"), hh, year_dir))
            date += timedelta(days=1)

        month_label = cur.strftime("%Y-%m")
        month_total = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(download_and_clip, ds, hh, yd, args.tmp): (ds, hh)
                for ds, hh, yd in tasks
            }
            for future in as_completed(futures):
                ds, hh = futures[future]
                try:
                    month_total += future.result()
                except Exception as e:
                    print(f"ERROR {ds} {hh}Z: {e}", flush=True)

        grand_total += month_total
        print(f"{month_label}: {month_total} COGs ({grand_total} total)", flush=True)
        cur = next_month

    print(f"\nDone. {grand_total} COGs written to {args.out}")


if __name__ == "__main__":
    main()
