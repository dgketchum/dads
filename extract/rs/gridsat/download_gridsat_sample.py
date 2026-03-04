"""Download GridSat-B1 sample day and export PNW subsets as COG GeoTIFFs.

Downloads all 8 timesteps for 2020-08-18, subsets to PNW bounds,
and writes per-channel COGs to extract/rs/gridsat/sample_tifs/.
"""

import os

import boto3
import numpy as np
import rasterio
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config
from rasterio.transform import from_bounds

BUCKET = "noaa-cdr-gridsat-b1-pds"
SAMPLE_DATE = "2020.08.18"
HOURS = ["00", "03", "06", "09", "12", "15", "18", "21"]
CHANNELS = ["vschn", "irwin_cdr", "irwvp"]
PNW_BOUNDS = {"lat_min": 42.0, "lat_max": 49.5, "lon_min": -125.0, "lon_max": -110.0}
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_tifs")
TMP_DIR = "/tmp/gridsat_sample"


def download_files(s3):
    """Download all 8 timesteps for the sample date."""
    os.makedirs(TMP_DIR, exist_ok=True)
    paths = []
    for hh in HOURS:
        key = f"data/2020/GRIDSAT-B1.{SAMPLE_DATE}.{hh}.v02r01.nc"
        local = os.path.join(TMP_DIR, os.path.basename(key))
        if not os.path.exists(local):
            print(f"Downloading {key}...")
            s3.download_file(BUCKET, key, local)
        else:
            print(f"Already have {os.path.basename(local)}")
        paths.append(local)
    return paths


def subset_and_export(nc_path):
    """Open one NetCDF, subset to PNW, write COGs for each channel."""
    ds = xr.open_dataset(nc_path, decode_times=True)

    # Subset to PNW (lat is ascending in GridSat)
    sub = ds.sel(
        lat=slice(PNW_BOUNDS["lat_min"], PNW_BOUNDS["lat_max"]),
        lon=slice(PNW_BOUNDS["lon_min"], PNW_BOUNDS["lon_max"]),
    )

    lat = sub["lat"].values
    lon = sub["lon"].values

    # Extract timestamp string from filename
    basename = os.path.basename(nc_path)
    parts = basename.replace("GRIDSAT-B1.", "").replace(".v02r01.nc", "").split(".")
    date_str = "".join(parts[:3])  # YYYYMMDD
    hour_str = parts[3]  # HH

    results = []
    for ch in CHANNELS:
        if ch not in sub.data_vars:
            print(f"  WARNING: {ch} not in {basename}, skipping")
            continue

        # Extract 2D numpy array (time=0)
        data = sub[ch].values[0].astype(np.float32)  # (lat, lon)

        # Flip to north-up if lat is ascending
        if lat[0] < lat[-1]:
            data = data[::-1]
            lat_ordered = lat[::-1]
        else:
            lat_ordered = lat

        nrows, ncols = data.shape
        # Half-pixel pad for bounds
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

        out_name = f"gridsat_b1_{date_str}_{hour_str}00_{ch}.tif"
        out_path = os.path.join(SAMPLE_DIR, out_name)
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

        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        results.append((ch, f"{hour_str}:00", data.shape, vmin, vmax))

    ds.close()
    return results


def main():
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    print("Downloading GridSat-B1 files for 2020-08-18...")
    paths = download_files(s3)

    print(f"\nExporting PNW subsets to {SAMPLE_DIR}/")
    all_results = []
    for p in paths:
        print(f"Processing {os.path.basename(p)}...")
        all_results.extend(subset_and_export(p))

    print(f"\n{'Channel':<12} {'Time':>6} {'Shape':>12} {'Min':>10} {'Max':>10}")
    print("-" * 54)
    for ch, time, shape, vmin, vmax in all_results:
        print(f"{ch:<12} {time:>6} {str(shape):>12} {vmin:>10.2f} {vmax:>10.2f}")

    print(f"\nWrote {len(all_results)} GeoTIFFs to {SAMPLE_DIR}/")


if __name__ == "__main__":
    main()
