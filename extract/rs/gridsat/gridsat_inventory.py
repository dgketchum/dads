"""Inventory the NOAA GridSat-B1 CDR archive on S3.

Lists files per year, downloads one sample NetCDF to extract variable metadata,
and writes a JSON inventory to extract/rs/gridsat/gridsat_b1_inventory.json.
"""

import json
import os
import tempfile

import boto3
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "noaa-cdr-gridsat-b1-pds"
PREFIX = "data/"
OUT_JSON = os.path.join(os.path.dirname(__file__), "gridsat_b1_inventory.json")


def list_years(s3):
    """Return sorted list of year prefixes in the bucket."""
    paginator = s3.get_paginator("list_objects_v2")
    years = set()
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            yr = cp["Prefix"].rstrip("/").split("/")[-1]
            if yr.isdigit():
                years.add(int(yr))
    return sorted(years)


def count_files_per_year(s3, years):
    """Count .nc files per year using pagination."""
    counts = {}
    for yr in years:
        prefix = f"{PREFIX}{yr}/"
        paginator = s3.get_paginator("list_objects_v2")
        n = 0
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            n += sum(1 for k in page.get("Contents", []) if k["Key"].endswith(".nc"))
        counts[yr] = n
        print(f"  {yr}: {n} files")
    return counts


def sample_variable_metadata(s3):
    """Download one sample file and extract variable info."""
    # Grab first file from 2020
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}2020/", MaxKeys=2)
    sample_key = None
    for obj in resp.get("Contents", []):
        if obj["Key"].endswith(".nc"):
            sample_key = obj["Key"]
            sample_size_bytes = obj["Size"]
            break

    if sample_key is None:
        raise RuntimeError("No sample file found in 2020/")

    local = os.path.join(tempfile.gettempdir(), os.path.basename(sample_key))
    print(f"Downloading sample: {sample_key} ({sample_size_bytes / 1e6:.1f} MB)")
    s3.download_file(BUCKET, sample_key, local)

    ds = xr.open_dataset(local)
    variables = {}
    for name, var in ds.data_vars.items():
        attrs = var.attrs
        variables[name] = {
            "long_name": attrs.get("long_name", ""),
            "units": attrs.get("units", ""),
            "dtype": str(var.dtype),
            "dimensions": list(var.dims),
            "shape": list(var.shape),
            "scale_factor": float(attrs["scale_factor"])
            if "scale_factor" in attrs
            else None,
            "add_offset": float(attrs["add_offset"]) if "add_offset" in attrs else None,
            "_FillValue": float(attrs["_FillValue"]) if "_FillValue" in attrs else None,
        }

    coord_info = {}
    for name, coord in ds.coords.items():
        coord_info[name] = {
            "size": int(coord.size),
            "min": float(coord.values.min()),
            "max": float(coord.values.max()),
            "units": coord.attrs.get("units", ""),
        }

    ds.close()
    os.remove(local)
    return variables, coord_info, sample_key, sample_size_bytes


def main():
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    print("Listing years...")
    years = list_years(s3)
    print(f"Found {len(years)} years: {years[0]}–{years[-1]}")

    print("Counting files per year...")
    counts = count_files_per_year(s3, years)

    print("Extracting variable metadata from sample file...")
    variables, coords, sample_key, sample_size = sample_variable_metadata(s3)

    inventory = {
        "bucket": BUCKET,
        "temporal_range": {"start_year": years[0], "end_year": years[-1]},
        "files_per_year": counts,
        "total_files": sum(counts.values()),
        "sample_file": {
            "key": sample_key,
            "size_bytes": sample_size,
        },
        "variables": variables,
        "coordinates": coords,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(inventory, f, indent=2)
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
