"""
Builder for stations.zarr — a dense (time, station) observation store.

Replaces per-station parquet files with a single zarr store containing all
station observations and static metadata. Source data: QC'd training parquets
at ``{parquet_root}/{target}/{fid}.parquet`` across 6 target variables.

Output arrays
-------------
1-D coords (S,):
    station_id, lat, lon, easting, northing, elevation, station_row, station_col
1-D time coord (T,):
    time — int64 days since 1970-01-01
2-D observation arrays (T, S):
    tmax_obs, tmin_obs, ea_obs, prcp_obs, rsds_obs, wind_obs
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import zarr

from cube.config import get_compression
from cube.grid import create_conus_grid

logger = logging.getLogger(__name__)

TARGET_VARIABLES = [
    "tmax_obs",
    "tmin_obs",
    "ea_obs",
    "prcp_obs",
    "rsds_obs",
    "wind_obs",
]

_EPOCH = np.datetime64("1970-01-01", "D")


# ---------------------------------------------------------------------------
# Pass 1 helpers — run in worker processes
# ---------------------------------------------------------------------------


def _read_one_metadata(path: str) -> Tuple[str, float, float, float]:
    """Read lat, lon, elevation from one parquet (first row)."""
    df = pd.read_parquet(path, columns=["lat", "lon", "elevation"])
    row = df.iloc[0]
    fid = Path(path).stem
    return fid, float(row["lat"]), float(row["lon"]), float(row["elevation"])


def _read_one_column(
    args: Tuple[str, str, int, np.int64, np.int64],
) -> Optional[Tuple[int, np.ndarray, np.ndarray]]:
    """Read a single parquet, extract target column, return (local_col, day_offsets, values).

    Returns None if the file is corrupt or unreadable.

    Parameters
    ----------
    args : tuple
        (parquet_path, target_col, local_col_index, start_day, end_day)
        where start_day/end_day are int64 days-since-epoch.
    """
    path, target_col, local_col, start_day, end_day = args
    try:
        df = pd.read_parquet(path, columns=[target_col])
    except Exception:
        return None

    # Convert DatetimeIndex to day offsets
    days = df.index.values.astype("datetime64[D]").astype(np.int64)

    # Filter to [start_day, end_day]
    mask = (days >= start_day) & (days <= end_day)
    days = days[mask]
    vals = df[target_col].values[mask]

    # Clamp float64 values to float32 range before casting to avoid overflow
    f32_max = np.finfo(np.float32).max
    vals = np.clip(vals, -f32_max, f32_max).astype(np.float32)

    # Deduplicate — keep last occurrence
    _, unique_idx = np.unique(days, return_index=True)
    days = days[unique_idx]
    vals = vals[unique_idx]

    # Shift to 0-based offset from start_day
    offsets = days - start_day

    return local_col, offsets.astype(np.int64), vals


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_stations(
    parquet_root: Path,
    num_workers: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scan parquet directories and extract the union of station metadata.

    Parameters
    ----------
    parquet_root : Path
        Root containing per-target subdirectories (e.g. ``tmax_obs/``, ``ea_obs/``).
    num_workers : int
        Workers for parallel metadata reads.

    Returns
    -------
    fids : ndarray of str, shape (S,)
        Sorted station ids.
    lats, lons, elevations : ndarray of float32, shape (S,)
    """
    parquet_root = Path(parquet_root)

    # Collect union of fids and remember one parquet per fid
    fid_to_path: Dict[str, Path] = {}
    for target in TARGET_VARIABLES:
        target_dir = parquet_root / target
        if not target_dir.is_dir():
            logger.warning("Target dir not found: %s", target_dir)
            continue
        for p in target_dir.glob("*.parquet"):
            fid = p.stem
            if fid not in fid_to_path:
                fid_to_path[fid] = p

    if not fid_to_path:
        raise FileNotFoundError(f"No parquet files found under {parquet_root}")

    # Sort fids lexicographically
    fids_sorted = sorted(fid_to_path.keys())
    paths = [str(fid_to_path[f]) for f in fids_sorted]

    logger.info(
        "Discovered %d stations across %d target dirs",
        len(fids_sorted),
        len(TARGET_VARIABLES),
    )

    # Parallel metadata extraction
    lats = np.empty(len(fids_sorted), dtype=np.float32)
    lons = np.empty(len(fids_sorted), dtype=np.float32)
    elevations = np.empty(len(fids_sorted), dtype=np.float32)

    fid_to_idx = {f: i for i, f in enumerate(fids_sorted)}

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_read_one_metadata, p): p for p in paths}
        for fut in as_completed(futures):
            fid, lat, lon, elev = fut.result()
            idx = fid_to_idx[fid]
            lats[idx] = lat
            lons[idx] = lon
            elevations[idx] = elev

    fids_arr = np.array(fids_sorted, dtype=str)
    return fids_arr, lats, lons, elevations


# ---------------------------------------------------------------------------
# Store creation + ingest
# ---------------------------------------------------------------------------


def build_stations_zarr(
    parquet_root: str,
    output_path: str,
    start_date: str = "1990-01-01",
    end_date: str = "2025-12-31",
    num_workers: int = 12,
    chunk_stations: int = 1000,
    overwrite: bool = False,
) -> Path:
    """Build the stations.zarr store from per-station parquet files.

    Parameters
    ----------
    parquet_root : str
        Root with per-target subdirectories.
    output_path : str
        Output zarr store path.
    start_date, end_date : str
        Temporal range (inclusive).
    num_workers : int
        Workers for parallel I/O.
    chunk_stations : int
        Number of stations per write chunk (controls memory).
    overwrite : bool
        If False, raise if store already exists.

    Returns
    -------
    Path to the created zarr store.
    """
    parquet_root = Path(parquet_root)
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Store already exists: {output_path}. Use overwrite=True to rebuild."
        )

    # ---- Pass 1: Discovery ----
    logger.info("Pass 1: discovering stations …")
    fids, lats, lons, elevations = discover_stations(
        parquet_root, num_workers=num_workers
    )
    n_stations = len(fids)

    # Compute projected coords and grid indices
    grid = create_conus_grid()
    easting, northing = grid.latlon_to_xy(
        lats.astype(np.float64), lons.astype(np.float64)
    )
    rows, cols = grid.latlon_to_rowcol(lats.astype(np.float64), lons.astype(np.float64))

    # ---- Pass 2: Create store ----
    logger.info("Pass 2: creating store (S=%d) …", n_stations)
    time_index = pd.date_range(start_date, end_date, freq="D")
    n_times = len(time_index)
    times_int = time_index.values.astype("datetime64[D]").astype(np.int64)

    compressor = get_compression("float")

    store = zarr.open(str(output_path), mode="w")

    # Time coordinate
    t_arr = store.create_dataset(
        "time", shape=times_int.shape, data=times_int, chunks=(365,), dtype="int64"
    )
    t_arr.attrs["units"] = "days since 1970-01-01"
    t_arr.attrs["calendar"] = "standard"

    # Station metadata arrays
    store.create_dataset(
        "station_id",
        shape=fids.shape,
        data=fids,
        dtype=str,
        chunks=(min(chunk_stations, n_stations),),
    )
    store.create_dataset(
        "lat", shape=lats.shape, data=lats, dtype="float32", chunks=(n_stations,)
    )
    store.create_dataset(
        "lon", shape=lons.shape, data=lons, dtype="float32", chunks=(n_stations,)
    )
    store.create_dataset(
        "easting",
        shape=easting.shape,
        data=easting.astype(np.float64),
        dtype="float64",
        chunks=(n_stations,),
    )
    store.create_dataset(
        "northing",
        shape=northing.shape,
        data=northing.astype(np.float64),
        dtype="float64",
        chunks=(n_stations,),
    )
    store.create_dataset(
        "elevation",
        shape=elevations.shape,
        data=elevations,
        dtype="float32",
        chunks=(n_stations,),
    )
    store.create_dataset(
        "station_row",
        shape=rows.shape,
        data=rows.astype(np.int32),
        dtype="int32",
        chunks=(n_stations,),
    )
    store.create_dataset(
        "station_col",
        shape=cols.shape,
        data=cols.astype(np.int32),
        dtype="int32",
        chunks=(n_stations,),
    )

    # Empty 2-D observation arrays
    for target in TARGET_VARIABLES:
        store.create_dataset(
            target,
            shape=(n_times, n_stations),
            chunks=(365, min(chunk_stations, n_stations)),
            dtype="float32",
            fill_value=np.nan,
            compressors=compressor,
        )

    # ---- Pass 3: Ingest ----
    start_day = np.int64(times_int[0])
    end_day = np.int64(times_int[-1])
    fid_list: List[str] = list(fids)

    for target in TARGET_VARIABLES:
        target_dir = parquet_root / target
        if not target_dir.is_dir():
            logger.warning("Skipping %s — directory not found", target)
            continue

        # Build fid→path mapping for this variable
        target_paths: Dict[str, Path] = {}
        for p in target_dir.glob("*.parquet"):
            target_paths[p.stem] = p

        logger.info("Pass 3: ingesting %s (%d files) …", target, len(target_paths))
        arr = store[target]
        values_written = 0
        skipped = 0

        for chunk_start in range(0, n_stations, chunk_stations):
            chunk_end = min(chunk_start + chunk_stations, n_stations)
            chunk_width = chunk_end - chunk_start
            buffer = np.full((n_times, chunk_width), np.nan, dtype=np.float32)

            # Collect work items for this chunk
            work_items = []
            for local_col, global_idx in enumerate(range(chunk_start, chunk_end)):
                fid = fid_list[global_idx]
                if fid in target_paths:
                    work_items.append(
                        (str(target_paths[fid]), target, local_col, start_day, end_day)
                    )

            if not work_items:
                arr[:, chunk_start:chunk_end] = buffer
                continue

            # Parallel reads
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                futures = [pool.submit(_read_one_column, item) for item in work_items]
                for fut in as_completed(futures):
                    result = fut.result()
                    if result is None:
                        skipped += 1
                        continue
                    local_col, offsets, vals = result
                    buffer[offsets, local_col] = vals
                    values_written += len(vals)

            arr[:, chunk_start:chunk_end] = buffer

        if skipped:
            logger.warning("  %s: skipped %d corrupt files", target, skipped)
        logger.info("  %s: wrote %d values", target, values_written)

    logger.info("Done — store at %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CDR integration
# ---------------------------------------------------------------------------


def add_cdr_indices(
    stations_path: str,
    cube_path: str,
) -> None:
    """Add CDR pixel index arrays to stations.zarr.

    Creates ``cdr_row_idx`` and ``cdr_col_idx`` arrays of shape (n_stations,)
    mapping each station to its nearest CDR native pixel.

    Parameters
    ----------
    stations_path : str
        Path to stations.zarr (opened mode='a').
    cube_path : str
        Path to cube.zarr containing cdr_lat / cdr_lon coordinate arrays.
    """
    cube = zarr.open(str(cube_path), mode="r")
    cdr_lat = cube["cdr_lat"][:]
    cdr_lon = cube["cdr_lon"][:]

    cdr_lat_max = cdr_lat[0]
    cdr_lon_min = cdr_lon[0]
    n_cdr_lat = len(cdr_lat)
    n_cdr_lon = len(cdr_lon)
    res = abs(cdr_lat[1] - cdr_lat[0]) if len(cdr_lat) > 1 else 0.05

    store = zarr.open(str(stations_path), mode="a")
    sta_lat = store["lat"][:]
    sta_lon = store["lon"][:]

    cdr_row_idx = np.round((cdr_lat_max - sta_lat) / res).astype(np.int32)
    cdr_col_idx = np.round((sta_lon - cdr_lon_min) / res).astype(np.int32)

    np.clip(cdr_row_idx, 0, n_cdr_lat - 1, out=cdr_row_idx)
    np.clip(cdr_col_idx, 0, n_cdr_lon - 1, out=cdr_col_idx)

    compressor = get_compression("int")
    for name, data in [("cdr_row_idx", cdr_row_idx), ("cdr_col_idx", cdr_col_idx)]:
        if name in store:
            del store[name]
        store.create_dataset(
            name,
            shape=data.shape,
            data=data,
            dtype="int32",
            chunks=(len(data),),
            compressors=compressor,
        )

    logger.info(
        "Added cdr_row_idx, cdr_col_idx to %s (%d stations)",
        stations_path,
        len(sta_lat),
    )


def add_cdr_features(
    stations_path: str,
    cube_path: str,
    chunk_time: int = 365,
    chunk_stations: int = 1000,
) -> None:
    """Extract CDR features from cdr_native store into stations.zarr.

    Creates 12 new arrays in stations.zarr: sr1..bt3 (float32) and
    sr1_miss..bt3_miss (uint8), each of shape (n_time, n_stations).

    Requires ``cdr_row_idx`` and ``cdr_col_idx`` to already exist in
    stations.zarr (see ``add_cdr_indices``).

    Parameters
    ----------
    stations_path : str
        Path to stations.zarr.
    cube_path : str
        Path to cube.zarr containing cdr_native group.
    chunk_time : int
        Time chunk size for output arrays.
    chunk_stations : int
        Station chunk size for output arrays.
    """
    from cube.config import CDR_FEATURES, CDR_MISS_FEATURES

    store = zarr.open(str(stations_path), mode="a")
    cube = zarr.open(str(cube_path), mode="r")
    cdr_group = cube["cdr_native"]

    # Read station CDR indices
    cdr_row = store["cdr_row_idx"][:]
    cdr_col = store["cdr_col_idx"][:]
    n_stations = len(cdr_row)

    # Determine time dimension
    n_time = store["time"].shape[0]

    logger.info("Extracting CDR features: %d times x %d stations", n_time, n_stations)

    compressor_f = get_compression("float")
    compressor_i = get_compression("int")

    # Create output arrays
    all_vars = CDR_FEATURES + CDR_MISS_FEATURES
    for var in all_vars:
        is_miss = var.endswith("_miss")
        dtype = "uint8" if is_miss else "float32"
        fill = np.uint8(1) if is_miss else np.nan
        comp = compressor_i if is_miss else compressor_f

        if var in store:
            del store[var]
        store.create_dataset(
            var,
            shape=(n_time, n_stations),
            chunks=(min(chunk_time, n_time), min(chunk_stations, n_stations)),
            dtype=dtype,
            fill_value=fill,
            compressors=comp,
        )

    # Extract by time chunks
    for t_start in range(0, n_time, chunk_time):
        t_end = min(t_start + chunk_time, n_time)

        for var in all_vars:
            src = cdr_group[var]
            # Read the time slab from CDR native
            slab = src[t_start:t_end, :, :]  # (t_len, cdr_lat, cdr_lon)

            # Extract station pixels
            station_vals = slab[:, cdr_row, cdr_col]  # (t_len, n_stations)

            store[var][t_start:t_end, :] = station_vals

        if t_start % (chunk_time * 5) == 0:
            logger.info("  time %d / %d", t_start, n_time)

    logger.info("CDR feature extraction complete")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_stations_zarr(store_path: str) -> Dict[str, bool]:
    """Run basic integrity checks on a stations.zarr store.

    Returns dict mapping check names to pass/fail booleans.
    """
    store_path = Path(store_path)
    checks: Dict[str, bool] = {}

    checks["exists"] = store_path.exists()
    if not checks["exists"]:
        return checks

    store = zarr.open(str(store_path), mode="r")

    # Required arrays
    required_1d = [
        "time",
        "station_id",
        "lat",
        "lon",
        "easting",
        "northing",
        "elevation",
        "station_row",
        "station_col",
    ]
    for name in required_1d:
        checks[f"{name}_exists"] = name in store

    for target in TARGET_VARIABLES:
        checks[f"{target}_exists"] = target in store

    # Dimension consistency
    if "time" in store and "station_id" in store:
        n_times = store["time"].shape[0]
        n_stations = store["station_id"].shape[0]
        checks["time_len"] = n_times > 0
        checks["station_len"] = n_stations > 0

        for target in TARGET_VARIABLES:
            if target in store:
                checks[f"{target}_shape"] = store[target].shape == (n_times, n_stations)

    # Time attrs
    if "time" in store:
        checks["time_units"] = (
            store["time"].attrs.get("units") == "days since 1970-01-01"
        )
        checks["time_calendar"] = store["time"].attrs.get("calendar") == "standard"

    return checks
