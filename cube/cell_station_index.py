"""
Builder for cell_station_index.zarr — grid-cell-to-station neighbor map.

Precomputes the k-nearest stations for every grid cell, used at inference time
so each cell can look up its neighbor stations and load their recent obs from
stations.zarr.  Same GNN architecture applies — stations are always the
neighbors, whether the target is a station (training) or a grid cell (inference).

Output arrays (y, x, k):
    station_idx   — int32 indices into stations.zarr station dim (-1 for invalid)
    distance_km   — float32 Haversine distance
    bearing_sin   — float32 sin(bearing from cell to station)
    bearing_cos   — float32 cos(bearing from cell to station)

When ``elevation`` is provided for both grid cells and stations, the BallTree
query uses the same 3-D metric as graph.zarr:
    d = sqrt(dx² + dy² + (elev_scale·dz)²)
Otherwise it falls back to 2-D Euclidean over (easting, northing).
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import zarr
from sklearn.neighbors import BallTree

from cube.config import get_compression
from cube.graph import _haversine_bearing, _haversine_distance_km
from cube.grid import MasterGrid

logger = logging.getLogger(__name__)


def build_cell_station_index(
    stations_zarr: str,
    output_path: str,
    grid: Optional[MasterGrid] = None,
    k: int = 10,
    elev_scale: float = 10.0,
    grid_elevation: Optional[np.ndarray] = None,
    overwrite: bool = False,
) -> Path:
    """Build cell_station_index.zarr for a grid region.

    Parameters
    ----------
    stations_zarr : str
        Path to stations.zarr (source of station metadata).
    output_path : str
        Destination zarr store.
    grid : MasterGrid or None
        Grid definition.  Defaults to PNW test region.
    k : int
        Number of nearest-neighbor stations per cell.
    elev_scale : float
        Multiplier on elevation for 3-D distance (same as graph.zarr).
        Only used when ``grid_elevation`` is provided.
    grid_elevation : np.ndarray or None
        Optional (n_y, n_x) float array of cell elevations in metres.  When
        provided, enables elevation-aware 3-D neighbor selection matching
        the graph.zarr metric.
    overwrite : bool
        Raise if store exists and this is False.

    Returns
    -------
    Path to the created zarr store.
    """
    from cube.grid import create_test_region_grid

    stations_zarr = Path(stations_zarr)
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists. Use overwrite=True to rebuild.")

    if grid is None:
        grid = create_test_region_grid()

    n_y, n_x = grid.shape
    logger.info("Grid: %d x %d (%d cells), k=%d", n_y, n_x, n_y * n_x, k)

    # ------------------------------------------------------------------
    # 1. Load station metadata
    # ------------------------------------------------------------------
    logger.info("Step 1: loading station metadata from %s", stations_zarr)
    sz = zarr.open(str(stations_zarr), mode="r")
    lats = sz["lat"][:].astype(np.float64)
    lons = sz["lon"][:].astype(np.float64)
    eastings = sz["easting"][:].astype(np.float64)
    northings = sz["northing"][:].astype(np.float64)
    elevations = sz["elevation"][:].astype(np.float64)
    n_stations = len(lats)
    logger.info("  %d stations loaded", n_stations)

    # ------------------------------------------------------------------
    # 2. Build BallTree over stations
    # ------------------------------------------------------------------
    use_3d = grid_elevation is not None
    if use_3d:
        logger.info("Step 2: building 3-D BallTree (elev_scale=%.1f)", elev_scale)
        nan_elev = np.isnan(elevations)
        if nan_elev.any():
            med = np.nanmedian(elevations)
            elevations[nan_elev] = med
            logger.info(
                "  filled %d NaN station elevations with median %.1f",
                nan_elev.sum(),
                med,
            )
        station_pts = np.column_stack([eastings, northings, elev_scale * elevations])
    else:
        logger.info("Step 2: building 2-D BallTree (no grid elevation provided)")
        station_pts = np.column_stack([eastings, northings])

    tree = BallTree(station_pts, metric="euclidean")

    # ------------------------------------------------------------------
    # 3. Query k-nearest stations for every grid cell (row-batched)
    # ------------------------------------------------------------------
    logger.info("Step 3: querying %d nearest stations per cell", k)
    station_idx = np.full((n_y, n_x, k), -1, dtype=np.int32)
    distance_km = np.full((n_y, n_x, k), np.nan, dtype=np.float32)
    bearing_sin = np.full((n_y, n_x, k), np.nan, dtype=np.float32)
    bearing_cos = np.full((n_y, n_x, k), np.nan, dtype=np.float32)

    # Pre-compute cell geographic coordinates for Haversine
    cell_lat = grid.lat  # (n_y, n_x)
    cell_lon = grid.lon  # (n_y, n_x)

    x_1d = grid.x  # easting, shape (n_x,)
    y_1d = grid.y  # northing, shape (n_y,)

    log_interval = max(1, n_y // 20)
    for row in range(n_y):
        northing_val = y_1d[row]
        # Build query points for all cells in this row
        if use_3d:
            row_elev = grid_elevation[row, :]  # (n_x,)
            nan_mask = np.isnan(row_elev)
            if nan_mask.any():
                row_elev = row_elev.copy()
                row_elev[nan_mask] = 0.0
            query = np.column_stack(
                [
                    x_1d,
                    np.full(n_x, northing_val),
                    elev_scale * row_elev,
                ]
            )
        else:
            query = np.column_stack(
                [
                    x_1d,
                    np.full(n_x, northing_val),
                ]
            )

        _, idx = tree.query(query, k=k)  # (n_x, k)
        station_idx[row] = idx.astype(np.int32)

        # Haversine distance and bearing (geographic)
        row_lat = cell_lat[row, :]  # (n_x,)
        row_lon = cell_lon[row, :]  # (n_x,)

        for ki in range(k):
            nb_idx = idx[:, ki]  # (n_x,)
            nb_lat = lats[nb_idx]
            nb_lon = lons[nb_idx]
            distance_km[row, :, ki] = _haversine_distance_km(
                row_lat,
                row_lon,
                nb_lat,
                nb_lon,
            ).astype(np.float32)
            brg = _haversine_bearing(row_lat, row_lon, nb_lat, nb_lon)
            bearing_sin[row, :, ki] = np.sin(brg).astype(np.float32)
            bearing_cos[row, :, ki] = np.cos(brg).astype(np.float32)

        if row % log_interval == 0:
            logger.info("  row %d / %d (%.0f%%)", row, n_y, 100 * row / n_y)

    logger.info("  query complete")

    # ------------------------------------------------------------------
    # 4. Write zarr store
    # ------------------------------------------------------------------
    logger.info("Step 4: writing zarr store to %s", output_path)
    compressor_f = get_compression("float")
    compressor_i = get_compression("int")

    # Chunk strategy: contiguous spatial tiles for inference reads
    chunk_y = min(64, n_y)
    chunk_x = min(64, n_x)

    store = zarr.open(str(output_path), mode="w")

    store.create_dataset(
        "station_idx",
        shape=station_idx.shape,
        data=station_idx,
        dtype="int32",
        chunks=(chunk_y, chunk_x, k),
        compressors=compressor_i,
    )
    store.create_dataset(
        "distance_km",
        shape=distance_km.shape,
        data=distance_km,
        dtype="float32",
        chunks=(chunk_y, chunk_x, k),
        compressors=compressor_f,
    )
    store.create_dataset(
        "bearing_sin",
        shape=bearing_sin.shape,
        data=bearing_sin,
        dtype="float32",
        chunks=(chunk_y, chunk_x, k),
        compressors=compressor_f,
    )
    store.create_dataset(
        "bearing_cos",
        shape=bearing_cos.shape,
        data=bearing_cos,
        dtype="float32",
        chunks=(chunk_y, chunk_x, k),
        compressors=compressor_f,
    )

    # Root attrs
    store.attrs["k"] = k
    store.attrs["elev_scale"] = elev_scale if use_3d else None
    store.attrs["elevation_aware"] = use_3d
    store.attrs["grid"] = grid.to_dict()
    store.attrs["n_stations"] = int(n_stations)

    logger.info("Done — cell_station_index.zarr at %s", output_path)
    logger.info("  shape: (%d, %d, %d)", n_y, n_x, k)
    logger.info("  elevation-aware: %s", use_3d)
    return output_path


# ---------------------------------------------------------------------------
# CDR pixel index
# ---------------------------------------------------------------------------


def add_cdr_pixel_index(
    store_path: str,
    cube_path: str,
    grid: Optional[MasterGrid] = None,
) -> None:
    """Add CDR pixel index arrays to an existing cell_station_index.zarr.

    Creates ``cdr_row_idx`` and ``cdr_col_idx`` arrays of shape (n_y, n_x)
    that map each 1 km grid cell to its nearest CDR native pixel.

    Parameters
    ----------
    store_path : str
        Path to existing cell_station_index.zarr (opened mode='a').
    cube_path : str
        Path to cube.zarr containing cdr_lat / cdr_lon coordinate arrays.
    grid : MasterGrid, optional
        Grid definition. Defaults to PNW test region.
    """
    from cube.grid import create_test_region_grid

    store_path = Path(store_path)
    cube_path = Path(cube_path)

    if grid is None:
        grid = create_test_region_grid()

    # Read CDR coordinate arrays from cube.zarr
    cube = zarr.open(str(cube_path), mode="r")
    cdr_lat = cube["cdr_lat"][:]  # 1-D, N to S
    cdr_lon = cube["cdr_lon"][:]  # 1-D, W to E

    cdr_lat_max = cdr_lat[0]
    cdr_lon_min = cdr_lon[0]
    n_cdr_lat = len(cdr_lat)
    n_cdr_lon = len(cdr_lon)
    res = abs(cdr_lat[1] - cdr_lat[0]) if len(cdr_lat) > 1 else 0.05

    logger.info("CDR grid: %d lat x %d lon, res=%.4f", n_cdr_lat, n_cdr_lon, res)

    # Get 2-D lat/lon for every grid cell
    cell_lat = grid.lat  # (n_y, n_x)
    cell_lon = grid.lon  # (n_y, n_x)

    # Vectorized index computation
    cdr_row_idx = np.round((cdr_lat_max - cell_lat) / res).astype(np.int32)
    cdr_col_idx = np.round((cell_lon - cdr_lon_min) / res).astype(np.int32)

    # Clamp to valid range
    np.clip(cdr_row_idx, 0, n_cdr_lat - 1, out=cdr_row_idx)
    np.clip(cdr_col_idx, 0, n_cdr_lon - 1, out=cdr_col_idx)

    # Write to store (append mode)
    compressor = get_compression("int")
    n_y, n_x = grid.shape
    chunk_y = min(64, n_y)
    chunk_x = min(64, n_x)

    store = zarr.open(str(store_path), mode="a")
    for name, data in [("cdr_row_idx", cdr_row_idx), ("cdr_col_idx", cdr_col_idx)]:
        if name in store:
            del store[name]
        store.create_dataset(
            name,
            shape=data.shape,
            data=data,
            dtype="int32",
            chunks=(chunk_y, chunk_x),
            compressors=compressor,
        )

    logger.info(
        "Added cdr_row_idx, cdr_col_idx to %s (shape %s)", store_path, (n_y, n_x)
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_cell_station_index(store_path: str) -> Dict[str, bool]:
    """Run integrity checks on a cell_station_index.zarr store."""
    store_path = Path(store_path)
    checks: Dict[str, bool] = {}

    checks["exists"] = store_path.exists()
    if not checks["exists"]:
        return checks

    store = zarr.open(str(store_path), mode="r")

    required = ["station_idx", "distance_km", "bearing_sin", "bearing_cos"]
    for name in required:
        checks[f"{name}_exists"] = name in store
    if not all(checks.get(f"{n}_exists", False) for n in required):
        return checks

    si = store["station_idx"]
    n_y, n_x, k = si.shape

    for name in required:
        checks[f"{name}_shape"] = store[name].shape == (n_y, n_x, k)

    checks["station_idx_dtype"] = si.dtype == np.int32
    checks["distance_km_dtype"] = store["distance_km"].dtype == np.float32

    # Sample a small slab to check values
    sample_si = si[0, : min(100, n_x), :]
    n_sta = store.attrs.get("n_stations", None)
    if n_sta is not None:
        checks["station_idx_range"] = bool(
            np.all((sample_si >= -1) & (sample_si < n_sta))
        )

    sample_dist = store["distance_km"][0, : min(100, n_x), :]
    valid = sample_si >= 0
    if valid.any():
        checks["distance_positive"] = bool(np.all(sample_dist[valid] >= 0))

    sample_bs = store["bearing_sin"][0, : min(100, n_x), :]
    sample_bc = store["bearing_cos"][0, : min(100, n_x), :]
    if valid.any():
        checks["bearing_sin_range"] = bool(
            np.all(sample_bs[valid] >= -1.0) and np.all(sample_bs[valid] <= 1.0)
        )
        checks["bearing_cos_range"] = bool(
            np.all(sample_bc[valid] >= -1.0) and np.all(sample_bc[valid] <= 1.0)
        )

    for attr in ["k", "grid", "n_stations"]:
        checks[f"attr_{attr}"] = attr in store.attrs

    return checks


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    stations = (
        sys.argv[1] if len(sys.argv) > 1 else "/data/ssd2/dads_cube/stations.zarr"
    )
    output = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "/data/ssd2/dads_cube/cell_station_index.zarr"
    )
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    build_cell_station_index(
        stations_zarr=stations,
        output_path=output,
        k=k,
        overwrite=True,
    )
