"""
Station-derived harmonic tmax climatology layer builder for the data cube.

Builds DOY-indexed tmax climatology from station observations via:
1. Per-station 2-harmonic fit on DOY means
2. Lapse-rate Ridge regression of each coefficient (train stations only)
3. IDW residual interpolation to fill spatial detail
4. Evaluation at each DOY to produce the final surface

Used as the fine-tuning/production background surface: B(x, doy).

Input: stations.zarr (tmax_obs, lat, lon, elevation, easting, northing)
       graph.zarr (is_train for train/val split)
       cube.zarr/static/elevation (grid-cell elevations)

Output: cube.zarr/doy_indexed/tmax_clim with shape (365, n_y, n_x)
        Units: degrees Celsius
"""

from __future__ import annotations

import datetime
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import zarr

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

from cube.config import CHUNKS
from cube.layers.base import BaseLayer

logger = logging.getLogger(__name__)

N_HARMONICS_DEFAULT = 2
MIN_YEARS_DEFAULT = 5
MIN_R2 = 0.5
ELEV_SCALE_DEFAULT = 10.0
K_IDW_DEFAULT = 20
IDW_POWER_DEFAULT = 2


def _date_to_doy365(month: int, day: int) -> int:
    """Map (month, day) to DOY in [1, 365], with Feb 29 mapped to 60 (Mar 1)."""
    if month == 2 and day == 29:
        return 60
    return datetime.date(2001, month, day).timetuple().tm_yday


def _compute_doy_means(tmax_obs, time_days):
    """Compute per-station DOY means from tmax observations.

    Args:
        tmax_obs: (n_time, n_stations) array of tmax observations (NaN for missing)
        time_days: (n_time,) array of days since epoch (int)

    Returns:
        doy_means: (365, n_stations) array of DOY means (NaN where insufficient data)
        n_years: (n_stations,) array of number of years with any valid data
    """
    epoch = datetime.date(1970, 1, 1)
    n_time, n_stations = tmax_obs.shape

    # Pre-compute DOY for each time step
    doys = np.empty(n_time, dtype=np.int32)
    years_per_station = [set() for _ in range(n_stations)]

    for t in range(n_time):
        dt = epoch + datetime.timedelta(days=int(time_days[t]))
        doys[t] = _date_to_doy365(dt.month, dt.day)

    # Accumulate sums/counts per DOY per station
    doy_sum = np.zeros((365, n_stations), dtype=np.float64)
    doy_count = np.zeros((365, n_stations), dtype=np.int32)

    for t in range(n_time):
        d_idx = doys[t] - 1  # 0-indexed
        valid = ~np.isnan(tmax_obs[t, :])
        doy_sum[d_idx, valid] += tmax_obs[t, valid]
        doy_count[d_idx, valid] += 1

        # Track years per station
        dt = epoch + datetime.timedelta(days=int(time_days[t]))
        yr = dt.year
        valid_idx = np.where(valid)[0]
        for si in valid_idx:
            years_per_station[si].add(yr)

    n_years = np.array([len(s) for s in years_per_station], dtype=np.int32)

    doy_means = np.where(doy_count > 0, doy_sum / doy_count, np.nan).astype(np.float32)
    return doy_means, n_years


def _build_harmonic_basis(n_harmonics=2):
    """Build (365, 1 + 2*n_harmonics) harmonic basis matrix.

    Columns: [1, sin(2pi*d/365), cos(2pi*d/365), sin(4pi*d/365), cos(4pi*d/365), ...]
    where d = 1..365.
    """
    d = np.arange(1, 366, dtype=np.float64)
    cols = [np.ones(365)]
    for h in range(1, n_harmonics + 1):
        cols.append(np.sin(2 * np.pi * h * d / 365.0))
        cols.append(np.cos(2 * np.pi * h * d / 365.0))
    return np.column_stack(cols)


def _fit_harmonics(doy_means, n_harmonics=2):
    """Fit harmonic basis to per-station DOY means.

    Args:
        doy_means: (365, n_stations) DOY means (may contain NaN)
        n_harmonics: number of harmonic pairs

    Returns:
        coeffs: (n_stations, 1 + 2*n_harmonics) fitted coefficients
        r2: (n_stations,) R² of fit (NaN where fit failed)
    """
    n_stations = doy_means.shape[1]
    n_coeffs = 1 + 2 * n_harmonics
    basis = _build_harmonic_basis(n_harmonics)

    coeffs = np.full((n_stations, n_coeffs), np.nan, dtype=np.float64)
    r2 = np.full(n_stations, np.nan, dtype=np.float64)

    for s in range(n_stations):
        y = doy_means[:, s]
        valid = ~np.isnan(y)
        n_valid = valid.sum()
        if n_valid < n_coeffs + 1:
            continue

        B = basis[valid]
        yv = y[valid].astype(np.float64)

        try:
            c, _, _, _ = np.linalg.lstsq(B, yv, rcond=None)
        except np.linalg.LinAlgError:
            continue

        y_pred = B @ c
        ss_res = np.sum((yv - y_pred) ** 2)
        ss_tot = np.sum((yv - yv.mean()) ** 2)
        if ss_tot > 0:
            r2[s] = 1.0 - ss_res / ss_tot
        else:
            r2[s] = 0.0

        coeffs[s] = c

    return coeffs, r2


def _fit_regression(coeffs, elevation, lat, lon, is_train):
    """Fit Ridge regression of each harmonic coefficient against terrain features.

    Uses train stations only. Predictors: [elevation, elevation², lat, lon, lat*elevation]

    Args:
        coeffs: (n_stations, n_coeffs) harmonic coefficients
        elevation: (n_stations,) station elevations in meters
        lat: (n_stations,) station latitudes
        lon: (n_stations,) station longitudes
        is_train: (n_stations,) boolean mask for train stations

    Returns:
        models: list of fitted Ridge models (one per coefficient)
        train_preds: (n_train, n_coeffs) predictions at train stations
    """
    from sklearn.linear_model import Ridge

    n_coeffs = coeffs.shape[1]
    elev = elevation.astype(np.float64)
    X = np.column_stack(
        [
            elev,
            elev**2,
            lat.astype(np.float64),
            lon.astype(np.float64),
            lat.astype(np.float64) * elev,
        ]
    )

    train_mask = is_train.astype(bool)
    X_train = X[train_mask]

    models = []
    for ci in range(n_coeffs):
        y_train = coeffs[train_mask, ci]
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        models.append(model)

    return models, X


def _predict_regression(models, elevation_grid, lat_grid, lon_grid):
    """Predict coefficient surfaces using fitted regression models.

    Args:
        models: list of fitted Ridge models
        elevation_grid: (n_y, n_x) grid elevations
        lat_grid: (n_y, n_x) grid latitudes
        lon_grid: (n_y, n_x) grid longitudes

    Returns:
        coeff_surfaces: (n_coeffs, n_y, n_x) predicted coefficient surfaces
    """
    n_y, n_x = elevation_grid.shape
    elev_flat = elevation_grid.ravel().astype(np.float64)
    lat_flat = lat_grid.ravel().astype(np.float64)
    lon_flat = lon_grid.ravel().astype(np.float64)

    # Mask out cells with NaN elevation (ocean / no-data)
    valid = ~np.isnan(elev_flat)
    elev_valid = np.where(valid, elev_flat, 0.0)

    X_grid = np.column_stack(
        [
            elev_valid,
            elev_valid**2,
            lat_flat,
            lon_flat,
            lat_flat * elev_valid,
        ]
    )

    n_coeffs = len(models)
    coeff_surfaces = np.full((n_coeffs, n_y * n_x), np.nan, dtype=np.float64)

    for ci, model in enumerate(models):
        pred = model.predict(X_grid[valid])
        coeff_surfaces[ci, valid] = pred

    coeff_surfaces = coeff_surfaces.reshape(n_coeffs, n_y, n_x)
    return coeff_surfaces


def _idw_residuals(residuals, station_pts_3d, grid_pts_3d, k=20, power=2):
    """IDW-interpolate station residuals to grid points.

    Args:
        residuals: (n_stations,) residual values
        station_pts_3d: (n_stations, 3) station coords [easting, northing, scaled_elev]
        grid_pts_3d: (n_grid, 3) grid coords [easting, northing, scaled_elev]
        k: number of nearest neighbors
        power: IDW power parameter

    Returns:
        grid_residuals: (n_grid,) interpolated residuals
    """
    from sklearn.neighbors import BallTree

    tree = BallTree(station_pts_3d, metric="euclidean")
    k_actual = min(k, len(station_pts_3d))

    dist, idx = tree.query(grid_pts_3d, k=k_actual)

    # Avoid division by zero (exact matches)
    dist = np.maximum(dist, 1e-10)
    weights = 1.0 / dist**power

    neighbor_resid = residuals[idx]  # (n_grid, k)
    grid_residuals = np.sum(weights * neighbor_resid, axis=1) / np.sum(weights, axis=1)

    return grid_residuals


def _evaluate_harmonics(coeff_surfaces, n_harmonics=2):
    """Evaluate harmonic surfaces at all 365 DOYs.

    Args:
        coeff_surfaces: (n_coeffs, n_y, n_x) coefficient surfaces

    Returns:
        clim: (365, n_y, n_x) climatology surface
    """
    basis = _build_harmonic_basis(n_harmonics)  # (365, n_coeffs)

    # Reshape for broadcasting: basis (365, n_coeffs, 1, 1) * surfaces (1, n_coeffs, n_y, n_x)
    clim = np.einsum("dc,cyx->dyx", basis, coeff_surfaces)
    return clim.astype(np.float32)


class TmaxClimLayer(BaseLayer):
    """
    Builds DOY-indexed station-derived tmax climatology for fine-tuning/production.

    Computes per-station harmonic fits, fits lapse-rate regressions on train
    stations, IDW-interpolates residuals, and evaluates at all 365 DOYs.

    Output: (365, y, x) array in degrees Celsius.
    """

    @property
    def name(self) -> str:
        return "doy_indexed"

    @property
    def variables(self) -> List[str]:
        return ["tmax_clim"]

    @property
    def dimensions(self) -> Tuple[str, ...]:
        return ("doy", "y", "x")

    @property
    def chunks(self) -> Dict[str, int]:
        return CHUNKS["doy_indexed"]

    def build(
        self,
        source_paths: Optional[Dict[str, str]] = None,
        overwrite: bool = False,
        stations_zarr: Optional[str] = None,
        graph_zarr: Optional[str] = None,
        min_years: int = MIN_YEARS_DEFAULT,
        n_harmonics: int = N_HARMONICS_DEFAULT,
        elev_scale: float = ELEV_SCALE_DEFAULT,
        k_idw: int = K_IDW_DEFAULT,
        idw_power: int = IDW_POWER_DEFAULT,
    ) -> None:
        """Build tmax_clim layer from station observations.

        Args:
            source_paths: Dict with 'stations_zarr' and 'graph_zarr' keys
            overwrite: Whether to overwrite existing data
            stations_zarr: Path to stations.zarr (overrides source_paths)
            graph_zarr: Path to graph.zarr (overrides source_paths)
            min_years: Minimum years of data for a station to be included
            n_harmonics: Number of harmonic pairs in the basis
            elev_scale: Multiplier for elevation in 3D BallTree distance
            k_idw: Number of IDW neighbors
            idw_power: IDW distance exponent
        """
        source_paths = source_paths or self.config.source_paths

        sz_path = stations_zarr or source_paths.get("stations_zarr")
        gz_path = graph_zarr or source_paths.get("graph_zarr")

        if sz_path is None:
            raise ValueError("stations_zarr path not provided")
        if gz_path is None:
            raise ValueError("graph_zarr path not provided")

        logger.info("Building tmax_clim layer")
        logger.info("  stations_zarr: %s", sz_path)
        logger.info("  graph_zarr:    %s", gz_path)

        # --- Open store and check for existing data ---
        store = self._open_store("a")
        self._write_coords(store)
        self._write_doy_coord(store)
        group = self._ensure_group(store)

        if "tmax_clim" in group and not overwrite:
            logger.info("tmax_clim already exists, skipping (use overwrite=True)")
            return

        # --- Load station data ---
        sz = zarr.open(str(sz_path), mode="r")
        tmax_obs = sz["tmax_obs"][:]  # (n_time, n_stations)
        time_days = sz["time"][:]  # (n_time,)
        stn_lat = sz["lat"][:]  # (n_stations,)
        stn_lon = sz["lon"][:]
        stn_elev = sz["elevation"][:]
        stn_easting = sz["easting"][:]
        stn_northing = sz["northing"][:]

        n_stations = len(stn_lat)
        logger.info("  Loaded %d stations", n_stations)

        # --- Load train/val split ---
        gz = zarr.open(str(gz_path), mode="r")
        is_train = gz["is_train"][:].astype(bool)  # (n_stations,)

        # --- Step 1: Compute DOY means and fit harmonics ---
        logger.info("  Step 1: Computing DOY means and harmonic fits")
        doy_means, n_years = _compute_doy_means(tmax_obs, time_days)

        # Filter stations with enough data
        enough_years = n_years >= min_years
        logger.info(
            "    %d / %d stations have >= %d years",
            enough_years.sum(),
            n_stations,
            min_years,
        )

        coeffs, r2 = _fit_harmonics(doy_means, n_harmonics)

        # Filter by R², enough years, and valid metadata (no NaN in coords/elev)
        valid_meta = (
            ~np.isnan(stn_elev)
            & ~np.isnan(stn_lat)
            & ~np.isnan(stn_lon)
            & ~np.isnan(stn_easting)
            & ~np.isnan(stn_northing)
        )
        good_fit = (~np.isnan(r2)) & (r2 >= MIN_R2) & enough_years & valid_meta
        n_good = good_fit.sum()
        logger.info("    %d stations pass quality filters (R² >= %.1f)", n_good, MIN_R2)

        if n_good < 10:
            raise RuntimeError(
                f"Only {n_good} stations pass quality filters — need at least 10"
            )

        # Subset to good stations
        coeffs_good = coeffs[good_fit]
        stn_lat_good = stn_lat[good_fit]
        stn_lon_good = stn_lon[good_fit]
        stn_elev_good = stn_elev[good_fit]
        stn_easting_good = stn_easting[good_fit]
        stn_northing_good = stn_northing[good_fit]
        is_train_good = is_train[good_fit]

        # --- Step 2: Lapse-rate regression (train only) ---
        logger.info(
            "  Step 2: Ridge regression on train stations (%d train)",
            is_train_good.sum(),
        )
        models, X_all = _fit_regression(
            coeffs_good, stn_elev_good, stn_lat_good, stn_lon_good, is_train_good
        )

        # Load grid elevation and lat/lon
        elev_grid = store["static"]["elevation"][:]  # (n_y, n_x)
        lat_grid = store["lat"][:]  # (n_y, n_x)
        lon_grid = store["lon"][:]  # (n_y, n_x)

        coeff_surfaces_reg = _predict_regression(models, elev_grid, lat_grid, lon_grid)

        # --- Step 3: IDW residual interpolation ---
        logger.info("  Step 3: IDW residual interpolation (k=%d)", k_idw)

        # Predict regression at station locations
        n_coeffs = coeffs_good.shape[1]
        station_reg_preds = np.column_stack(
            [m.predict(X_all) for m in models]
        )  # (n_good_stations, n_coeffs)

        residuals_all = coeffs_good - station_reg_preds  # (n_good_stations, n_coeffs)

        # Build 3D station and grid points for BallTree
        station_pts_3d = np.column_stack(
            [
                stn_easting_good,
                stn_northing_good,
                stn_elev_good * elev_scale,
            ]
        )

        # Grid points: flatten valid cells
        xx, yy = np.meshgrid(self.grid.x, self.grid.y)
        grid_easting_flat = xx.ravel()
        grid_northing_flat = yy.ravel()
        grid_elev_flat = elev_grid.ravel()

        # NaN-mask for ocean/missing elevation
        land_mask = ~np.isnan(grid_elev_flat)
        n_land = land_mask.sum()
        logger.info("    %d land cells (of %d total)", n_land, len(grid_elev_flat))

        grid_pts_3d_land = np.column_stack(
            [
                grid_easting_flat[land_mask],
                grid_northing_flat[land_mask],
                np.nan_to_num(grid_elev_flat[land_mask], nan=0.0) * elev_scale,
            ]
        )

        # IDW for each coefficient
        coeff_surfaces_final = coeff_surfaces_reg.copy()
        for ci in range(n_coeffs):
            resid_grid_land = _idw_residuals(
                residuals_all[:, ci],
                station_pts_3d,
                grid_pts_3d_land,
                k=k_idw,
                power=idw_power,
            )
            # Write back to full grid
            flat_surface = coeff_surfaces_final[ci].ravel()
            flat_surface[land_mask] += resid_grid_land
            flat_surface[~land_mask] = np.nan
            coeff_surfaces_final[ci] = flat_surface.reshape(
                self.grid.n_y, self.grid.n_x
            )

            if ci == 0 or ci == n_coeffs - 1:
                logger.info("    Coefficient %d/%d IDW done", ci + 1, n_coeffs)

        # --- Step 4: Evaluate at each DOY ---
        logger.info("  Step 4: Evaluating harmonics at 365 DOYs")
        clim = _evaluate_harmonics(coeff_surfaces_final, n_harmonics)

        # Write to zarr
        self._write_array(group, "tmax_clim", clim, overwrite=overwrite)

        # --- Step 5: Validation stats ---
        logger.info("  Step 5: Validation statistics")
        val_mask = good_fit & ~is_train
        n_val = val_mask.sum()

        val_stats = {}
        if n_val > 0:
            val_doy_means = doy_means[:, val_mask]

            # Get grid predictions at val station locations
            val_easting = stn_easting[val_mask]
            val_northing = stn_northing[val_mask]

            # Filter to stations within grid bounds
            x_min, y_min, x_max, y_max = self.grid.bounds
            in_bounds = (
                (val_easting >= x_min)
                & (val_easting <= x_max)
                & (val_northing >= y_min)
                & (val_northing <= y_max)
            )
            logger.info(
                "    %d / %d val stations within grid bounds", in_bounds.sum(), n_val
            )

            if in_bounds.sum() > 0:
                val_doy_means = val_doy_means[:, in_bounds]
                val_rows, val_cols = self.grid.xy_to_rowcol(
                    val_easting[in_bounds], val_northing[in_bounds]
                )
                n_val_in = int(in_bounds.sum())

                errors = []
                for s in range(n_val_in):
                    obs = val_doy_means[:, s]
                    pred = clim[:, val_rows[s], val_cols[s]]
                    valid = ~np.isnan(obs) & ~np.isnan(pred)
                    if valid.sum() > 0:
                        errors.append((obs[valid] - pred[valid]))

                if errors:
                    all_errors = np.concatenate(errors)
                    rmse = float(np.sqrt(np.mean(all_errors**2)))
                    bias = float(np.mean(all_errors))
                    val_stats["val_rmse_degC"] = round(rmse, 3)
                    val_stats["val_bias_degC"] = round(bias, 3)
                    val_stats["n_val_stations"] = n_val_in
                    logger.info(
                        "    Val RMSE: %.2f°C, Bias: %.2f°C (%d stations)",
                        rmse,
                        bias,
                        n_val_in,
                    )

        # Metadata
        store.attrs["tmax_clim_units"] = "degC"
        store.attrs["tmax_clim_n_stations"] = int(n_good)
        store.attrs["tmax_clim_n_harmonics"] = n_harmonics
        store.attrs["tmax_clim_min_years"] = min_years
        store.attrs["tmax_clim_min_r2"] = MIN_R2
        store.attrs["tmax_clim_k_idw"] = k_idw
        store.attrs["tmax_clim_elev_scale"] = elev_scale
        for k, v in val_stats.items():
            store.attrs[f"tmax_clim_{k}"] = v

        logger.info("tmax_clim layer complete")

    def validate(self) -> Dict[str, bool]:
        """Validate tmax_clim layer with physical range checks."""
        checks = super().validate()

        if not all(checks.values()):
            return checks

        try:
            store = zarr.open(str(self.store_path), mode="r")
            grp = store[self.name]

            clim = grp["tmax_clim"][:]
            valid = clim[~np.isnan(clim)]

            if len(valid) == 0:
                checks["has_data"] = False
                return checks
            checks["has_data"] = True

            # Physical range: PNW temps should be in [-40, 55] degC
            checks["temp_min"] = float(valid.min()) >= -60.0
            checks["temp_max"] = float(valid.max()) <= 60.0

            # Seasonal pattern: summer DOYs should be warmer than winter
            summer = clim[150:220, :, :]
            winter = clim[330:365, :, :]
            has_summer = np.any(~np.isnan(summer))
            has_winter = np.any(~np.isnan(winter))
            if has_summer and has_winter:
                checks["seasonal_pattern"] = float(np.nanmean(summer)) > float(
                    np.nanmean(winter)
                )
            else:
                checks["seasonal_pattern"] = True

            # Coverage: at least 50% non-NaN
            nan_frac = np.isnan(clim).mean()
            checks["coverage"] = nan_frac < 0.5

        except Exception as e:
            logger.error("Validation error: %s", e)
            checks["validation_error"] = False

        return checks
