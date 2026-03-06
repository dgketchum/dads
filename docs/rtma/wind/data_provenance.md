# Data Provenance

Every input to the Wind MVP traces to a direct observation, a direct measurement, or a deterministic physical quantity. RTMA is the *correction target* -- the model learns to fix RTMA's wind biases using station truth, terrain, and neighbor context.

---

## RTMA COG Archive

The Wind MVP uses the same RTMA COG archive as the [Humidity MVP](../mvp/data_provenance.md#rtma-cog-archive). Key properties are summarized here; see the humidity docs for full details on format, coverage, and day boundary alignment.

| Property | Value |
|----------|-------|
| Path | `/data/ssd1/rtma/tif/RTMA_YYYYMMDD.tif` |
| Coverage | 2012-01-01 to 2025-01-08 (4,688 files) |
| Format | LZW-compressed COG, EPSG:4326, Int32, 256x256 tiled |

### Wind-Relevant Bands

| Band | Field | Decode | Units | Role in Wind MVP |
|------|-------|--------|-------|-----------------|
| 4 | UGRD | / 100 | m/s | RTMA u-component (primary input + target construction) |
| 5 | VGRD | / 100 | m/s | RTMA v-component (primary input + target construction) |
| 8 | WIND | / 100 | m/s | RTMA wind speed (loss weighting) |
| 2 | TMP | / 100 | C | Weather context |
| 3 | DPT | / 100 | C | Stability proxy (tmp - dpt) |
| 1 | PRES | / 10 | kPa | Weather context |
| 9 | TCDC | / 100 | % | Weather context |
| 10 | ACPC01 | / 100 | mm/day | Weather context |

All 10 bands are sampled at station locations via `process/gridded/rtma_station_daily.py`, then `ea` (vapor pressure from Magnus formula) and `tmp_dpt_diff` are derived.

---

## MADIS Station Observations

Same source as the [Humidity MVP](../mvp/data_provenance.md#madis-station-observations), but different variables are used:

| Property | Value |
|----------|-------|
| Source | NOAA Meteorological Assimilation Data Ingest System |
| Path | `/data/ssd2/madis/daily/` (one Parquet per station) |
| Variables used | `u`, `v` (wind components, m/s); `wind` (speed); `wind_dir` (direction) |
| Coverage | Variable by station; typically 2010--2025 |

Observations are joined to RTMA-sampled values to compute parallel and perpendicular residual targets ($\delta_{w,par}$, $\delta_{w,perp}$).

!!! note "QC filtering"
    The station-day table builder applies:

    - Drop NaN wind observations
    - Wind speed range: 0--50 m/s
    - Component residual range: $|\delta_u| < 20$ m/s and $|\delta_v| < 20$ m/s

---

## Station Inventory

Same as the [Humidity MVP](../mvp/data_provenance.md#station-inventory):

| Property | Value |
|----------|-------|
| Source | MADIS 02JULY2025 inventory |
| Path | `/nas/dads/met/stations/madis_02JULY2025_mgrs.csv` |
| PNW clip | `artifacts/madis_pnw.csv` (9,168 stations) |
| Bounds | $-125.0$ to $-104.0$ lon, $42.0$ to $49.0$ lat |

---

## Terrain

Same 6-band GeoTIFF as the [Humidity MVP](../mvp/data_provenance.md#terrain), sampled at station points rather than as gridded patches:

| Channel | Method | Notes |
|---------|--------|-------|
| `elevation` | Direct from DEM | Meters, average resampling |
| `slope` | Horn's 3x3 | Degrees |
| `aspect_sin` | $\sin(\text{aspect})$ | Circular encoding |
| `aspect_cos` | $\cos(\text{aspect})$ | Circular encoding |
| `tpi_4` | TPI, 4-pixel radius | Topographic Position Index |
| `tpi_10` | TPI, 10-pixel radius | Broader-scale ridge/valley signal |

---

## Winstral Sx

!!! note "New in Wind MVP"
    Sx is unique to the Wind MVP -- not used in the Humidity MVP.

Winstral Sx measures the maximum upwind terrain slope angle along a ray from each station, quantifying shelter or exposure from specific directions.

| Property | Value |
|----------|-------|
| Source | Computed from projected DEM (EPSG:5071, 250 m cells) |
| Build script | `prep/build_station_sx.py` |
| Output | `artifacts/sx_pnw.parquet` |
| Azimuth bins | 16 (0, 22.5, 45, ..., 337.5 degrees) |
| Search distances | 2 km (local) and 10 km (outlying) |
| Columns | 32 directional values + 2 derived = 34 total |

### Computation

For each station and each azimuth bin, the algorithm:

1. Casts rays from the station location in the DEM's projected CRS
2. For each ray, walks outward from the station at pixel resolution
3. Records the maximum terrain slope angle: $\text{Sx} = \max_d \frac{z_d - z_\text{station}}{d}$
4. Uses a 30-degree sector (3 sub-rays per azimuth) for robustness

### Physical Interpretation

- **Positive Sx** = station is sheltered (upwind terrain obstacle rises above station)
- **Negative Sx** = station is exposed (terrain slopes away in that direction)
- Values typically range from $-0.3$ to $+0.5$ (slope angles in radians)

### Derived Features

| Feature | Computation | Physical meaning |
|---------|------------|-----------------|
| `terrain_openness` | Mean of all 32 Sx values | Overall shelter vs exposure |
| `terrain_directionality` | Std of 10 km Sx values | How directionally variable the terrain exposure is |

---

## Flow-Terrain Interactions

Dynamic features computed at table-build time from the combination of RTMA wind direction and station terrain:

| Feature | Computation | Physical meaning |
|---------|------------|-----------------|
| `flow_upslope` | $\hat{e}_{par} \cdot \hat{e}_{upslope}$ | Alignment of RTMA wind with local terrain slope |
| `flow_cross` | $\hat{e}_{par} \cdot \hat{e}_{cross}$ | Cross-slope wind component |
| `wind_aligned_sx` | Sx interpolated to RTMA wind-from direction (10 km) | Terrain shelter/exposure from the actual upwind direction |

These features explicitly model how terrain deflects and shelters wind, varying daily with wind direction.

---

## Complete Feature Roster

| Group | Features | Count |
|-------|----------|-------|
| RTMA weather | ugrd, vgrd, wind, tmp, dpt, pres, tcdc, prcp, ea, tmp_dpt_diff | 10 |
| Terrain | elevation, slope, aspect_sin, aspect_cos, tpi_4, tpi_10 | 6 |
| Sx directional | sx_{000..337}_2k, sx_{000..337}_10k | 32 |
| Sx derived | terrain_openness, terrain_directionality | 2 |
| Flow-terrain | flow_upslope, flow_cross, wind_aligned_sx | 3 |
| Temporal | doy_sin, doy_cos | 2 |
| Location | latitude, longitude | 2 |
| **Total** | | **57** |

!!! warning "Feature flag control"
    Not all features are used in every experiment. The `use_sx` flag controls Sx + derived (34 features), and `use_flow_terrain` controls flow-terrain interactions (3 features). The MLP baseline uses only 20 features (RTMA + terrain + temporal + location).

---

## Edge Features

Edge attributes encode the spatial relationship between neighboring stations:

| Feature | Type | Description |
|---------|------|-------------|
| `distance_norm` | Static | Haversine distance (z-score normalized) |
| `bearing_sin` | Static | $\sin(\text{bearing}_{i \to j})$ |
| `bearing_cos` | Static | $\cos(\text{bearing}_{i \to j})$ |
| `delta_elevation` | Static | $z_j - z_i$ (z-score normalized) |
| `delta_tpi` | Static | TPI difference (reserved, currently 0) |
| `upwind_cos` | Dynamic | $\cos(\theta_\text{wind} - \theta_\text{bearing})$ -- alignment of wind with edge direction |
| `upwind_sin` | Dynamic | $\sin(\theta_\text{wind} - \theta_\text{bearing})$ -- cross-wind component of edge |

Static attributes are computed once from the station inventory. Dynamic attributes (upwind_cos, upwind_sin) are recomputed daily from RTMA wind direction, encoding whether a neighbor is upwind or downwind of the target station.
