# Data Provenance

Every input to the MVP traces to a direct observation, a direct measurement, or a deterministic physical quantity. RTMA is the *correction target*, not a covariate -- the model learns to fix RTMA's biases using station truth and terrain/RS context.

---

## RTMA COG Archive

| Property | Value |
|----------|-------|
| Source | NOAA Real-Time Mesoscale Analysis, exported via Google Earth Engine |
| Path | `/data/ssd1/rtma/tif/RTMA_YYYYMMDD.tif` |
| Format | LZW-compressed Cloud-Optimized GeoTIFF, 256×256 tiled |
| CRS | EPSG:4326 (WGS84 geographic) |
| Size | 3608 × 1876 pixels per band |
| Data type | Int32 (scaled integer encoding) |
| Coverage | 2012-01-01 to 2025-01-08 (4,688 files); 2024 fully complete |

### Band Decoding

| Band | Field | Decode | Units | Notes |
|------|-------|--------|-------|-------|
| 1 | PRES | ÷ 10 | kPa | Stored as integer hPa |
| 2 | TMP | ÷ 100 | °C | |
| 3 | DPT | ÷ 100 | °C | Primary humidity input |
| 4 | UGRD | ÷ 100 | m/s | |
| 5 | VGRD | ÷ 100 | m/s | |
| 6 | SPFH | -- | kg/kg | **All zeros in archive** -- unusable |
| 7 | WDIR | ÷ 100 | degrees | |
| 8 | WIND | ÷ 100 | m/s | |
| 9 | TCDC | ÷ 100 | % | |
| 10 | ACPC01 | ÷ 100 | mm/day | Convective precip accumulation |

!!! warning "SPFH caveat"
    Band 6 (SPFH, specific humidity) is all zeros throughout the archive. The MVP derives $e_a$ from dewpoint (DPT) via the Magnus formula instead:

    $$e_a = 0.6108 \cdot \exp\!\left(\frac{17.27 \cdot T_d}{T_d + 237.3}\right)$$

    where $T_d$ is dewpoint temperature in °C.

### Day Boundary

Each COG represents a UTC-day aggregate from the Earth Engine export. MADIS observations are aligned to UTC-day via `utc_midpoint` mode: each station's local day is mapped to the UTC day that contains its local-noon midpoint.

---

## MADIS Station Observations

| Property | Value |
|----------|-------|
| Source | NOAA Meteorological Assimilation Data Ingest System |
| Path | `/data/ssd2/madis/daily/` (one Parquet per station) |
| Variable used | `ea` -- actual vapor pressure (kPa) |
| Coverage | Variable by station; typically 2010--2025 |
| Networks | ~95K stations across CONUS (sub-daily aggregated to daily) |

The MVP uses only the `ea` column. Observations are joined to RTMA-sampled values at the station pixel to compute $\Delta\!\log e_a$.

!!! note "Outlier filtering"
    MADIS archives contain garbage values (up to $10^{303}$). The patch index builder applies:

    - Minimum $e_a$: $10^{-4}$ kPa (physical floor)
    - Maximum $|\Delta\!\log e_a|$: 3.0 (removes extreme residuals)

---

## Station Inventory

| Property | Value |
|----------|-------|
| Source | MADIS 02JULY2025 inventory |
| Path | `/nas/dads/met/stations/madis_02JULY2025_mgrs.csv` |
| PNW clip | `artifacts/madis_pnw.csv` (9,168 stations) |
| Bounds | $-125.0°$ to $-104.0°$ lon, $42.0°$ to $49.0°$ lat |
| Fields used | `station_id`, `latitude`, `longitude` |

Bounds are cube-aligned to the DADS production grid (OR/WA/ID/MT).

---

## Auxiliary Covariates

### Terrain

| Property | Value |
|----------|-------|
| Source | DEM-derived (30 m, resampled to RTMA grid) |
| Format | 6-band GeoTIFF aligned to RTMA extent |
| Build | `prep/build_terrain_tif.py` |

| Channel | Method | Notes |
|---------|--------|-------|
| `elevation` | Direct from DEM | Meters, average resampling |
| `slope` | Horn's 3×3 | Degrees |
| `aspect_sin` | $\sin(\text{aspect})$ | Circular encoding |
| `aspect_cos` | $\cos(\text{aspect})$ | Circular encoding |
| `tpi_4` | TPI, 4-pixel radius | Topographic Position Index |
| `tpi_10` | TPI, 10-pixel radius | Broader-scale ridge/valley signal |

### Clear-Sky Solar Irradiance

| Property | Value |
|----------|-------|
| Source | GRASS GIS `r.sun` on DEM |
| Format | 365-band GeoTIFF (one band per DOY) |
| Units | Wh/m²/day |
| Channel | `rsun` -- single band selected by DOY at runtime |

### Landsat Climatological Composite

| Property | Value |
|----------|-------|
| Source | Landsat surface reflectance + thermal (Earth Engine composites) |
| Format | 35-band GeoTIFF (7 spectral bands × 5 seasonal periods) |
| Periods | P0: Jan--Mar, P1: Mar--May, P2: May--Jul, P3: Jul--Sep, P4: Sep--Dec |
| Period selected by date at runtime |

| Channel | Measurement | Descaling |
|---------|-------------|-----------|
| `ls_b2`--`ls_b7` | Surface reflectance | raw / 10000 → [0, 1] |
| `ls_b10` | Thermal brightness temperature | raw / 100 → K |

Gap-fill: per-pixel seasonal mean across all years. Missing winter composites (especially P0) are filled with this climatological mean.

---

## Complete Feature Roster

| Channel | Source | Resolution | Cadence | Count |
|---------|--------|-----------|---------|-------|
| tmp_c | RTMA (TMP) | ~2.5 km (RTMA native) | Daily | 1 |
| dpt_c | RTMA (DPT) | ~2.5 km | Daily | 1 |
| ugrd, vgrd | RTMA (UGRD, VGRD) | ~2.5 km | Daily | 2 |
| pres_kpa | RTMA (PRES) | ~2.5 km | Daily | 1 |
| tcdc_pct | RTMA (TCDC) | ~2.5 km | Daily | 1 |
| prcp_mm | RTMA (ACPC01) | ~2.5 km | Daily | 1 |
| ea_kpa | Derived from DPT | ~2.5 km | Daily | 1 |
| doy_sin, doy_cos | Computed | N/A | Daily | 2 |
| elevation, slope | DEM | 1 km | Static | 2 |
| aspect_sin, aspect_cos | DEM | 1 km | Static | 2 |
| tpi_4, tpi_10 | DEM | 1 km | Static | 2 |
| rsun | r.sun | 1 km | 365 DOY | 1 |
| ls_b2--ls_b7, ls_b10 | Landsat | 1 km | 5 periods/year | 7 |
| **Total** | | | | **24** |
