# MADIS Observation Data: Acquisition, Permissions, and Quality Control

## Data Source

We obtain hourly mesonet observations from the **Meteorological Assimilation Data Ingest
System (MADIS)**, operated by NOAA/ESRL. Files are archived as gzip-compressed netCDF at
hourly resolution (`YYYYMMDD_HHMM.gz`), each containing all reporting stations for that hour
across CONUS and adjacent areas.

**Archive endpoint:**

```
https://madis-data.ncep.noaa.gov/madisResearch/data/archive/{YYYY}/{MM}/{DD}/LDAD/mesonet/netCDF/
```

### Elevated Access (LDAD)

We hold a MADIS Research account, which grants access to the **LDAD (Local Data Acquisition
and Dissemination)** mesonet tier. This is a broader collection than the publicly available
MADIS mesonet feed: LDAD includes observations from private and cooperative networks whose
data-sharing agreements restrict redistribution but permit research use. The practical
difference is roughly 2-3x more stations in the western US, which matters for spatial
coverage in areas like the interior Pacific Northwest where public networks are sparse.

Authentication is via username/password over HTTPS. Downloads are parallelized with `aria2c`
(16 connections per file, 5 files concurrent) or `wget` as a fallback, managed by
`extract/met_data/obs/madis_download.py`.

### Archive Coverage

- **Period of record:** 2001-07-01 through present
- **Update cadence:** Hourly, typically available within 1-2 hours of observation time
- **Volume:** ~210,000+ hourly files as of early 2026; individual files range from ~300 KB
  (2001) to ~35 MB (2025) as the network has grown

---

## Quality Control Pipeline

Our QC strategy combines two philosophically different approaches: an operational,
spatially-aware system adapted from NOAA's Real-Time Mesoscale Analysis (RTMA), and a
statistical, station-centric system drawn from the agricultural weather community. These are
complementary by design.

### Phase 1: Extract-Time Filtering

Applied during initial extraction from raw netCDF to daily parquet
(`process/obs/extract_madis_daily.py`):

1. **DD flag filter** -- MADIS assigns each observation a Data Descriptor flag summarizing
   its upstream QC status. We accept only `V` (Verified), `S` (Screened), `C` (Coarse pass),
   and `G` (Subjectively good). Observations flagged `X` (failed), `Q` (questioned), `B`
   (subjectively bad), or `Z` (no QC applied) are rejected.

2. **QCR bitmask filter** -- MADIS runs a three-level internal QC chain (validity bounds,
   temporal consistency, spatial consistency / buddy checks). The QCR field is a bitmask
   indicating which checks an observation *failed*. We reject on bits 1 (master failure), 2
   (validity bounds), 5 (temporal consistency), 6 (statistical spatial consistency), and 7
   (spatial buddy check). We intentionally skip bit 4 (internal consistency) to avoid
   over-filtering co-reported variables.

3. **Physical bounds** -- Tighter than MADIS defaults, informed by agronomic plausibility:
   temperature -50 to 60 C, dewpoint -68 to 32 C, relative humidity 2-110%, wind speed
   0-35 m/s (vs MADIS 0-129), wind direction 0-360.

### Phase 2: GSI Reject-List Filtering

Applied when building the training patch index (`prep/build_rtma_patch_index.py`):

The RTMA's data assimilation engine is the **Gridpoint Statistical Interpolation (GSI)**
system ([NOAA-EMC/GSI](https://github.com/NOAA-EMC/GSI)). NOAA maintains operational
reject lists that flag stations whose observations are systematically unreliable for a given
variable and time-of-day. These lists are published in the
[NOAA-EMC/GSI-fix](https://github.com/NOAA-EMC/GSI-fix) repository and are the same files
used by the operational RTMA.

We apply the following reject lists:

| List | Stations | Scope |
|------|----------|-------|
| `rtma_t_rejectlist` | ~4,400 | Temperature, all hours |
| `rtma_t_day_rejectlist` | ~650 | Temperature, daytime |
| `rtma_t_night_rejectlist` | ~120 | Temperature, nighttime |
| `rtma_q_rejectlist` | ~6,100 | Humidity, all hours |
| `rtma_q_day_rejectlist` | ~130 | Humidity, daytime |
| `rtma_w_rejectlist` | ~2,300 | Wind, all hours |

These lists originate primarily from Weather Forecast Office (WFO) local QC (~92%) with
contributions from global and dynamic QC processes. In our PNW domain, roughly 14% of
humidity-reporting stations and 12% of temperature-reporting stations appear on these lists.

### Phase 3: Statistical / Station-Centric QC

For long-term station characterization and drift detection, we adapt methods from
**agweather-qaqc** ([WSWUP/agweather-qaqc](https://github.com/WSWUP/agweather-qaqc)),
implemented in `utils/qaqc_calc.py`:

- **Modified Z-score outlier detection** -- Per-station, per-month robust outlier flagging
  (threshold 3.5 sigma). Catches garbage values that pass DD/QCR flags.
- **RH sensor drift correction** -- Yearly 99th-percentile analysis; assumes RH_max should
  approach 100% multiple times per year and derives a multiplicative correction factor.
- **Dewpoint-temperature consistency** -- Flags days where T_dew > T_min beyond a 2 C
  tolerance (excluding fog/precipitation events).
- **Isolated observation removal** -- Strips orphan data points surrounded by NaN windows.
- **Station-level rejection rate monitoring** -- Blacklists stations exceeding 50% rejection
  over a trailing 90-day window.

---

## Philosophy: Why Two Systems

### RTMA / GSI: Spatial, Short-Term, Operational

The RTMA is a 2D variational analysis that assimilates surface observations into a
gridded first-guess field every hour. Its QC (implemented in GSI) is fundamentally
**spatial**: it compares each observation against its neighbors and against the model
background, rejecting those whose "innovation" (observation minus background) exceeds a
gross-error threshold. The reject lists are a hard prior layered on top, built from
operational experience.

This system is authoritative for **short-term, event-scale reliability**. If a station is
producing physically plausible but spatially inconsistent values right now -- a broken
anemometer reading calm during a windstorm, a thermometer with a new radiation bias --
the GSI innovation check and the reject lists will catch it. The RTMA is also widely
depended upon: its gridded analyses are a primary input to NWS forecasts, fire weather
assessments, and agricultural decision support.

We pull the operational reject lists directly from
[NOAA-EMC/GSI-fix](https://github.com/NOAA-EMC/GSI-fix) and the QC logic is documented in
the GSI source at [NOAA-EMC/GSI](https://github.com/NOAA-EMC/GSI), particularly
`src/gsi/sfcobsqc.f90` (reject/use list application) and `src/gsi/setupq.f90` (humidity
gross-error checks).

### agweather-qaqc: Statistical, Long-Term, Station-Centric

The agweather-qaqc package was developed for quality control of daily agricultural weather
station data, with a focus on detecting problems that manifest over weeks to months rather
than hours. Its checks are **temporal and distributional**: they look at a single station's
history and ask whether recent behavior is consistent with that station's long-term
statistics.

This catches failure modes that spatial checks miss entirely: slow sensor drift (e.g., an RH
sensor gradually reading 10% low over a year), seasonal calibration shifts, and data
recording artifacts. These are precisely the problems that matter for bias-correction
modeling, where we need not just "is this observation roughly correct today" but "is this
station's long-term relationship to the gridded field stable and trustworthy."

> Dunkerly, C., Huntington, J. L., McEvoy, D., Morway, A., & Allen, R. G. (2024).
> agweather-qaqc: An Interactive Python Package for Quality Assurance and Quality Control of
> Daily Agricultural Weather Data and Calculation of Reference Evapotranspiration. *Journal
> of Open Source Software*, 9(97), 6368.
> [doi:10.21105/joss.06368](https://doi.org/10.21105/joss.06368)

### Complementarity

The two systems are orthogonal by design:

| Dimension | RTMA / GSI | agweather-qaqc |
|-----------|-----------|----------------|
| Spatial scope | All stations simultaneously | One station at a time |
| Temporal scope | Single hour | Months to years |
| Failure mode | Spatially inconsistent obs | Instrument drift, distributional shift |
| Decision basis | Innovation vs. background field | Station's own history |
| Update frequency | Hourly (reject lists: periodic) | Retrospective / batch |

By layering both, we reject observations that are bad *right now* (GSI) and observations
from stations that are unreliable *in general* (agweather-qaqc), while keeping the maximum
number of good observations for training.
