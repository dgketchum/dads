# RTMA Humidity MVP — Progress Notes (2026-02-06)

## Where we are

We split MVP work from cube/spatial refactor:
- `cube` branch remains for Phase 0 cube/spatial work.
- Humidity MVP is now based on `main` in a clean worktree (`mvp/rtma-humidity`).

We also created a temporary Codex worktree under `/tmp/dads-mvp-codex` to develop MVP utilities that can be
cherry-picked into the real MVP worktree.

Worktree details: see `notes/WORKTREE_SETUP.md`.

## RTMA archive verification

We verified the existing RTMA daily GeoTIFF archive at:
- `/data/ssd1/rtma/tif/RTMA_YYYYMMDD.tif`

`gdalinfo` shows these are daily COGs in EPSG:4326 with 10 bands:
`PRES, TMP, DPT, UGRD, VGRD, SPFH, WDIR, WIND, TCDC, ACPC01`
stored as `Int32` with band descriptions matching the Earth Engine export signature.

This strongly suggests the archive was produced by `extract/rs/earth_engine/rtma_export.py` (or a close variant):
- daily aggregation from hourly imagery
- integer casting and band scaling (implied; not stored as GeoTIFF scale/offset metadata)

### Important caveats discovered

1) **Scaling/units are implicit**
- `PRES` appears to be stored as hPa (EE export divides by 100 then casts to int).
- Many other bands are likely stored with a `*100` scale factor (so decode with `/100`).
- The GeoTIFFs do *not* advertise scale/offset metadata, so decoding must be explicit in the sampler.

2) **`SPFH` appears unusable (all zeros in spot checks)**
- Sampling multiple points and dates returned `0` for band `SPFH`.
- This means we should *not* rely on `SPFH` for deriving `ea` from specific humidity in this archive.

3) **Recommended `ea_rtma` baseline for MVP**
- Derive `ea_rtma` from dewpoint (`DPT`) using saturation vapor pressure:
  - `ea = es(Td)` (kPa), where `Td` is dewpoint in °C.
- This avoids the SPFH issue and keeps the baseline consistent and interpretable.

4) **Day boundary semantics**
- EE daily aggregations are effectively UTC-day. If station obs are local-day, we need to align
  semantics (either standardize to UTC-day for MVP, or do local-day mapping intentionally).

## MVP utilities added (Codex worktree)

These files exist in `/tmp/dads-mvp-codex` and are ready to commit/cherry-pick:
- `process/gridded/rtma_station_daily.py`
  - Converts hourly RTMA/URMA station extracts (monthly Parquets written by `extract/met_data/grid/rtma_extact.py`)
    into daily per-station Parquets with a stable schema and derived fields.
- `prep/build_station_day_table.py`
  - Builds a single (fid, day) Parquet by joining obs + RTMA/URMA daily tables.
  - Computes `delta_ea_rtma` / `delta_ea_urma` when baselines are present.
  - Includes an optional leakage-safe mask for humidity-proxy fields (keeps `ea_rtma/ea_urma`).

We also scripted the commit/cherry-pick workflow in `commits.sh` (in the main repo) for convenience.

## Next step (agreed)

Add a sampler that reads the existing daily RTMA COG archive (`/data/ssd1/rtma/tif/RTMA_YYYYMMDD.tif`) and
produces station-aligned daily tables quickly, computing `ea_rtma` from `DPT` instead of `SPFH`.
