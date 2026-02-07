# RTMA Humidity MVP — Progress Notes (2026-02-06)

## Where we are

We split MVP work from cube/spatial refactor:
- `cube` branch remains for Phase 0 cube/spatial work.
- Humidity MVP is now based on `main` in a clean worktree (`mvp/rtma-humidity`) at `/home/dgketchum/code/dads-mvp`.

During agent/sandbox sessions, a temporary worktree may be used under `/tmp` to develop changes and then
cherry-pick them into `/home/dgketchum/code/dads-mvp`. This is non-canonical; assume the canonical starting point
is always `/home/dgketchum/code/dads`.

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

### Practical decoding assumptions (based on spot checks)

The values we sampled are consistent with:
- `PRES` stored as integer **hPa** (no `/100` decoding; convert to kPa with `/10` if needed).
- Most other fields stored as integer `value * 100` (decode with `/100`):
  - `TMP`, `DPT` → °C
  - `UGRD`, `VGRD`, `WIND` → m/s
  - `WDIR` → degrees
  - `ACPC01` → mm (daily sum; `value/100`)
  - `TCDC` likely percent or fraction-derived percent; confirm via distribution check before using.

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

These utilities were implemented and committed in the Codex worktree and should be cherry-picked into
`/home/dgketchum/code/dads-mvp`:

- Commit: `b22b518` (`MVP: RTMA daily station tables + station-day join`)
- Cherry-pick:
  - `cd /home/dgketchum/code/dads-mvp && git cherry-pick b22b518`

Files:
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

## Proposed modeling target: patch-based RTMA bias correction (PNW, 1 km)

Goal: learn a *spatially coherent* daily correction to RTMA humidity and wind using point MADIS observations, then
apply that correction back onto the 1 km grid.

### Labels (station-day)

Compute supervised targets as residuals against RTMA sampled at the station location (nearest or bilinear):

- Humidity (recommended): `delta_log_ea = log(ea_obs) - log(ea_rtma)`
  - `ea_rtma` is derived from RTMA `DPT` via saturation vapor pressure (`ea = es(Td)`), per the recommendation above.
  - Log residual keeps the correction multiplicative and stabilizes variance across dry/wet regimes.
- Wind (vector residual): `delta_u = u_obs - u_rtma`, `delta_v = v_obs - v_rtma`
  - Prefer u/v over direction residuals to avoid angle wrap and speed-direction coupling.

Notes:
- Only station obs are used to form `y`; all predictors come from RTMA + static grids, so inference is leakage-safe.
- Day semantics must be explicit (UTC-day vs local-day); misalignment here becomes irreducible label noise.

### Inputs (gridded covariates)

Per-day RTMA bands at 1 km (decoded from the COGs):
- `TMP`, `DPT`, `UGRD`, `VGRD`, `PRES`, `TCDC`, `ACPC01`
- Optional derived channels: `ea_rtma` (from `DPT`), `wind_speed` (from u/v)

Static 1 km grids aligned to the RTMA grid:
- elevation, slope, aspect (sin/cos), TPI (radius choices TBD)

Time encodings (broadcast as constant grids for the patch/day):
- `doy_sin`, `doy_cos`

### Optional covariates (Landsat + NOAA CDR): phased-in, grid-first

For a patch model that outputs *full-grid* corrections, every input channel should be available as a *gridded raster*
domain-wide at inference time. Station-only RS extracts are useful for station models, but are not directly usable as
patch channels unless we also export the same sources as aligned grids.

Recommended phasing:
- MVP stack: RTMA + terrain + DOY encodings (everything is dense and inference-safe).
- Extended stack:
  - Landsat: use annual/seasonal composites (not daily scenes), aggregated/resampled onto the 1 km RTMA grid.
  - NOAA CDR: use gridded products (often coarser), resampled onto the RTMA grid as large-scale regime covariates.
  - Include QA/missingness masks for any RS channels that can be absent or unreliable.

### Learning setup (point-supervised fully convolutional)

Model: small U-Net / FCN that maps an input patch `(C, H, W)` to correction patch `(K, H, W)` where:
- `K=1` for humidity-only (`delta_log_ea`)
- `K=3` for humidity + wind (`delta_log_ea`, `delta_u`, `delta_v`)

Training:
- Sample patches centered on stations for each day (e.g., 64x64 pixels as a starting point).
- Forward-pass produces correction over the whole patch.
- Supervise only at the station pixel (center) with robust loss (Huber); add a light smoothness penalty (TV/Laplacian)
  on the predicted correction field so learned bias remains spatially coherent.
- Reweight sampling by station density (e.g., per-tile balancing) so urban clusters do not dominate.

Inference:
- Run the model fully convolutionally over the PNW grid for a given day to produce correction grids.
- Apply corrections to RTMA:
  - `ea_corr = ea_rtma * exp(delta_log_ea)`
  - `u_corr = u_rtma + delta_u`, `v_corr = v_rtma + delta_v` (derive speed/dir as needed)

### MVP data artifacts we need (to support patch training)

1) Station-day residual table (fast tabular)
- Join `ea_obs` / wind obs with sampled RTMA baselines to compute `delta_*`.
- This is the minimal "are residuals sensible?" check and supports non-spatial baselines.

2) Patch dataset (train-ready)
- `X`: RTMA patch + terrain patch + time encodings
- `y`: station residuals at the center pixel (`delta_log_ea`, `delta_u`, `delta_v`)
- Metadata: `fid`, `day`, station lat/lon, pixel row/col, sampling method, QC flags, local/UTC day mapping
- Prefer on-the-fly COG reads for iteration; optionally add a cache layer (Zarr/sharded NPZ) once stable.

### Early decisions (keep explicit in code + metadata)

- Station day alignment: do we re-aggregate MADIS to UTC-day (matches COG), or map local-day obs to UTC-day?
- Patch size: start 64x64 (~64 km) and tune; ensure padding behavior is deterministic at domain edges.
- Sampling: nearest vs bilinear for point residuals (pick one and stick to it for training + evaluation).

## MVP progress update (2026-02-07)

### Environment (uv + pyproject)

- Repo now uses `uv` with `pyproject.toml` + `uv.lock` and pins Python `3.13` via `.python-version`.
- `requirements.txt` has been removed; dependencies are declared in `pyproject.toml`.

### Patch pipeline implemented

- New patch sampler for RTMA daily COGs:
  - `process/gridded/rtma_patch_sampler.py` (windowed patch read + band decode consistent with the COG scaling notes)
- New patch index builder:
  - `prep/build_rtma_patch_index.py` (builds `fid, day, lat, lon, delta_log_ea` rows)
- New PNW clipping helper (station inventory convenience):
  - `prep/clip_stations_bounds.py`
- New MVP U-Net components for patch training:
  - `models/rtma_bias/unet.py`
  - `models/rtma_bias/patch_dataset.py`
  - `models/rtma_bias/train_patch_unet.py`

### MADIS day alignment support (join step)

- `prep/build_station_day_table.py` now supports tz-aware obs indexes via `--obs-day-mode`:
  - `local` (drop tz, keep local day)
  - `utc_midpoint` (map each local day to the UTC day containing its local-noon midpoint)
  - `utc_start` (map local midnight to UTC day)
- For MVP against UTC-day RTMA COGs, `utc_midpoint` is the default recommended mode to reduce day-boundary mismatch.

### Sanity run completed (real data, small scope)

Artifacts created to validate the full loop on PNW stations:

- PNW station inventories (bounds clip + optional sample):
  - `artifacts/madis_pnw.csv` (~8.9k stations)
  - `artifacts/madis_pnw_500.csv` (500-station sample)
- RTMA daily per-station extraction (COG sampler path):
  - Output example: `/nas/dads/mvp/rtma_daily_pnw_500_jan1wk` (2025-01-01..2025-01-07)
- Station-day join (MADIS `ea` + RTMA `ea_rtma` baseline):
  - Output example: `/nas/dads/mvp/station_day_ea_pnw_500_jan1wk.parquet`
- Patch index (train rows):
  - Output example: `/nas/dads/mvp/patch_index_ea_pnw_500_jan1wk.parquet` (1288 rows)
- First model training run (humidity-only, CPU, 1 epoch):
  - Output example: `/nas/dads/mvp/unet_ea_run1/ckpt_epoch_000.pt`, `ckpt_final.pt`, `train_config.json`

Commands used (sanity run):

```bash
cd /home/dgketchum/code/dads-mvp

# Clip full MADIS inventory to PNW bounds, then take a 500-station sample.
uv run python prep/clip_stations_bounds.py \
  --in /nas/dads/met/stations/madis_02JULY2025_mgrs.csv \
  --out artifacts/madis_pnw.csv \
  --bounds -125.5 41.5 -110.0 49.5 \
  --id-col station_id --lat-col latitude --lon-col longitude

uv run python prep/clip_stations_bounds.py \
  --in /nas/dads/met/stations/madis_02JULY2025_mgrs.csv \
  --out artifacts/madis_pnw_500.csv \
  --bounds -125.5 41.5 -110.0 49.5 \
  --id-col station_id --lat-col latitude --lon-col longitude \
  --limit 500 --seed 42

# RTMA daily per-station extracts from COGs (7-day window for a fast check).
uv run python process/gridded/rtma_station_daily.py \
  --source tif \
  --stations artifacts/madis_pnw_500.csv \
  --tif-root /data/ssd1/rtma/tif \
  --out-daily-root /nas/dads/mvp/rtma_daily_pnw_500_jan1wk \
  --model rtma \
  --start-date 2025-01-01 --end-date 2025-01-07 \
  --station-id-col station_id --lat-col latitude --lon-col longitude \
  --station-chunk-size 200

# Join MADIS obs with RTMA baseline to build station-day residual table.
uv run python prep/build_station_day_table.py \
  --obs-dir /data/ssd2/madis/daily \
  --obs-col ea \
  --rtma-daily-dir /nas/dads/mvp/rtma_daily_pnw_500_jan1wk \
  --stations-csv artifacts/madis_pnw_500.csv --station-id-col station_id \
  --obs-day-mode utc_midpoint \
  --out-file /nas/dads/mvp/station_day_ea_pnw_500_jan1wk.parquet \
  --overwrite

# Build patch index for patch sampling/training.
uv run python prep/build_rtma_patch_index.py \
  --station-day /nas/dads/mvp/station_day_ea_pnw_500_jan1wk.parquet \
  --stations-csv artifacts/madis_pnw_500.csv \
  --station-id-col station_id --lat-col latitude --lon-col longitude \
  --require-tif-root /data/ssd1/rtma/tif \
  --out /nas/dads/mvp/patch_index_ea_pnw_500_jan1wk.parquet

# Train humidity-only U-Net on CPU (GPU wheel is incompatible with Quadro P2000 sm_61).
PYTHONPATH=. uv run python models/rtma_bias/train_patch_unet.py \
  --patch-index /nas/dads/mvp/patch_index_ea_pnw_500_jan1wk.parquet \
  --tif-root /data/ssd1/rtma/tif \
  --out-dir /nas/dads/mvp/unet_ea_run1 \
  --epochs 1 --batch-size 32 --patch-size 64 \
  --num-workers 0 --device cpu
```

### Known issues / constraints discovered

- Quadro P2000 (GPU 1, sm_61) is incompatible with torch ≥2.10; **use RTX A6000 (GPU 0, sm_86)** instead.
  `CUDA_VISIBLE_DEVICES=0 --device cuda:0` works.
- MADIS obs contain garbage outliers (values up to 10^303); filtering needed before production use.

## Scaled-up PNW 2024 training run (2026-02-07)

### GPU confirmed

- RTX A6000 (48 GB VRAM, sm_86) is available as GPU 0 and fully supported by torch 2.10+cu128.
- Quadro P2000 (GPU 1) is display-only; ignore the torch compatibility warning.

### Domain and data

- PNW bounds updated to cube-aligned: `(-125.0, 42.0, -104.0, 49.0)` (OR/WA/ID/MT).
- Re-clipped MADIS inventory: `artifacts/madis_pnw.csv` → 9,168 stations.
- RTMA COG archive: all 365 days of 2024 available; only 8 days in Jan 2025.

### Pipeline execution (full 2024)

```bash
cd /home/dgketchum/code/dads-mvp

# 1. Re-clip stations to cube-aligned PNW bounds.
uv run python prep/clip_stations_bounds.py \
  --in /nas/dads/met/stations/madis_02JULY2025_mgrs.csv \
  --out artifacts/madis_pnw.csv \
  --bounds -125.0 42.0 -104.0 49.0 \
  --id-col station_id --lat-col latitude --lon-col longitude

# 2. RTMA daily per-station extracts from COGs (9168 stations, full 2024).
PYTHONPATH=. uv run python process/gridded/rtma_station_daily.py \
  --source tif \
  --stations artifacts/madis_pnw.csv \
  --tif-root /data/ssd1/rtma/tif \
  --out-daily-root /nas/dads/mvp/rtma_daily_pnw_2024 \
  --model rtma \
  --start-date 2024-01-01 --end-date 2024-12-31 \
  --station-id-col station_id --lat-col latitude --lon-col longitude \
  --station-chunk-size 2000 --overwrite

# 3. Join MADIS obs with RTMA baseline.
PYTHONPATH=. uv run python prep/build_station_day_table.py \
  --obs-dir /data/ssd2/madis/daily \
  --obs-col ea \
  --rtma-daily-dir /nas/dads/mvp/rtma_daily_pnw_2024 \
  --stations-csv artifacts/madis_pnw.csv --station-id-col station_id \
  --obs-day-mode utc_midpoint \
  --out-file /nas/dads/mvp/station_day_ea_pnw_2024.parquet \
  --overwrite

# 4. Build patch index.
PYTHONPATH=. uv run python prep/build_rtma_patch_index.py \
  --station-day /nas/dads/mvp/station_day_ea_pnw_2024.parquet \
  --stations-csv artifacts/madis_pnw.csv \
  --station-id-col station_id --lat-col latitude --lon-col longitude \
  --require-tif-root /data/ssd1/rtma/tif \
  --out /nas/dads/mvp/patch_index_ea_pnw_2024.parquet

# 5. Train U-Net on A6000 with preloaded COGs.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python models/rtma_bias/train_patch_unet.py \
  --patch-index /nas/dads/mvp/patch_index_ea_pnw_2024.parquet \
  --tif-root /data/ssd1/rtma/tif \
  --out-dir /nas/dads/mvp/unet_ea_pnw_2024 \
  --epochs 10 --batch-size 64 --patch-size 64 \
  --num-workers 2 --device cuda:0 --preload
```

### Data summary

- Station-day join: 13.3M total rows; 849K with both obs and RTMA for 2024.
- Patch index: 848,444 rows, 3,513 unique stations, 297 unique days (Jan–Oct 2024).
- `delta_log_ea` distribution: mean −0.003, std 0.29, median 0.011.

### I/O optimization: COG preloading

**Problem:** LZW-compressed COGs are extremely slow for random windowed reads:
- 64×64 patch read: ~170ms (random location), ~53ms (cached location).
- 848K reads/epoch × 170ms ≈ 40 hours/epoch — impossible.

**Solution:** Pre-load all unique COGs into RAM as decoded float32 arrays.
- 297 COGs × ~217 MB decoded ≈ 64 GB (machine has 540 GB RAM).
- Preload takes ~25 min (4.2s per COG × 297).
- `__getitem__` becomes a pure numpy slice (~0.2ms), 850× faster.
- Fork-based DataLoader workers share memory via copy-on-write.

Code changes:
- `models/rtma_bias/patch_dataset.py`: added `PatchDatasetConfig.preload`, `_preload_cogs()`,
  fast `_getitem_preloaded()` path, fallback on-disk path with file-handle caching.
- `process/gridded/rtma_patch_sampler.py`: `sample_rtma_patch()` accepts optional pre-opened `src`.
- `models/rtma_bias/train_patch_unet.py`: added `--preload/--no-preload` flag, `pin_memory`,
  per-epoch progress logging.

### Training results (10 epochs, A6000)

| Epoch | Loss   | Time (s) |
|-------|--------|----------|
| 0     | 0.0260 | 418      |
| 1     | 0.0259 | 394      |
| 2     | 0.0258 | 410      |
| 3     | 0.0257 | 385      |
| 4     | 0.0255 | 327      |
| 5     | 0.0254 | 323      |
| 6     | 0.0253 | 300      |
| 7     | 0.0245 | 308      |
| 8     | 0.0229 | 324      |
| 9     | 0.0201 | 323      |

- Loss still dropping at epoch 9 — more epochs and/or larger model should help.
- Total wall time: ~25 min preload + ~57 min training ≈ 82 min.
- Output: `/nas/dads/mvp/unet_ea_pnw_2024/`

### Next steps

- Run more epochs (loss still converging).
- Add terrain covariates (elevation, slope, aspect, TPI) to the patch input stack.
- Evaluate: residual distributions, spatial coherence of predicted corrections, hold-out validation.
- Filter MADIS outliers before training (garbage obs values up to 10^303).
- Extend to wind (u/v residuals, K=3 output channels).
