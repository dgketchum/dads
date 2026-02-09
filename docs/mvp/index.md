# MVP -- RTMA Humidity Bias-Correction

The MVP is a lightweight, gridded-to-gridded bias-correction system for RTMA actual vapor pressure ($e_a$) over the Pacific Northwest. It uses a patch U-Net trained on MADIS station observations to learn a spatially coherent daily correction field that is applied multiplicatively to the RTMA baseline.

## Relationship to Full DADS

The full [DADS pipeline](../architecture.md) decomposes each target variable into a station-derived climatological background and a daily anomaly predicted by a GNN from a neighbor graph. That system requires a complete data cube (stations.zarr, cube.zarr, graph.zarr), an autoencoder for station embeddings, and a graph neural network for spatial interpolation.

The MVP takes a deliberately simpler path:

| | Full DADS | MVP |
|---|---|---|
| Spatial model | GNN on station graph | U-Net on gridded patches |
| Background | Station-derived harmonic $B(x, \text{doy})$ | RTMA itself (bias-correction, not direct prediction) |
| Inputs | Station obs + cube features | RTMA COGs + terrain + rsun + Landsat |
| Stores | zarr cube, stations, graph | Parquet index + GeoTIFFs |
| Target | Anomaly $A$ (delta-mode) | $\Delta\!\log e_a$ (multiplicative correction) |

The MVP validates the core hypothesis -- that RTMA humidity can be improved with terrain-aware spatial learning -- before committing to the full GNN infrastructure.

## Pipeline at a Glance

1. **Station extraction** -- sample RTMA COGs at station locations, aggregate to daily (`process/gridded/rtma_station_daily.py`)
2. **Station-day join** -- join MADIS $e_a$ observations with RTMA baseline, compute residuals (`prep/build_station_day_table.py`)
3. **Patch index** -- build training rows with $\Delta\!\log e_a$ targets and outlier filtering (`prep/build_rtma_patch_index.py`)
4. **U-Net training** -- center-pixel supervision with Huber + TV loss, COGs preloaded into RAM (`models/rtma_bias/train_patch_unet.py`)

## Current Status

The best model (Run 5: 7 years of data, base-48 U-Net, 24 epochs) reduces RTMA $e_a$ MAE by **28%** (0.076 → 0.055 kPa) with R² = 0.60 on held-out stations.

**Key findings from ablation studies:**

- **Terrain and land cover dominate the correction signal.** Elevation, slope, TPI, solar geometry, and Landsat surface reflectance together account for the bulk of the model's skill. Adding these covariates in Run 2 lifted R² from 0.02 to 0.61.
- **Humidity inputs help, but less than expected.** Dropping dewpoint and $e_a$ from the input channels (Run 6) costs only ~3% in MAE and ~2.4 percentage points of RTMA improvement (24.4% vs 28.3%). The model reconstructs most of the bias-correction signal from weather, terrain, and vegetation context alone.
- **Bias-correction framing is clearly superior to direct prediction.** Predicting $\log(e_{a,\text{obs}})$ without humidity inputs (Run 4) merely matches RTMA; predicting $\Delta\!\log e_a$ with the same inputs (Run 6) beats it by 25%.
- **More data improves robustness.** Expanding from 1 year (2024) to 7 years (2018–2024) in Run 5 yielded similar peak accuracy but better generalisation across years and seasons.

## Chapters

- [System Architecture](architecture.md) -- pipeline, artifacts, model structure, config system
- [Data Provenance](data_provenance.md) -- sources, formats, decoding, feature roster
- [Algorithms](algorithms.md) -- target variable, loss, normalization, metrics, training strategy
