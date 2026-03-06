# MVP -- RTMA Wind Bias-Correction

The Wind MVP is a station-graph GNN (with MLP baseline) for RTMA wind bias correction over the Pacific Northwest. Given an RTMA wind estimate, terrain context, and neighboring station information, the model predicts a vector correction $(\hat{\delta}_{par}, \hat{\delta}_{perp})$ that corrects both wind speed and direction biases.

## Relationship to Full DADS

The full [DADS pipeline](../../cube/architecture.md) decomposes each target variable into a station-derived climatological background and a daily anomaly predicted by a GNN from a star graph. The Wind MVP validates whether RTMA wind can be improved with terrain-aware spatial learning using a simpler k-NN station graph and an edge-gated attention mechanism.

| | Full DADS | Wind MVP |
|---|---|---|
| Spatial model | GNN (TransformerConv) on star graph | EdgeGatedAttention on k-NN station graph |
| Background | Station-derived harmonic $B(x, \text{doy})$ | RTMA itself (bias-correction framing) |
| Inputs | Station obs + cube features | RTMA weather + terrain + Sx + neighbor edges |
| Stores | zarr cube, stations, graph | Parquet + precomputed `.pt` graphs |
| Target | Anomaly $A$ (delta-mode) | $(\delta_{w,par}, \delta_{w,perp})$ -- vector correction |

## Relationship to Humidity MVP

Both MVPs correct RTMA biases using terrain and station observations, but differ in spatial approach and target structure:

| | Humidity MVP | Wind MVP |
|---|---|---|
| Spatial model | Gridded U-Net (64x64 patches) | Station-graph GNN (k-NN, k=16) |
| Target | Scalar $\Delta\!\log e_a$ (1D) | Vector $(\delta_{par}, \delta_{perp})$ (2D) |
| Supervision | Center-pixel (one station per patch) | Per-node (all stations per day) |
| Loss | Huber + TV regularization | Huber + calm-wind weighting |
| Terrain features | Gridded elevation/slope/TPI | Point-sampled + Winstral Sx |
| Additional covariates | r.sun, Landsat composite | Flow-terrain interactions |
| Station inventory | 9,168 PNW stations (shared) | 9,168 PNW stations (shared) |
| Temporal split | Same pattern (train / val by year) | Same pattern (train / val by year) |

## Pipeline at a Glance

1. **Sx computation** -- compute Winstral Sx (upwind terrain shelter) at station locations from projected DEM (`prep/build_station_sx.py`)
2. **Station-day join** -- join MADIS wind obs + RTMA baselines + terrain + Sx, compute parallel/perpendicular residual targets (`prep/build_wind_station_day_table.py`)
3. **Graph precomputation** -- build per-day PyG graphs with raw features for all stations, save as `.pt` files (`prep/build_wind_graphs.py`)
4. **Training** -- train WindBiasGNN with edge-gated attention on precomputed graphs (`models/wind_bias/train_wind_gnn.py`)
5. **Inference** *(future)* -- apply trained model to full RTMA grid, treating each grid cell as a node

## Current Status

Steps 1--2 of the ablation ladder are complete. Step 3 (GNN + Sx + flow-terrain) is in progress; Step 4 (2-hop) is pending.

### Ablation Results

| Component | RTMA Baseline | MLP (Step 1) | GNN no-Sx (Step 2) |
|-----------|---------------|--------------|---------------------|
| Parallel MAE (m/s) | 1.371 | 0.686 | **0.656** |
| Perpendicular MAE (m/s) | 0.603 | 0.546 | **0.523** |
| Vector RMSE (m/s) | 2.179 | 1.440 | **1.393** |
| val_loss | — | 0.362 | **0.339** |

RTMA systematically overestimates wind speed by ~1.2 m/s (negative parallel delta = obs < RTMA). The perpendicular bias is smaller but non-trivial.

## Key Findings So Far

- **Parallel (speed) correction is strong.** The MLP halves along-wind MAE (1.37 to 0.69 m/s), learning the systematic RTMA speed overestimation from local features alone.
- **Neighbor attention helps both components.** The GNN (Step 2) improves parallel MAE by a further 4.4% (0.686 → 0.656) and perpendicular MAE by 4.2% (0.546 → 0.523) over the MLP, despite using the same 20 node features. The spatial context from edge-gated attention benefits speed correction as well as direction.
- **Vector RMSE down 36%.** The GNN reduces RTMA vector RMSE from 2.18 to 1.39 m/s -- a 3.3% improvement over the MLP's 1.44 m/s.
- **No overfitting.** Train and val losses remain close across both experiments, confirming the temporal split generalizes well.
- **Sx and flow-terrain features are the next test.** Step 3 adds 37 directional terrain features (Winstral Sx + flow-terrain interactions) to test whether explicit terrain exposure information further improves corrections.

## Chapters

- [System Architecture](architecture.md) -- pipeline, artifacts, model structure, config system
- [Data Provenance](data_provenance.md) -- sources, formats, feature roster, Sx computation
- [Algorithms](algorithms.md) -- target variable, loss, normalization, metrics, training strategy
