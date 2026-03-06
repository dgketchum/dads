# RTMA Bias Correction

*Terrain-aware correction of RTMA/URMA analysis fields using station observations.*

The RTMA bias-correction project trains models to predict spatially coherent daily
correction fields for RTMA meteorological variables. Each target variable has its
own model architecture chosen to match the spatial structure of the bias.

## Target Variables

| Variable | Model | Status |
|----------|-------|--------|
| [Humidity (ea)](humidity/index.md) | Patch U-Net | Validated (28% MAE reduction) |
| [Wind](wind/index.md) | Station-graph GNN | Validated (36% vector RMSE reduction) |
| [Shortwave Radiation](shortwave/index.md) | TBD | Stub |
| [Temperature](temperature/index.md) | TBD | Stub |
