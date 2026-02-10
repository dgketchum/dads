# Algorithms

---

## Target Variable

The model predicts a parallel/perpendicular decomposition of the wind vector error relative to the RTMA wind direction.

### Unit Vectors

Given RTMA wind components $(u, v)$ at a station:

$$
\hat{e}_{par} = \frac{(u, v)}{\|(u, v)\| + \epsilon}, \qquad
\hat{e}_{perp} = \frac{(-v, u)}{\|(u, v)\| + \epsilon}
$$

where $\epsilon = 10^{-6}$ prevents division by zero in calm conditions.

### Residual Targets

$$
\delta_{par} = (\mathbf{w}_\text{obs} - \mathbf{w}_\text{rtma}) \cdot \hat{e}_{par}
$$

$$
\delta_{perp} = (\mathbf{w}_\text{obs} - \mathbf{w}_\text{rtma}) \cdot \hat{e}_{perp}
$$

- **$\delta_{par}$** captures speed bias: positive means RTMA underestimates speed, negative means overestimate
- **$\delta_{perp}$** captures directional bias: non-zero values indicate RTMA wind direction error

### Why Parallel/Perpendicular?

- **Decouples error sources.** Speed error and direction error have different physical causes and spatial patterns. Separate targets let the model (and metrics) isolate each.
- **Interpretable components.** $\delta_{par}$ directly maps to speed bias; $\delta_{perp}$ maps to direction error.
- **Component-specific analysis.** The MLP baseline showed 50% speed improvement but only 9.5% direction improvement -- this decomposition revealed exactly where spatial context is needed.

### Why Bias-Correction?

The same argument as the [Humidity MVP](../mvp/algorithms.md#why-bias-correction-not-direct-prediction): by providing the RTMA estimate as input, the model only needs to learn the *residual error structure*, not reconstruct the full wind field from scratch. This makes the task much easier and the corrections more stable.

---

## EdgeGatedAttention Mechanism

The core spatial aggregation layer (`models/wind_bias/gnn.py`):

### Message Computation

For each directed edge $(j \to i)$ with source node $j$, target node $i$, and edge features $e_{ji}$:

$$
\alpha_{ji} = \text{softmax}_j\Bigl(\text{MLP}_\text{att}\bigl([h_i;\, h_j;\, e_{ji}]\bigr)\Bigr)
$$

The attention MLP maps concatenated features to a scalar: $\mathbb{R}^{2H + E} \to \mathbb{R}^{H} \to \mathbb{R}^{1}$, where $H$ is hidden_dim and $E = 7$ is edge_dim.

### Aggregation

$$
c_i = \sum_{j \in \mathcal{N}(i)} \alpha_{ji} \cdot h_j
$$

### Context Merge

$$
h_i' = \text{MLP}_\text{merge}\bigl([h_i;\, c_i]\bigr)
$$

### Multi-Hop

For $n\_hops > 1$, the process repeats with updated $h$. Each hop uses its own `EdgeGatedAttention` and merge layer. The output head receives $[h_\text{local};\, h_\text{final}]$, where $h_\text{local}$ is the node encoding before any attention (skip connection).

---

## k-NN Graph Construction

The neighbor graph is built via `build_knn_map()` in `wind_dataset.py`:

| Parameter | Value |
|-----------|-------|
| Algorithm | `BallTree` on station lat/lon (haversine metric) |
| $k$ | 16 neighbors per station |
| Max radius | 150 km (edges beyond this distance are pruned) |

### Static Edge Attributes

Computed once from the station inventory:

- **distance_km**: haversine distance between stations
- **bearing_sin, bearing_cos**: circular encoding of the compass bearing from station $i$ to $j$
- **delta_elevation**: $z_j - z_i$ in meters

Distance and delta_elevation are z-score normalized using statistics computed over all edges.

### Dynamic Edge Attributes

Recomputed daily from RTMA wind at the source station:

- **upwind_cos**: $\cos(\theta_\text{wind\_from} - \theta_\text{bearing})$ -- is the neighbor upwind?
- **upwind_sin**: $\sin(\theta_\text{wind\_from} - \theta_\text{bearing})$ -- cross-wind relationship

where $\theta_\text{wind\_from} = \text{atan2}(-u, -v)$ is the direction the wind is coming from.

---

## Calm-Wind Loss Weighting

Calm conditions produce noisy and unreliable wind observations. The loss is weighted to reduce their influence:

$$
w_i = \text{clamp}\!\left(\frac{\text{wind}_{\text{rtma},i}}{\theta},\ w_\text{min},\ 1.0\right)
$$

| Parameter | Value |
|-----------|-------|
| $\theta$ (calm threshold) | 2.0 m/s |
| $w_\text{min}$ (minimum weight) | 0.1 |

The combined loss:

$$
\mathcal{L} = \text{mean}\bigl(w_i \cdot \text{Huber}(\hat{y}_i,\, y_i)\bigr)
$$

with Huber $\delta = 2.0$ m/s. The Huber loss provides robustness to remaining outliers, while calm weighting ensures the model focuses on conditions where corrections are physically meaningful.

An optional correction magnitude penalty ($\lambda \cdot \|\hat{y}\|^2$) is available but currently set to 0.

---

## Normalization

Z-score normalization per feature, same approach as the [Humidity MVP](../mvp/algorithms.md#normalization):

- Statistics computed from the training set only (exclude val days and holdout stations)
- Standard deviation clamped to $\geq 10^{-8}$
- Saved to `norm_stats.json` alongside checkpoints

```json
{
  "feature_cols": ["ugrd_rtma", "vgrd_rtma", "wind_rtma", ...],
  "norm_stats": {
    "ugrd_rtma": {"mean": -0.42, "std": 3.18},
    "vgrd_rtma": {"mean": 0.15, "std": 2.91},
    ...
  }
}
```

---

## Graph Precomputation

`prep/build_wind_graphs.py` precomputes per-day PyG `Data` objects:

- **Raw (unnormalized) features** for **all** stations per day
- Includes all 57 features regardless of experiment config
- Normalization, feature column selection, spatial holdout filtering applied at `__getitem__` time
- This keeps precomputed graphs **split-agnostic**: the same `.pt` files serve all experiments

Each `.pt` file contains:

| Field | Shape | Content |
|-------|-------|---------|
| `x` | $(N, 57)$ | Node features (raw) |
| `y` | $(N, 2)$ | Targets $(\delta_{par}, \delta_{perp})$ |
| `edge_index` | $(2, E)$ | Directed edges |
| `edge_attr` | $(E, 7)$ | Edge features |
| `rtma_wind` | $(N,)$ | RTMA wind speed (for loss weighting) |
| `fids` | list[str] | Station IDs (for holdout filtering) |

Total storage: ~3.7 GB for 2,448 days (2018--2024).

---

## Spatial Holdout

10% of stations are held out from training, stratified by elevation quartile:

1. Group all stations into 4 elevation quartiles
2. From each quartile, randomly select a proportional number of stations (seeded)
3. These stations are **excluded from training** but **included in validation**
4. Holdout station IDs are saved to `holdout_fids.json` for reproducibility

This ensures the model is evaluated on stations with diverse terrain characteristics, not just low-elevation valley stations.

---

## Metrics

| Metric | TensorBoard Tag | Description |
|--------|----------------|-------------|
| val_loss | `val_loss` | Huber + calm weighting |
| Parallel MAE | `val/mae_par` | MAE of $\hat{\delta}_{par}$ vs $\delta_{par}$ (m/s) |
| Perpendicular MAE | `val/mae_perp` | MAE of $\hat{\delta}_{perp}$ vs $\delta_{perp}$ (m/s) |
| Vector RMSE | `val/vector_rmse` | $\sqrt{\text{mean}(\|\hat{y} - y\|^2)}$ (m/s) |

!!! note "Baseline comparison"
    The RTMA baseline corresponds to zero correction ($\hat{\delta} = 0$). Any reduction in these metrics relative to the baseline indicates the model is improving RTMA wind estimates.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr = $10^{-3}$, weight_decay = $10^{-5}$) |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=10, min_lr=$10^{-6}$) |
| Precision | float16-mixed (GPU), float32 (CPU) |
| Gradient clipping | max_norm = 1.0 |
| Checkpointing | Top-3 by val_loss (ModelCheckpoint) |
| Early stopping | patience = 20 on val_loss |
| Batch size | 16 days per batch |
| Train split | 2018--2023 (temporal), holdout stations excluded |
| Val split | 2024 (temporal), holdout stations included |
| Hardware | RTX A6000, CUDA:0 |

---

## Ablation Plan

A 4-step ladder isolates the contribution of each architectural component:

| Step | Config | Model | Features | Node dim | Key test |
|------|--------|-------|----------|----------|----------|
| 1 | `wind_mlp_baseline.toml` | MLP (no graph) | RTMA + terrain + temporal + location | 20 | Local-only baseline |
| 2 | `wind_gnn_no_sx.toml` | GNN (1 hop) | Same as Step 1 + 7 edge features | 20 | Does neighbor context improve perp? |
| 3 | `wind_gnn_full.toml` | GNN (1 hop) | + Sx (34) + flow-terrain (3) | 57 | Does directional terrain exposure help? |
| 4 | `wind_gnn_2hop.toml` | GNN (2 hops) | Same as Step 3 | 57 | Does longer-range context help? |

---

## MLP Baseline Results (Step 1)

Best checkpoint: epoch 95 (`ckpt-epoch=095-val_loss=0.3624.ckpt`)

| Component | RTMA Baseline | MLP | Reduction |
|-----------|---------------|-----|-----------|
| Parallel MAE (m/s) | 1.371 | 0.686 | 50.0% |
| Perpendicular MAE (m/s) | 0.603 | 0.546 | 9.5% |
| Vector RMSE (m/s) | 2.179 | 1.440 | 33.9% |

### Learning Curve

| Epoch | val_loss | mae_par | mae_perp | vec_rmse |
|------:|---------:|--------:|---------:|---------:|
| 0 | 0.460 | 0.829 | 0.577 | 1.615 |
| 5 | 0.424 | 0.775 | 0.564 | 1.546 |
| 10 | 0.405 | 0.752 | 0.561 | 1.515 |
| 20 | 0.387 | 0.722 | 0.554 | 1.484 |
| 30 | 0.381 | 0.714 | 0.548 | 1.475 |
| 50 | 0.371 | 0.701 | 0.545 | 1.457 |
| 75 | 0.364 | 0.691 | 0.546 | 1.444 |
| 95 | 0.362 | 0.686 | 0.546 | 1.440 |

The model reaches 95% of final performance by epoch 30, with marginal gains thereafter.

---

## Inference Strategy

!!! note "Forward-looking"
    Inference over the full RTMA grid is planned but not yet implemented.

- **MLP mode**: trivially parallelizable. Each RTMA grid cell is treated as an independent node with the same feature vector as during training. A pointwise forward pass produces the correction at every cell.
- **GNN mode**: define a k-NN graph over RTMA grid cells, compute static edge attributes (distance, bearing, delta_elevation) and dynamic edge attributes (upwind_cos, upwind_sin from RTMA wind). Run the full forward pass to produce a corrected wind vector field over the PNW domain.
- **Output**: $\mathbf{w}_\text{corrected} = \mathbf{w}_\text{rtma} + \hat{\delta}_{par} \cdot \hat{e}_{par} + \hat{\delta}_{perp} \cdot \hat{e}_{perp}$ at every grid cell.
