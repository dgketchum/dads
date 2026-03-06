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

The same argument as the [Humidity MVP](../humidity/algorithms.md#why-bias-correction-not-direct-prediction): by providing the RTMA estimate as input, the model only needs to learn the *residual error structure*, not reconstruct the full wind field from scratch. This makes the task much easier and the corrections more stable.

---

## GNN Overview

The model learns to correct RTMA wind at station locations by combining each station's own features with information gathered from nearby stations. In plain language, here is what happens end to end:

**1. Encode each station independently.** Every station's raw features (RTMA weather, terrain, Sx sheltering, etc.) are passed through a small MLP to produce a hidden vector. At this point no station knows anything about its neighbors.

**2. Each neighbor makes a case for how relevant it is.** For every edge j→i (neighbor j reporting to target station i), the model concatenates three things: the target's hidden state, the neighbor's hidden state, and the edge features (distance, bearing, elevation difference, upwind alignment). This combined vector goes through an MLP that outputs a single scalar — an importance score for that neighbor.

**3. Scores become weights via softmax.** The importance scores for all neighbors of a given target station are passed through a softmax so they sum to 1. This is the attention — the model learns which neighbors matter most. A nearby upwind station on similar terrain might get weight 0.4, while a distant station behind a ridge gets 0.02.

**4. Weighted sum produces context.** Each neighbor's hidden state is scaled by its attention weight and these are summed to produce a single context vector for the target station — a neighborhood summary of what surrounding stations collectively say about local conditions.

**5. Merge local + context.** The target's own hidden state and the neighborhood context are concatenated and compressed back down through a merge MLP. If there are 2 hops, steps 2–5 repeat so that second-hop neighbors effectively see information from two edges away.

**6. Predict.** The final output head takes the station's pre-attention representation alongside its post-attention representation and predicts two wind bias components (parallel and perpendicular to RTMA flow).

### Edge selection

The neighbor graph is built in two stages. First, a static k-NN map is precomputed once over the full station inventory using geographic distance (k=16). Second, on each training day the graph is pruned to only stations present in that day's active set — neighbors that didn't report are simply dropped, so the topology varies day to day.

Importantly, **neighbor selection is purely spatial — wind direction does not determine which stations are connected.** All k=16 nearest neighbors are included regardless of whether they are upwind, downwind, or crosswind. What wind direction *does* control is a pair of edge features: `upwind_cos` and `upwind_sin`, which encode the angular relationship between the day's RTMA wind direction and the bearing from neighbor to target. The attention mechanism then learns to weight upwind neighbors more heavily when that context is informative — but this is a soft, learned preference, not a hard selection rule. A downwind neighbor on similar terrain may still receive substantial attention weight if the model finds it useful.

Edge features combine static geometry (distance, bearing, delta-elevation) with these per-day dynamic upwind alignment components. Because edges are informed entirely by gridded RTMA fields and precomputed terrain — both of which have near-complete spatial coverage — missing data in edge construction is rare.

---

## EdgeGatedAttention Mechanism

The key distinction from standard graph attention (like GAT) is that **edge features directly participate in computing attention weights**. The model doesn't just ask "how similar are stations i and j?" — it asks "how similar are they *given that j is 12 km away, bearing northwest, 300 m higher, and directly upwind?*" The spatial and topographic relationship between two stations modulates how much one should influence the other. This is what "edge-gated" means: the edge attributes gate the flow of information between nodes.

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

Z-score normalization per feature, same approach as the [Humidity MVP](../humidity/algorithms.md#normalization):

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

## Ablation Results

### Step 1: MLP Baseline

Best checkpoint: epoch 95 (`ckpt-epoch=095-val_loss=0.3624.ckpt`)

| Component | RTMA Baseline | MLP | Reduction |
|-----------|---------------|-----|-----------|
| Parallel MAE (m/s) | 1.371 | 0.686 | 50.0% |
| Perpendicular MAE (m/s) | 0.603 | 0.546 | 9.5% |
| Vector RMSE (m/s) | 2.179 | 1.440 | 33.9% |

### Step 2: GNN no-Sx

Best checkpoint: epoch 94 (`val_loss=0.3394`)

| Component | MLP (Step 1) | GNN no-Sx (Step 2) | Δ vs MLP |
|-----------|--------------|---------------------|----------|
| Parallel MAE (m/s) | 0.686 | 0.656 | −4.4% |
| Perpendicular MAE (m/s) | 0.546 | 0.523 | −4.2% |
| Vector RMSE (m/s) | 1.440 | 1.393 | −3.3% |
| val_loss | 0.362 | 0.339 | −6.4% |

Neighbor attention improves both components, not just perpendicular as initially hypothesized. The edge-gated attention mechanism learns useful spatial relationships even without directional terrain (Sx) features.

### Cumulative Summary

| Component | RTMA | MLP | GNN no-Sx | Total reduction |
|-----------|------|-----|-----------|-----------------|
| Parallel MAE (m/s) | 1.371 | 0.686 | 0.656 | **52.2%** |
| Perpendicular MAE (m/s) | 0.603 | 0.546 | 0.523 | **13.3%** |
| Vector RMSE (m/s) | 2.179 | 1.440 | 1.393 | **36.1%** |

---

## Gridded Inference

The model is trained on station graphs but needs no station observations at inference time. Every input feature — RTMA weather fields, terrain, Sx sheltering indices, flow-terrain interactions — comes from gridded products that are available at any location. Station observations only ever appear in the training targets (the residual between observed and RTMA wind). This means the trained model can be applied directly to a regular grid to produce a wall-to-wall corrected wind field.

!!! note "Forward-looking"
    Inference over the full RTMA grid is planned but not yet implemented.

### Approach

**1. Build a grid graph.** Lay down nodes on the RTMA grid (or any desired output resolution) over the domain of interest. Each node's features are populated from the same sources used in training: RTMA analysis fields for that day, precomputed terrain attributes, and Sx values derived from terrain + the day's RTMA wind direction.

**2. Connect neighbors.** Build edges using the same spatial k-NN logic used in training. For a regular grid the neighbor structure is uniform — each cell connects to its k nearest grid cells. Edge features (distance, bearing, delta-elevation, upwind alignment) are computed identically to training.

**3. Run message passing.** The trained node encoder, attention layers, and output head process the grid graph exactly as they processed station graphs. Each grid cell attends to its neighbors, gathers context about the surrounding terrain and weather regime, and predicts a wind correction.

**4. Apply corrections.** The model outputs parallel and perpendicular bias components at each grid cell. These are projected back into u/v space using the RTMA flow direction at that cell:

$$
\mathbf{w}_\text{corrected} = \mathbf{w}_\text{rtma} + \hat{\delta}_{par} \cdot \hat{e}_{par} + \hat{\delta}_{perp} \cdot \hat{e}_{perp}
$$

### MLP mode

For the MLP baseline (no graph), inference is trivially parallelizable — each RTMA grid cell is treated as an independent node with the same feature vector as during training. A pointwise forward pass produces the correction at every cell.

### Domain shift considerations

Training graphs are sparse and irregular (thousands of stations, unevenly spaced) while inference graphs are dense and regular (potentially millions of grid cells at uniform spacing). The attention weights learned on sparse station neighborhoods may behave differently on dense grid neighborhoods — for example, a model trained with neighbors typically 10–50 km away will see neighbors at ~2.5 km spacing on the RTMA grid. Whether this transfer works well is an empirical question. Potential mitigations include training on subsampled grid patches or fine-tuning on grid-structured inputs.

A second consideration is that training stations are not uniformly distributed — they cluster in valleys and near population centers. The model may have less exposure to ridgetop or remote terrain configurations. Spatial holdout evaluation partially tests this, but gridded inference will inevitably extrapolate to terrain positions underrepresented in the station network.
