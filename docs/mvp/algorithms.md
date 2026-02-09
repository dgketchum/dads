# Algorithms

---

## Target Variable

The model predicts a log-space residual between observed and RTMA-derived vapor pressure:

$$
\Delta\!\log e_a = \log(e_{a,\text{obs}}) - \log(e_{a,\text{rtma}})
$$

At inference the correction is applied multiplicatively:

$$
e_{a,\text{corrected}} = e_{a,\text{rtma}} \cdot \exp(\widehat{\Delta\!\log e_a})
$$

### Why log-space?

- **Variance stabilization.** Absolute humidity spans two orders of magnitude across dry continental winters and humid maritime summers. In linear space the loss would be dominated by high-humidity samples; log-space makes the correction scale-free.
- **Multiplicative structure.** RTMA biases tend to be proportional to magnitude (e.g., consistently 10% too dry). A log-space residual naturally captures this.
- **Non-negativity.** Exponentiation guarantees $e_{a,\text{corrected}} > 0$ regardless of the predicted correction magnitude.

### Why bias-correction, not direct prediction?

Run 4 tested direct prediction of $\log(e_{a,\text{obs}})$ without humidity inputs (dewpoint, $e_a$). The model converged to RTMA-level accuracy (MAE 0.110 vs RTMA baseline 0.111) -- it could reconstruct humidity from temperature, wind, and terrain, but could not *improve* on what RTMA already provides. The bias-correction framing (Runs 1--3, 5) achieves 28% improvement because the model has access to RTMA's estimate and only needs to learn the *residual error structure*.

---

## UNetSmall Architecture

A minimal 2-level encoder-decoder with skip connections, designed for fast iteration:

| Layer | Operation | Output Shape |
|-------|-----------|-------------|
| `down1` | ConvBlock($C_\text{in}$, $b$) | $(B, b, 64, 64)$ |
| `pool1` | MaxPool2d(2) | $(B, b, 32, 32)$ |
| `down2` | ConvBlock($b$, $2b$) | $(B, 2b, 32, 32)$ |
| `pool2` | MaxPool2d(2) | $(B, 2b, 16, 16)$ |
| `mid` | ConvBlock($2b$, $4b$) | $(B, 4b, 16, 16)$ |
| `up2` | ConvTranspose2d($4b$, $2b$, k=2, s=2) | $(B, 2b, 32, 32)$ |
| concat | cat(up2, skip₂) | $(B, 4b, 32, 32)$ |
| `dec2` | ConvBlock($4b$, $2b$) | $(B, 2b, 32, 32)$ |
| `up1` | ConvTranspose2d($2b$, $b$, k=2, s=2) | $(B, b, 64, 64)$ |
| concat | cat(up1, skip₁) | $(B, 2b, 64, 64)$ |
| `dec1` | ConvBlock($2b$, $b$) | $(B, b, 64, 64)$ |
| `out` | Conv2d($b$, 1, k=1) | $(B, 1, 64, 64)$ |

Each `ConvBlock` is: Conv2d(3×3, pad=1) $\to$ ReLU $\to$ Conv2d(3×3, pad=1) $\to$ ReLU.

| base ($b$) | Params | Runs |
|------------|--------|------|
| 32 | ~120K | 1, 2, 3, 4 |
| 48 | ~270K | 5 |

---

## Input Channel Stack

Channels are stacked in a fixed order determined by the feature group resolution:

| Slot | Channels | Source | Count |
|------|----------|--------|-------|
| 1--8 | tmp_c, dpt_c, ugrd, vgrd, pres_kpa, tcdc_pct, prcp_mm, ea_kpa | RTMA COG | 8 |
| 9--10 | doy_sin, doy_cos | Computed from date | 2 |
| 11--16 | elevation, slope, aspect_sin, aspect_cos, tpi_4, tpi_10 | Terrain GeoTIFF | 6 |
| 17 | rsun | r.sun GeoTIFF (DOY band) | 1 |
| 18--24 | ls_b2, ls_b3, ls_b4, ls_b5, ls_b6, ls_b7, ls_b10 | Landsat composite (period band) | 7 |
| | | **Total** | **24** |

All channels are extracted as aligned 64×64 patches and normalized to zero mean / unit variance before entering the model.

---

## Center-Pixel Supervision

Station observations provide point labels at patch centers. The model predicts a full 64×64 correction field, but the supervised signal comes from a single pixel:

$$
\mathcal{L}_\text{fit} = \text{Huber}\bigl(\hat{y}_{c,c},\ y_\text{obs}\bigr)
$$

where $\hat{y}_{c,c}$ is the predicted correction at the center pixel and $y_\text{obs}$ is the station-observed $\Delta\!\log e_a$. Huber loss (with $\delta = 1.0$) is used for robustness to the remaining outliers that pass the filtering threshold.

### Total Variation Regularization

To ensure spatial coherence of the predicted correction field (preventing the model from learning a spike at the center and noise elsewhere), a TV penalty is applied to the full patch output:

$$
\mathcal{L}_\text{TV} = \frac{1}{N}\sum_{i,j}\bigl(|\hat{y}_{i,j+1} - \hat{y}_{i,j}| + |\hat{y}_{i+1,j} - \hat{y}_{i,j}|\bigr)
$$

### Combined Loss

$$
\mathcal{L} = \mathcal{L}_\text{fit} + \lambda \cdot \mathcal{L}_\text{TV}
$$

with $\lambda = 10^{-3}$ (default). This balances center-pixel accuracy with smooth spatial fields suitable for gridded application.

---

## Normalization

Per-channel mean and standard deviation are computed from the training set during dataset initialization and saved to `norm_stats.json`:

```json
{
  "channels": ["tmp_c", "dpt_c", "ugrd", "vgrd", ...],
  "mean": [5.23, 1.87, -0.12, ...],
  "std": [10.1, 9.4, 3.2, ...]
}
```

- Statistics are computed over all valid (non-NaN) pixels in preloaded COGs.
- Standard deviation is clamped to $\geq 10^{-8}$ to avoid division by zero.
- Normalization is critical: Run 1 (unnormalized) achieved only 4% improvement vs baseline; Run 2 (normalized) jumped to 29%.
- The `norm_stats.json` file is saved alongside checkpoints and must be used at inference time.

---

## COG Preloading Strategy

LZW-compressed COGs are slow for random windowed reads:

| Access pattern | Latency | Throughput at 848K reads |
|---------------|---------|--------------------------|
| Random 64×64 windowed read | ~170 ms | ~40 hours/epoch |
| Preloaded numpy slice | ~0.2 ms | ~3 minutes/epoch |

The dataset preloads all unique daily COGs into RAM as decoded float16 arrays. With 2,556 COGs (7 years) at ~108 MB each in float16, this requires ~276 GB -- well within the machine's 540 GB.

Key implementation details:

- **Float16 storage**: COGs are decoded to float32 and downcast to float16 for storage, halving memory. Individual patches are upcast to float32 on access.
- **Threaded loading**: COGs are loaded in parallel via a thread pool.
- **Fork COW sharing**: DataLoader workers (forked processes) share preloaded memory via copy-on-write, so `num_workers=2--4` adds no memory overhead.
- **Training throughput**: ~25 min preload + ~5 min/epoch (A6000, float16-mixed).

---

## Metrics

### Log-space metrics ($\Delta\!\log e_a$)

| Metric | Tag | Description |
|--------|-----|-------------|
| val_loss | `val_loss` | Huber + TV at center pixel |
| val_mae | `val_mae` | MAE of predicted vs observed residual |
| val_rmse | `val_rmse` | RMSE |
| val_r2 | `val_r2` | Fraction of residual variance explained |
| val_bias | `val_bias` | Mean signed error (pred $-$ target) |
| baseline_mae | `baseline_mae` | MAE of zero-correction (raw RTMA) |
| val_rtma_mae | `val_rtma_mae` | MAE of RTMA baseline in log-space |

### $e_a$-space metrics (kPa)

These reconstruct actual vapor pressure values for stakeholder-interpretable evaluation:

| Metric | Tag | Description |
|--------|-----|-------------|
| Model MAE | `val/mae_ea_kpa` | MAE of corrected $e_a$ vs observed |
| RTMA MAE | `val/mae_rtma_ea_kpa` | MAE of uncorrected RTMA vs observed |
| Model MAPE | `val/mape_ea` | Mean absolute percentage error |
| RTMA MAPE | `val/mape_rtma_ea` | RTMA MAPE for comparison |
| Improvement | `val/pct_improvement` | $(1 - \text{MAE}_\text{model} / \text{MAE}_\text{rtma}) \times 100$ |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW, lr = $3 \times 10^{-4}$ |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=7, min_lr=$10^{-6}$) |
| Precision | float16-mixed (GPU), float32 (CPU) |
| Gradient clipping | max_norm = 1.0 |
| Checkpointing | Top-5 by val_loss (ModelCheckpoint) |
| Early stopping | patience = 20 on val_loss |
| Train/val split | 80/20 by station (seeded), shared across all runs |

---

## Ablation Results

Five runs tested incremental additions to the input stack:

| Run | Inputs | Channels | val_mae | $\Delta$ vs RTMA | R² |
|-----|--------|----------|---------|-------------------|-----|
| 1 | RTMA only (no norm) | 10 | 0.107 | $-$4% | 0.02 |
| 2 | + terrain + rsun + **normalization** | 17 | 0.079 | $-$29% | 0.61 |
| 3 | + Landsat | 24 | 0.080 | $-$28% | 0.69 |
| 4 | Drop humidity inputs (direct pred.) | 22 | 0.110 | $-$1% | 0.89* |
| 5 | All features, base=48, 7 years | 24 | 0.073 | $-$29% | 0.60 |

*Run 4's R² is against total $\log e_a$ variance, not residual variance.

Key findings:

- **Normalization is essential** (Run 1 → 2): without it, high-magnitude bands (pressure) dominate and the model barely learns.
- **Terrain covariates provide the largest single gain** (2% → 29% improvement): elevation, slope, and TPI give the model spatial context about where RTMA struggles.
- **Landsat adds ~8 pp of explained variance** (R² 0.61 → 0.69) at fixed epochs, providing land-cover context.
- **Humidity inputs are essential** (Run 4): without dewpoint and $e_a$ as inputs, the model cannot improve on RTMA.
- **More data improves generalization** (Run 5): 7 years of training produces more robust corrections than 1 year, though per-year R² is slightly lower due to greater inter-annual variance.

---

## LR Scheduling

The learning rate follows a ReduceLROnPlateau schedule:

| Parameter | Value |
|-----------|-------|
| Monitor | val_loss |
| Factor | 0.5 (halve LR) |
| Patience | 7 epochs without improvement |
| Min LR | $10^{-6}$ |

In Run 5 (24 epochs), the scheduler had not yet triggered a reduction -- the model was still improving at the initial learning rate. With early stopping patience of 20, training can continue for up to 100 epochs.
