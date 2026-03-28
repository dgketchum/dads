# Grid Feature Policy

## Purpose

This note complements `policy/VALIDATION_POLICY.md`.

Its purpose is to freeze a canonical grid/query feature family for benchmark
architecture comparisons so that results are not confounded by changing:

- grid feature inventories
- feature order
- tiling schemes
- sparse supervision geometry
- input normalization
- raster/query alignment

The immediate use case is to establish the relative efficacy of:

- single-head grid residual correction
- 2-head multitask grid residual correction
- later comparisons to `core-graph-v0`

under one fixed information set.

## Scope

This policy applies to benchmark-grade HRRR background grid/query residual
correction experiments.

It does not replace `policy/VALIDATION_POLICY.md`.
That document still governs:

- train/validation/test protocol
- holdouts
- leakage rules
- benchmark vs publication semantics

This document freezes the grid and feature definition used within that
benchmark protocol.

## Canonical Family

Define a canonical benchmark family named:

- `grid-core-v0`

All single-vs-multitask grid comparisons should use this family unless an
explicit new version is created.

Any change to:

- grid feature inventory
- feature order
- target grid definition
- tile size
- tile overlap / stride policy
- sparse query supervision geometry
- input normalization policy
- station-to-grid sampling rule

creates a new family version, for example `grid-core-v1`.

Do not compare results across family versions as if they were the same
benchmark.

## Grid Policy

For `grid-core-v0`:

- frozen tensor semantics: the 37-channel input tensor is assembled at
  training time from pixel-aligned raster sources; the family is defined by
  the feature inventory, channel order, normalization, tiling, and
  supervision rules below — not by any particular artifact format
- one fixed target grid definition
- one fixed raster alignment / projection (EPSG:5070, 1 km)
- all co-registered raster sources must share the same CRS, transform, and
  spatial extent; the dataset must assert this at init
- CDR (EPSG:4326, 0.05°) is the sole non-aligned source; its grid geometry
  must be consistent across all days used in a run
- keep tiling fixed across all compared runs

The scored benchmark population is not the full grid. It is the held-out
station/query set sampled from that grid under
`policy/VALIDATION_POLICY.md`.

## Tile Policy

`grid-core-v0` uses:

- fixed tile size: `64 x 64` grid cells
- fixed stride policy recorded in artifact metadata
- fixed station/query sampling rule inside each tile

If tile size or stride changes, that is a new family version.

Tiles may overlap, but overlap policy must be:

- fixed across all runs in the family
- recorded in artifact metadata
- held constant for single-head vs multitask comparisons

## Sparse Supervision Policy

This is the defining rule of `grid-core-v0`:

- supervise all eligible station/query points that fall inside a tile
- do not use center-pixel-only supervision

Each sample must carry:

- grid input tensor
- query point coordinates inside the tile
- target values for those query points
- target valid mask
- split-derived loss mask

This avoids the location-fingerprinting failure mode documented in the older
U-Net experiments.

## Grid Feature Policy

`grid-core-v0` should mirror the `core-graph-v0` information set as closely as
possible in raster/query form.

### HRRR weather

- `dpt_hrrr`
- `dswrf_hrrr`
- `ea_hrrr`
- `hpbl_hrrr`
- `pres_hrrr`
- `spfh_hrrr`
- `tcdc_hrrr`
- `tmax_hrrr`
- `tmin_hrrr`
- `tmp_hrrr`
- `ugrd_hrrr`
- `vgrd_hrrr`
- `wdir_hrrr`
- `wind_hrrr`

### CDR

- `bt15_cdr`
- `cloud_state_cdr`
- `i1_cdr`
- `i2_cdr`
- `szen_cdr`

### Terrain

- `elevation`
- `slope`
- `aspect_sin`
- `aspect_cos`
- `tpi_4`
- `tpi_10`

### Radiation / remote sensing

- `rsun`
- `ls_b2`
- `ls_b3`
- `ls_b4`
- `ls_b5`
- `ls_b6`
- `ls_b7`
- `ls_b10`

### Temporal / location

- `doy_sin`
- `doy_cos`
- `latitude`
- `longitude`

Total: 37 grid features.

## Exclusions

The following are excluded from `grid-core-v0`:

- station observations
- innovations
- rasterized observation channels
- source-density channels derived from observations
- station ID or fid features
- learned per-station embeddings
- DA source pathways

If any of these are added, that experiment is not `grid-core-v0`.

## Feature Order Policy

Feature order must be frozen and recorded in run metadata.

Every `grid-core-v0` training run must emit the following in its run
directory (alongside `experiment.toml` and `norm_stats.json`):

- `feature_manifest.json`: ordered list of input channel names
- `target_manifest.json`: ordered list of target names
- `tile_manifest.json`: tile size, stride policy, overlap policy
- `query_sampling_manifest.json`: supervision rule, scoring rule, holdout
  gating
- `split_pointer.json`: paths to holdout artifact and train/val year spec

Training code must load and preserve feature order exactly.

## Query Sampling Policy

For benchmark scoring:

- held-out stations act as proxy query locations
- model output is sampled at those query locations
- raw HRRR baseline is computed at those same query locations

The station-to-grid sampling rule must be fixed and recorded in metadata.

Recommended default:

- bilinear interpolation of grid predictions to station locations

If sampling rule changes, that is a new family version.

## Input Normalization Policy

Use one input normalization policy for the entire family.

For `grid-core-v0`:

- fit normalization on training-year, non-holdout-supervised tiles / query
  populations only
- reuse the same input stats for validation
- keep the input policy identical across single-head and multitask runs

If input normalization changes, that is a new family version.

Whatever input policy is used, it must be:

- fit on training data only
- identical across all compared runs
- documented in run metadata

## Regularization Policy

Because the old patch experiments overfit via spatial memorization,
`grid-core-v0` should include baseline regularization from the start:

- normalization in conv blocks or equivalent
- dropout in the backbone
- no center-pixel-only supervision

Spatial augmentation may be used only if:

- it is documented in the artifact/run metadata
- it is held fixed across compared runs

If augmentation policy changes, that is a new family version.

## Output / Loss Scaling Policy

For multitask runs, target heads must be loss-scaled so raw unit differences do
not dominate the shared trunk.

For benchmark-grade multitask comparisons:

- fit per-head scale values on training data only
- use robust scales such as train-set MAD, IQR-derived scale, or train-set
  baseline MAE
- compute multitask loss in normalized target space
- report validation metrics in raw physical units

Do not use raw-unit composite loss alone to decide whether multitask is better
than single-head.

## Benchmark Protocol

All `grid-core-v0` comparisons must follow `policy/VALIDATION_POLICY.md` in
`benchmark` mode.

That means:

- train on 2018-2023
- validate on 2024
- use canonical holdout fids
- compute training loss on non-holdout query points only
- compute validation loss on holdout query points only
- treat held-out stations as proxy query locations

## Model Comparison Policy

When the question is single-head vs multitask, freeze:

- raster source directories
- feature inventory
- feature order
- target grid definition
- tile size
- stride / overlap policy
- split artifact
- backbone family
- depth / width
- dropout
- optimizer
- learning rate schedule
- batch size
- random seed set

Only the target configuration may vary.

## Experiment Ladder

Use this sequence to establish the relative efficacy of single vs multitask.

### Stage 1: Single-head baselines

Run:

- `S-tmax`: predict `delta_tmax` only
- `S-tmin`: predict `delta_tmin` only

These are the reference baselines.

### Stage 2: Minimal multitask comparison

Run:

- `M-2head`: predict `delta_tmax` and `delta_tmin`

This is the primary multitask test.

## Comparison Rule

`grid-core-v0` results may be compared directly to `core-graph-v0` only if
they share:

- the same holdout artifact
- the same train/val year split
- the same held-out station scoring population
- the same raw HRRR baseline definition
- the same target definition

The comparison is between spatial representations, not between different
benchmark populations.

## Benchmark Validity Checks

Before treating a run as part of `grid-core-v0`, verify:

- run metadata matches the 37-feature manifest
- no observation-derived grid channels are present
- tile size and stride match the family definition
- query sampling is all-query sparse supervision, not center-pixel only
- holdout scoring population is explicitly recorded
- raw HRRR baseline is computed on the same scored query points
- input normalization is fit on training data only
- backbone includes batch normalization and dropout (see Regularization
  Policy)

## Current Decision Rule

Use `grid-core-v0` to answer three narrow questions:

1. Can a grid/query residual model beat raw HRRR on the held-out benchmark
   population?
2. Does it beat `core-graph-v0` under the same benchmark?
3. Does 2-head multitask help or hurt in the grid family?

Do not mix DA into this family.
