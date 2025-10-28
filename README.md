Data Assimilation and DownScaling (DADS) turns scattered observations and remote‑sensing signals into coherent,
local‑scale weather intelligence. Its aim is simple: squeeze every bit of information we can from the existing
meteorological and remote sensing observation record to produce robust, geographically aware estimates 
where information is sparse. DADS treats space
and time as first‑class structure: a spatial graph anchors stations to their geographic neighborhoods; sequential models
capture day‑to‑day evolution; representation learning summarizes longer‑term regime similarity. The system is
deliberately modular: raw inputs are unified into a consistent station‑time schema; a single, variable‑specific scaler
standardizes features across all stages; the graph encodes spatial bias without overfitting to any one dataset; and
lightweight neural components compose into an end‑to‑end predictor. Reproducibility and transparency are
priorities—splits are deterministic, feature order and scaling are persisted, and each artifact can be inspected or
swapped without disturbing the whole. DADS is built for breadth (continental domains, long time spans) and depth (
high‑resolution sensing, diverse stations), embracing imperfect coverage by elevating day‑aware selection over heavy
prefiltering. The result is a practical pathway from raw data to station‑level predictions that respect geography,
leverage modern learning, and remain auditable at every step.

# dads

Data Assimilation and DownScaling

## Data Extraction (Inputs)

- Station metadata
    - Consolidate station inventories from trusted sources into a single layer with stable identifiers and locations.
    - Keep a unique station id (fid) and geographic coordinates; tile/index fields are optional but helpful for
      batching.

- Observed station time series (targets)
    - Curate daily observations for the variables of interest (e.g., temperature, irradiance, precipitation, humidity,
      wind).
    - Clean and align to a consistent timeline per station; do not mix in gridded “comparators” at this stage.

- Remote sensing features
    - Provide per‑station time series of spectral/radiative features aligned to station dates, with clear provenance and
      stable band names.

- Static descriptors
    - Supply clear‑sky solar by day‑of‑year and terrain metrics (slope, aspect, elevation, TPI) as station‑level
      features.

Notes

- All per‑station files should use the same station id and share a common or joinable date index.
- Comparator or validation grids are separate inputs and not part of extraction.

## Training Pipeline

- Build per‑station sequences with a consistent schema, aligning observation targets, remote‑sensing features, solar
  proxy, and terrain.
- Fit a variable‑specific MinMax scaler and persist it with feature names; use the same scaler across all modeling
  stages.
- Construct a spatial graph that encodes geographic neighborhoods; persist splits, node index, and graph metadata for
  reproducibility.
- Produce near‑term, day‑specific neighbor contexts on‑the‑fly via a lightweight temporal encoder (TCN) over
  12‑day windows; no disk caching required.
- Train a representation model to produce station embeddings from multi‑feature sequences.
- Train a lightweight spatio‑temporal GNN that fuses embeddings, edge attributes, and node contexts to predict the
  target.
- Validate shapes and consistency before long runs; monitor metrics and checkpoints during training.

## When to rerun what

- Changing raw inputs or sequence schema: rebuild sequences, refit the scaler, rebuild the graph, retrain
  sequence/representation models, recache contexts and embeddings, then retrain the GNN.
- Changing graph logic only: rebuild the graph and retrain the GNN (reuse contexts and embeddings).
- Changing the sequence model: retrain it, recache node contexts, retrain the GNN.
- Changing the representation model: re‑infer/retrain embeddings, retrain the GNN.

## Determinism

- Graph splits use `np.random.default_rng(random_state)`.
- Dataset neighbor sampling seeds a local RNG from `(target_station, day_int)`.
