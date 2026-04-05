"""
DA-aware patch dataset for grid backbone Stage C.

Extends ``HRRRPatchDataset`` to also return source station tensors
(context, payload, raw elevation) needed by ``GridDAFusion``.

Held-out stations are excluded from the source set but remain as
supervision query targets via the existing ``sta_holdout`` mask.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from models.hrrr_da.patch_assim_dataset import (
    HRRRPatchDataset,
    _build_day_slices,
    _load_or_build_tile_index,
)

# Default DA payload columns (aligned with DA v0 policy).
DEFAULT_PAYLOAD_COLS = ["delta_tmax", "delta_tmin"]


class GridDAPatchDataset(Dataset):
    """Wraps ``HRRRPatchDataset`` and augments each sample with source DA tensors.

    Parameters
    ----------
    base_kwargs : dict
        All keyword arguments forwarded to ``HRRRPatchDataset.__init__``.
    payload_cols : list[str]
        Observation-derived columns to include in the source payload.
        A ``valid_<col>`` binary flag is appended for each.
    teacher_dir, teacher_pattern : str | None
        Optional URMA teacher directory for dense stabiliser loss in Stage C.
        If provided, returns ``y_dense`` and ``y_valid`` like ``DenseIncrementDataset``.
    increment_map : dict[str, str] | None
        ``{urma_band: hrrr_band}`` for dense teacher targets.
    """

    def __init__(
        self,
        payload_cols: list[str] | None = None,
        teacher_dir: str | None = None,
        teacher_pattern: str = "URMA_1km_{date}.tif",
        increment_map: dict[str, str] | None = None,
        **base_kwargs,
    ):
        self._base = HRRRPatchDataset(**base_kwargs)
        self.payload_cols = list(payload_cols or DEFAULT_PAYLOAD_COLS)
        self.source_pay_dim = len(self.payload_cols) * 2  # value + valid flag each

        self.teacher_dir = teacher_dir
        self.teacher_pattern = teacher_pattern
        self.increment_map = dict(increment_map or {})

        # URMA / HRRR band indices for dense teacher (optional)
        self._urma_band_idx: dict[str, int] = {}
        self._hrrr_band_idx: dict[str, int] = {}
        if self.teacher_dir and self.increment_map:
            from models.hrrr_da.hetero_dataset import _discover_band_names
            import os

            # Search for a valid URMA file (not just the first sample day)
            for sample_day in self._base.samples["day"].drop_duplicates().sort_values():
                urma_path = os.path.join(
                    self.teacher_dir,
                    self.teacher_pattern.format(
                        date=pd.Timestamp(sample_day).strftime("%Y%m%d")
                    ),
                )
                if os.path.exists(urma_path):
                    urma_names = _discover_band_names(urma_path, "urma")
                    self._urma_band_idx = {n: i for i, n in enumerate(urma_names)}
                    break

            # Full HRRR band names (pre-drop) for increment lookup
            example_bg = self._base._background_path(self._base.samples.iloc[0]["day"])
            bg_names = _discover_band_names(example_bg, "bg")
            self._hrrr_band_idx = {n: i for i, n in enumerate(bg_names)}

        # Pre-load raw elevation band index from static terrain TIF
        self._elev_band_idx: int | None = None
        if self._base.static_tifs:
            for tif_path, names in zip(
                self._base.static_tifs, self._base._static_tif_band_names
            ):
                for i, n in enumerate(names):
                    if n in ("elevation", "terrain_pnw_1km_0"):
                        self._elev_tif = tif_path
                        self._elev_band_idx = i
                        break
                if self._elev_band_idx is not None:
                    break

    # Delegate properties
    @property
    def samples(self):
        return self._base.samples

    @property
    def in_channels(self):
        return self._base.in_channels

    @property
    def feature_names(self):
        return self._base.feature_names

    @property
    def target_names(self):
        return self._base.target_names

    @property
    def norm_stats(self):
        return self._base.norm_stats

    def __len__(self):
        return len(self._base)

    def _augment_with_da(
        self,
        base_sample: tuple,
        x_patch: torch.Tensor,
        day: pd.Timestamp,
        r0: int,
        c0: int,
        *,
        prefetched_neighbors: tuple | None = None,
        train_query_frac: float = 0.0,
        train_min_query_source_dist_px: int = 0,
        train_source_dropout_prob: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> dict:
        """Add DA source tensors to a base patch sample.

        This is the shared augmentation logic used by both the station-centered
        ``__getitem__`` and the day-tile variant.

        When ``train_query_frac > 0``, non-holdout stations are split into
        disjoint source and query subsets.  Source stations provide DA tensors;
        query stations are supervised with the full (bg + DA) prediction.
        Source stations too close to any query station (within
        ``train_min_query_source_dist_px``) are removed from the source set.
        """
        (
            _x_patch,
            sta_rows,
            sta_cols,
            sta_targets,
            sta_valid,
            sta_holdout,
            sta_is_center,
        ) = base_sample

        H = W = self._base.patch_size
        n_sta = sta_rows.shape[0]

        # --- Source stations: non-holdout in-patch stations ---
        if prefetched_neighbors is not None:
            day_group, sta_r, sta_c = prefetched_neighbors
        else:
            day_group, sta_r, sta_c = self._base._neighbor_day_data(day)

        # Collect ALL non-holdout in-patch stations with their pixel coords and
        # payload.  We need the full list before partitioning.
        candidate_rows: list[int] = []
        candidate_cols: list[int] = []
        candidate_pay: list[list[float]] = []
        candidate_fids: list[str] = []

        if day_group is not None and sta_r is not None and not day_group.empty:
            in_patch = (
                (sta_r >= r0) & (sta_r < r0 + H) & (sta_c >= c0) & (sta_c < c0 + W)
            )
            for j in np.where(in_patch)[0]:
                fid_j = str(day_group.iloc[j]["fid"])
                if fid_j in self._base._holdout_fids:
                    continue
                candidate_rows.append(int(sta_r[j] - r0))
                candidate_cols.append(int(sta_c[j] - c0))
                candidate_fids.append(fid_j)
                pay = []
                for col in self.payload_cols:
                    val = day_group.iloc[j].get(col, float("nan"))
                    is_valid = pd.notna(val)
                    pay.append(float(val) if is_valid else 0.0)
                    pay.append(1.0 if is_valid else 0.0)
                candidate_pay.append(pay)

        n_cand = len(candidate_rows)

        # --- Source / query partition (train only) ---
        # sta_is_source / sta_is_query are per-supervision-station masks that
        # indicate the role assigned during this sample.  For validation
        # (train_query_frac == 0) all non-holdout stations are sources and
        # is_query is all-False.
        sta_is_source = torch.zeros(n_sta, dtype=torch.bool)
        sta_is_query = torch.zeros(n_sta, dtype=torch.bool)

        if train_query_frac > 0 and n_cand >= 2:
            if rng is None:
                rng = np.random.default_rng()
            n_query = max(1, int(round(n_cand * train_query_frac)))
            n_query = min(n_query, n_cand - 1)  # keep at least 1 source
            perm = rng.permutation(n_cand)
            query_idx = set(perm[:n_query].tolist())
            source_idx = set(perm[n_query:].tolist())

            # Enforce min-distance: remove sources too close to any query
            if train_min_query_source_dist_px > 0:
                qr = np.array([candidate_rows[i] for i in query_idx])
                qc = np.array([candidate_cols[i] for i in query_idx])
                to_remove = set()
                for si in source_idx:
                    sr, sc = candidate_rows[si], candidate_cols[si]
                    dists = np.sqrt((qr - sr) ** 2 + (qc - sc) ** 2)
                    if dists.min() < train_min_query_source_dist_px:
                        to_remove.add(si)
                source_idx -= to_remove

            # Source dropout
            if train_source_dropout_prob > 0 and len(source_idx) > 1:
                keep = rng.random(len(source_idx)) >= train_source_dropout_prob
                if not keep.any():
                    keep[0] = True  # keep at least one
                source_idx = {s for s, k in zip(sorted(source_idx), keep) if k}

            # Map candidate indices → supervision-station masks.
            # Both _build_patch_sample and the candidate loop iterate the same
            # day_group rows in the same order, skipping holdout FIDs.  So the
            # k-th non-holdout supervision station corresponds to candidate
            # index k.  We use a sequential counter rather than pixel-coordinate
            # matching to handle multiple stations on the same 1 km cell.
            cand_idx = 0
            for si in range(n_sta):
                if sta_holdout[si]:
                    continue
                if cand_idx < n_cand:
                    if cand_idx in source_idx:
                        sta_is_source[si] = True
                    elif cand_idx in query_idx:
                        sta_is_query[si] = True
                cand_idx += 1
        elif train_query_frac > 0 and n_cand < 2:
            # Sparse tile: not enough stations for a split.  Mark all
            # non-holdout as source so they get background-only supervision
            # (no DA query loss, consistent with the split contract).
            for si in range(n_sta):
                if not sta_holdout[si]:
                    sta_is_source[si] = True
            source_idx = set(range(n_cand))
        else:
            # No partition: all non-holdout stations are sources (legacy path
            # and validation).
            for si in range(n_sta):
                if not sta_holdout[si]:
                    sta_is_source[si] = True
            source_idx = set(range(n_cand))

        # Build DA source tensors from the source subset only
        src_rows_list = (
            [candidate_rows[i] for i in sorted(source_idx)] if n_cand > 0 else []
        )
        src_cols_list = (
            [candidate_cols[i] for i in sorted(source_idx)] if n_cand > 0 else []
        )
        src_pay_list = (
            [candidate_pay[i] for i in sorted(source_idx)] if n_cand > 0 else []
        )

        n_src = len(src_rows_list)

        # Source context: extract from normalised x_patch at source pixels
        if n_src > 0:
            src_r = torch.tensor(src_rows_list, dtype=torch.long)
            src_c = torch.tensor(src_cols_list, dtype=torch.long)
            src_ctx = x_patch[:, src_r, src_c].T  # (n_src, C)
            src_pay = torch.tensor(src_pay_list, dtype=torch.float32)
        else:
            src_r = torch.zeros(0, dtype=torch.long)
            src_c = torch.zeros(0, dtype=torch.long)
            src_ctx = torch.zeros(0, x_patch.shape[0], dtype=torch.float32)
            src_pay = torch.zeros(0, self.source_pay_dim, dtype=torch.float32)

        src_valid = torch.ones(n_src, dtype=torch.bool)

        # Raw elevation patch (un-normalised) for geometry computation
        if self._elev_band_idx is not None:
            elev_data = self._base.raster_cache.get(self._elev_tif)
            raw_elev = np.nan_to_num(
                elev_data["data"][self._elev_band_idx, r0 : r0 + H, c0 : c0 + W].astype(
                    "float32"
                ),
                nan=0.0,
            )
            raw_elev_patch = torch.from_numpy(raw_elev).unsqueeze(0)  # (1, H, W)
        else:
            raw_elev_patch = torch.zeros(1, H, W)

        # Source elevation
        if n_src > 0 and self._elev_band_idx is not None:
            src_elev = raw_elev_patch[0, src_r, src_c]  # (n_src,)
        else:
            src_elev = torch.zeros(n_src)

        # --- Optional dense teacher target ---
        y_dense: torch.Tensor | None = None
        y_valid_dense: torch.Tensor | None = None
        if self.teacher_dir and self.increment_map:
            import os

            urma_path = os.path.join(
                self.teacher_dir,
                self.teacher_pattern.format(date=day.strftime("%Y%m%d")),
            )
            if os.path.exists(urma_path) and self._urma_band_idx:
                urma_data = self._base.raster_cache.get(urma_path)
                bg_data = self._base.raster_cache.get(self._base._background_path(day))
                n_t = len(self.increment_map)
                yd = np.empty((n_t, H, W), dtype="float32")
                yv = np.empty((n_t, H, W), dtype="bool")
                for t_i, (uname, hname) in enumerate(self.increment_map.items()):
                    if uname in self._urma_band_idx and hname in self._hrrr_band_idx:
                        u = urma_data["data"][
                            self._urma_band_idx[uname], r0 : r0 + H, c0 : c0 + W
                        ].astype("float32")
                        h = bg_data["data"][
                            self._hrrr_band_idx[hname], r0 : r0 + H, c0 : c0 + W
                        ].astype("float32")
                        delta = u - h
                        valid = np.isfinite(delta)
                        yd[t_i] = np.where(valid, delta, 0.0)
                        yv[t_i] = valid
                    else:
                        yd[t_i] = 0.0
                        yv[t_i] = False
                y_dense = torch.from_numpy(yd)
                y_valid_dense = torch.from_numpy(yv)

        return {
            "x_patch": x_patch,
            "sta_rows": sta_rows,
            "sta_cols": sta_cols,
            "sta_targets": sta_targets,
            "sta_valid": sta_valid,
            "sta_holdout": sta_holdout,
            "sta_is_center": sta_is_center,
            "sta_is_source": sta_is_source,
            "sta_is_query": sta_is_query,
            "src_rows": src_r,
            "src_cols": src_c,
            "src_ctx": src_ctx,
            "src_pay": src_pay,
            "src_valid": src_valid,
            "raw_elev_patch": raw_elev_patch,
            "src_elev": src_elev,
            "y_dense": y_dense,
            "y_valid_dense": y_valid_dense,
        }

    def __getitem__(self, idx: int):
        # Get base sample
        base_sample = self._base[idx]
        x_patch = base_sample[0]

        row = self._base.samples.iloc[idx]
        day = pd.Timestamp(row["day"]).normalize()
        H = W = self._base.patch_size

        # Recover patch bounds (same logic as base dataset)
        r_center = int(self._base._center_rows[idx])
        c_center = int(self._base._center_cols[idx])
        r0 = int(np.clip(r_center - H // 2, 0, self._base._domain_H - H))
        c0 = int(np.clip(c_center - W // 2, 0, self._base._domain_W - W))

        return self._augment_with_da(base_sample, x_patch, day, r0, c0)


def collate_grid_da(batch: list[dict]) -> dict:
    """Collate GridDAPatchDataset samples, padding both station and source dims."""
    B = len(batch)

    # --- Grid input (fixed size) ---
    x = torch.stack([s["x_patch"] for s in batch])

    # --- Station supervision (variable length, pad to max) ---
    max_sta = max(s["sta_rows"].shape[0] for s in batch)
    n_targets = (
        batch[0]["sta_targets"].shape[1] if batch[0]["sta_targets"].ndim > 1 else 1
    )

    sta_rows = torch.zeros(B, max_sta, dtype=torch.long)
    sta_cols = torch.zeros(B, max_sta, dtype=torch.long)
    sta_targets = torch.zeros(B, max_sta, n_targets, dtype=torch.float32)
    sta_valid = torch.zeros(B, max_sta, n_targets, dtype=torch.bool)
    sta_holdout = torch.zeros(B, max_sta, dtype=torch.bool)
    sta_is_center = torch.zeros(B, max_sta, dtype=torch.bool)
    sta_is_source = torch.zeros(B, max_sta, dtype=torch.bool)
    sta_is_query = torch.zeros(B, max_sta, dtype=torch.bool)

    for i, s in enumerate(batch):
        n = s["sta_rows"].shape[0]
        sta_rows[i, :n] = s["sta_rows"]
        sta_cols[i, :n] = s["sta_cols"]
        sta_targets[i, :n] = s["sta_targets"]
        sta_valid[i, :n] = s["sta_valid"]
        sta_holdout[i, :n] = s["sta_holdout"]
        sta_is_center[i, :n] = s["sta_is_center"]
        sta_is_source[i, :n] = s["sta_is_source"]
        sta_is_query[i, :n] = s["sta_is_query"]

    # --- Source stations (variable length, pad to max) ---
    max_src = max(s["src_rows"].shape[0] for s in batch)
    src_ctx_dim = batch[0]["src_ctx"].shape[1] if batch[0]["src_ctx"].ndim > 1 else 0
    src_pay_dim = batch[0]["src_pay"].shape[1] if batch[0]["src_pay"].ndim > 1 else 0

    src_rows = torch.zeros(B, max(max_src, 1), dtype=torch.long)
    src_cols = torch.zeros(B, max(max_src, 1), dtype=torch.long)
    src_ctx = torch.zeros(B, max(max_src, 1), src_ctx_dim, dtype=torch.float32)
    src_pay = torch.zeros(B, max(max_src, 1), src_pay_dim, dtype=torch.float32)
    src_valid = torch.zeros(B, max(max_src, 1), dtype=torch.bool)
    src_elev = torch.zeros(B, max(max_src, 1), dtype=torch.float32)

    for i, s in enumerate(batch):
        n = s["src_rows"].shape[0]
        if n > 0:
            src_rows[i, :n] = s["src_rows"]
            src_cols[i, :n] = s["src_cols"]
            src_ctx[i, :n] = s["src_ctx"]
            src_pay[i, :n] = s["src_pay"]
            src_valid[i, :n] = s["src_valid"]
            src_elev[i, :n] = s["src_elev"]

    raw_elev_patch = torch.stack([s["raw_elev_patch"] for s in batch])

    # --- Optional dense teacher (handle mixed availability per sample) ---
    any_dense = any(s["y_dense"] is not None for s in batch)
    if any_dense:
        # Infer shape from the first available sample
        ref = next(s["y_dense"] for s in batch if s["y_dense"] is not None)
        n_t, pH, pW = ref.shape
        y_dense_list = []
        y_valid_list = []
        for s in batch:
            if s["y_dense"] is not None:
                y_dense_list.append(s["y_dense"])
                y_valid_list.append(s["y_valid_dense"])
            else:
                y_dense_list.append(torch.zeros(n_t, pH, pW))
                y_valid_list.append(torch.zeros(n_t, pH, pW, dtype=torch.bool))
        y_dense = torch.stack(y_dense_list)
        y_valid_dense = torch.stack(y_valid_list)
    else:
        y_dense = None
        y_valid_dense = None

    return {
        "x_patch": x,
        "sta_rows": sta_rows,
        "sta_cols": sta_cols,
        "sta_targets": sta_targets,
        "sta_valid": sta_valid,
        "sta_holdout": sta_holdout,
        "sta_is_center": sta_is_center,
        "sta_is_source": sta_is_source,
        "sta_is_query": sta_is_query,
        "src_rows": src_rows,
        "src_cols": src_cols,
        "src_ctx": src_ctx,
        "src_pay": src_pay,
        "src_valid": src_valid,
        "raw_elev_patch": raw_elev_patch,
        "src_elev": src_elev,
        "y_dense": y_dense,
        "y_valid_dense": y_valid_dense,
    }


def collate_grid_da_day_tile(batch: list[list[dict]]) -> dict:
    """Flatten day-centric batches into the standard Stage C collated format.

    Each item in *batch* is a list of K sample dicts (one day's worth of tiles).
    We flatten all day payloads into a single sample list and delegate to
    ``collate_grid_da``.
    """
    flat = []
    for day_samples in batch:
        flat.extend(day_samples)
    return collate_grid_da(flat)


class GridDADayTileBatchDataset(Dataset):
    """Day-centric tile batch dataset for Stage C DA training.

    One dataset sample = one day.  ``__getitem__`` loads that day's rasters
    once and returns *K* tile-patch dicts augmented with DA source tensors.

    Mirrors ``HRRRDayTileBatchDataset`` geometry but uses ``GridDAPatchDataset``
    DA augmentation for each tile.  Validation stays station-centered.

    ``set_epoch(epoch)`` varies the per-day tile selection each epoch.
    """

    def __init__(
        self,
        *,
        tile_stride: int | None = None,
        tiles_per_day: int | None = None,
        tile_sampling_seed: int = 42,
        train_query_frac: float = 0.0,
        train_min_query_source_dist_px: int = 0,
        train_source_dropout_prob: float = 0.0,
        payload_cols: list[str] | None = None,
        teacher_dir: str | None = None,
        teacher_pattern: str = "URMA_1km_{date}.tif",
        increment_map: dict[str, str] | None = None,
        **base_kwargs,
    ):
        self.train_query_frac = float(train_query_frac)
        self.train_min_query_source_dist_px = int(train_min_query_source_dist_px)
        self.train_source_dropout_prob = float(train_source_dropout_prob)

        # Build the DA-augmentation wrapper around an HRRRPatchDataset
        self._da = GridDAPatchDataset(
            payload_cols=payload_cols,
            teacher_dir=teacher_dir,
            teacher_pattern=teacher_pattern,
            increment_map=increment_map,
            **base_kwargs,
        )
        base = self._da._base  # the inner HRRRPatchDataset

        stride = tile_stride if tile_stride is not None else base.patch_size
        if stride <= 0:
            raise ValueError(f"tile_stride must be > 0, got {stride}")
        if stride > base.patch_size:
            raise ValueError(
                f"tile_stride must be <= patch_size ({base.patch_size}), got {stride}"
            )

        self.tile_stride = int(stride)
        self.tiles_per_day = tiles_per_day
        self.tile_sampling_seed = tile_sampling_seed

        # Build FULL tile index (no per-day cap) so set_epoch() can resample
        station_samples = base.samples
        self._tile_index = _load_or_build_tile_index(
            station_samples=station_samples,
            cache_dir=base.metadata_cache_dir,
            domain_h=base._domain_H,
            domain_w=base._domain_W,
            patch_size=base.patch_size,
            tile_stride=self.tile_stride,
            tiles_per_day=None,
            tile_sampling_seed=self.tile_sampling_seed,
        )
        self._tile_r0 = self._tile_index["tile_r0"].to_numpy(dtype="int32")
        self._tile_c0 = self._tile_index["tile_c0"].to_numpy(dtype="int32")
        self._tile_day_slices = _build_day_slices(self._tile_index["day"])

        # Rebuild samples as one-row-per-day for DayGroupedSampler compatibility
        day_list = sorted(self._tile_index["day"].unique())
        self.samples = pd.DataFrame({"day": day_list})

        # Shared int so DataLoader workers see epoch updates from the main
        # process callback (same pattern as HRRRDayTileBatchDataset).
        self._shared_epoch = mp.Value("i", 0)

    def set_epoch(self, epoch: int) -> None:
        """Update epoch so tile choices change across epochs."""
        self._shared_epoch.value = epoch

    # Delegate properties
    @property
    def in_channels(self):
        return self._da.in_channels

    @property
    def feature_names(self):
        return self._da.feature_names

    @property
    def target_names(self):
        return self._da.target_names

    @property
    def norm_stats(self):
        return self._da.norm_stats

    @property
    def source_pay_dim(self):
        return self._da.source_pay_dim

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> list[dict]:
        epoch = self._shared_epoch.value
        day = pd.Timestamp(self.samples.iloc[idx]["day"]).normalize()
        base = self._da._base

        bounds = self._tile_day_slices.get(day)
        if bounds is None:
            return []
        start, stop = bounds
        tile_r0 = self._tile_r0[start:stop]
        tile_c0 = self._tile_c0[start:stop]
        n_tiles = len(tile_r0)

        if self.tiles_per_day is not None and n_tiles > self.tiles_per_day:
            rng = np.random.default_rng(self.tile_sampling_seed + epoch * 100_000 + idx)
            chosen = np.sort(rng.choice(n_tiles, self.tiles_per_day, replace=False))
            tile_r0 = tile_r0[chosen]
            tile_c0 = tile_c0[chosen]
            n_tiles = self.tiles_per_day

        # Fetch day's neighbor data ONCE for all K tiles
        prefetched = base._neighbor_day_data(day)

        # Per-day RNG for reproducible source/query splits
        tile_rng = np.random.default_rng(
            self.tile_sampling_seed + epoch * 100_000 + idx + 7
        )

        samples = []
        for i in range(n_tiles):
            r0 = int(tile_r0[i])
            c0 = int(tile_c0[i])
            base_sample = base._build_patch_sample(
                day=day,
                r0=r0,
                c0=c0,
                prefetched_neighbors=prefetched,
            )
            x_patch = base_sample[0]
            da_dict = self._da._augment_with_da(
                base_sample,
                x_patch,
                day,
                r0,
                c0,
                prefetched_neighbors=prefetched,
                train_query_frac=self.train_query_frac,
                train_min_query_source_dist_px=self.train_min_query_source_dist_px,
                train_source_dropout_prob=self.train_source_dropout_prob,
                rng=tile_rng,
            )
            samples.append(da_dict)
        return samples
