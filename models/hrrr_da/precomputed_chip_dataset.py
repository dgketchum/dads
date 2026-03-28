"""
Dataset for precomputed grid-core-v0 chips.

Loads per-day .pt files from a chip directory. Each file contains all
station-centered patches for one day, pre-extracted and normalized.
No raster I/O at training time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class PrecomputedChipDataset(Dataset):
    """Load precomputed grid chips from per-day .pt files.

    Each .pt file contains a dict with tensors for all station patches on that
    day. This dataset flattens across days to yield individual (patch, stations)
    samples.

    Parameters
    ----------
    chip_dir : str
        Directory with per-day .pt files and meta.json.
    train_days : set | None
        Filter to these days only.
    """

    def __init__(
        self,
        chip_dir: str,
        train_days: set | None = None,
    ):
        super().__init__()

        meta_path = os.path.join(chip_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No meta.json in {chip_dir}")
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.feature_names = self.meta["feature_names"]
        self.target_names = self.meta["target_names"]
        self.in_channels = self.meta["in_channels"]

        # Load per-day files
        pt_files = sorted(Path(chip_dir).glob("*.pt"))
        if train_days is not None:
            day_strs = {pd.Timestamp(d).strftime("%Y-%m-%d") for d in train_days}
            pt_files = [p for p in pt_files if p.stem in day_strs]

        # Build a flat index: (file_idx, sample_idx_within_day)
        self._files: list[Path] = []
        self._offsets: list[tuple[int, int]] = []  # (file_idx, local_idx)
        for fi, fp in enumerate(pt_files):
            data = torch.load(fp, weights_only=False)
            n = data["x"].shape[0]
            self._files.append(fp)
            for si in range(n):
                self._offsets.append((fi, si))

        # Cache: only one day loaded at a time
        self._cached_fi: int = -1
        self._cached_data: dict | None = None

    def __len__(self) -> int:
        return len(self._offsets)

    def _load_day(self, fi: int) -> dict:
        if fi != self._cached_fi:
            self._cached_data = torch.load(self._files[fi], weights_only=False)
            self._cached_fi = fi
        return self._cached_data

    def __getitem__(self, idx: int):
        fi, si = self._offsets[idx]
        data = self._load_day(fi)
        return (
            data["x"][si],
            data["sta_rows"][si],
            data["sta_cols"][si],
            data["sta_targets"][si],
            data["sta_valid"][si],
            data["sta_holdout"][si],
            data["sta_is_center"][si],
        )
