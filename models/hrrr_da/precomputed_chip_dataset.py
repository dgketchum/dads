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

        # Build index from chip_index.json (fast) or by loading files (slow fallback)
        self._files: list[Path] = list(pt_files)
        self._offsets: list[tuple[int, int]] = []
        self._sample_days: list[pd.Timestamp] = []

        index_path = os.path.join(chip_dir, "chip_index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                chip_index = json.load(f)
            for fi, fp in enumerate(self._files):
                n = chip_index.get(fp.stem, 0)
                day_ts = pd.Timestamp(fp.stem)
                for si in range(n):
                    self._offsets.append((fi, si))
                    self._sample_days.append(day_ts)
        else:
            for fi, fp in enumerate(self._files):
                data = torch.load(fp, weights_only=False)
                n = data["x"].shape[0]
                day_ts = pd.Timestamp(fp.stem)
                for si in range(n):
                    self._offsets.append((fi, si))
                    self._sample_days.append(day_ts)

        # Expose a samples-like DataFrame for DayGroupedSampler compatibility
        self.samples = pd.DataFrame({"day": self._sample_days})

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
