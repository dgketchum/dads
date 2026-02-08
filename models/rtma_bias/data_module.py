from __future__ import annotations

import lightning as L
import torch
from torch.utils.data import DataLoader, Subset

from models.rtma_bias.patch_dataset import PatchDatasetConfig, RtmaHumidityPatchDataset


def _collate(batch):
    xb, yb, meta = zip(*batch)
    return torch.stack(list(xb), dim=0), torch.stack(list(yb), dim=0), list(meta)


class RtmaPatchDataModule(L.LightningDataModule):
    def __init__(
        self,
        patch_index: str,
        tif_root: str,
        patch_size: int = 64,
        batch_size: int = 64,
        num_workers: int = 2,
        val_frac: float = 0.2,
        seed: int = 42,
        preload: bool = True,
        terrain_tif: str | None = None,
        rsun_tif: str | None = None,
        landsat_tif: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.patch_index = patch_index
        self.tif_root = tif_root
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_frac = val_frac
        self.seed = seed
        self.preload = preload
        self.terrain_tif = terrain_tif
        self.rsun_tif = rsun_tif
        self.landsat_tif = landsat_tif

        self._in_channels: int | None = None
        self.train_ds: Subset | None = None
        self.val_ds: Subset | None = None

    @property
    def in_channels(self) -> int:
        if self._in_channels is None:
            raise RuntimeError("Call setup() before accessing in_channels")
        return self._in_channels

    def setup(self, stage: str | None = None):
        if self.train_ds is not None:
            return
        cfg = PatchDatasetConfig(
            tif_root=self.tif_root,
            patch_size=self.patch_size,
            preload=self.preload,
            terrain_tif=self.terrain_tif,
            rsun_tif=self.rsun_tif,
            landsat_tif=self.landsat_tif,
        )
        full_ds = RtmaHumidityPatchDataset(self.patch_index, cfg)
        self._in_channels = full_ds.in_channels

        n = len(full_ds)
        gen = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(n, generator=gen).tolist()
        n_val = int(n * self.val_frac)
        self.val_ds = Subset(full_ds, indices[:n_val])
        self.train_ds = Subset(full_ds, indices[n_val:])

    def save_norm_stats(self, path: str) -> None:
        """Delegate to the underlying full dataset."""
        ds = self.train_ds.dataset if self.train_ds is not None else None
        if ds is None:
            raise RuntimeError("Call setup() before save_norm_stats()")
        ds.save_norm_stats(path)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=_collate,
        )
