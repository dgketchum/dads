from __future__ import annotations

import pandas as pd
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from models.rtma_bias.patch_dataset import PatchDatasetConfig, RtmaHumidityPatchDataset


def _collate(batch):
    xb, yb, meta = zip(*batch)
    return torch.stack(list(xb), dim=0), torch.stack(list(yb), dim=0), list(meta)


def _pair_collate(batch):
    xi, yi, meta_i, xj, yj, meta_j, pair_meta = zip(*batch)
    return (
        torch.stack(list(xi), dim=0),
        torch.stack(list(yi), dim=0),
        list(meta_i),
        torch.stack(list(xj), dim=0),
        torch.stack(list(yj), dim=0),
        list(meta_j),
        list(pair_meta),
    )


class RtmaPairDataset(Dataset):
    """Dataset over precomputed station-day pairs backed by a point dataset."""

    def __init__(self, base_ds: RtmaHumidityPatchDataset, pair_df: pd.DataFrame):
        self.base_ds = base_ds
        self.pairs = pair_df.reset_index(drop=True)

    def __len__(self) -> int:
        return int(len(self.pairs))

    def __getitem__(self, idx: int):
        r = self.pairs.iloc[int(idx)]
        idx_i = int(r["idx_i"])
        idx_j = int(r["idx_j"])

        xi, yi, meta_i = self.base_ds[idx_i]
        xj, yj, meta_j = self.base_ds[idx_j]

        pair_meta = {
            "fid_i": str(r.get("fid_i", "")),
            "fid_j": str(r.get("fid_j", "")),
        }
        if "day" in r and pd.notna(r["day"]):
            pair_meta["day"] = pd.Timestamp(r["day"])
        if "dist_km" in r and pd.notna(r["dist_km"]):
            pair_meta["dist_km"] = float(r["dist_km"])

        return xi, yi, meta_i, xj, yj, meta_j, pair_meta


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
        target_col: str = "delta_log_ea",
        target_cols: tuple[str, ...] | None = None,
        rtma_channels: tuple[str, ...] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        decoded: bool = False,
        val_mgrs_tiles: tuple[str, ...] | None = None,
        use_pairwise_loss: bool = False,
        pair_index: str | None = None,
        pair_dmax_km: float | None = None,
        pair_sample_per_batch: int | None = None,
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
        self.target_col = target_col
        self.target_cols = target_cols
        self.rtma_channels = rtma_channels
        self.start_date = start_date
        self.end_date = end_date
        self.decoded = decoded
        self.val_mgrs_tiles = val_mgrs_tiles
        self.use_pairwise_loss = bool(use_pairwise_loss)
        self.pair_index = pair_index
        self.pair_dmax_km = pair_dmax_km
        self.pair_sample_per_batch = pair_sample_per_batch

        self._in_channels: int | None = None
        self.train_ds: Subset | None = None
        self.val_ds: Subset | None = None
        self.train_pair_ds: RtmaPairDataset | None = None
        self.val_pair_ds: RtmaPairDataset | None = None

    @property
    def in_channels(self) -> int:
        if self._in_channels is None:
            raise RuntimeError("Call setup() before accessing in_channels")
        return self._in_channels

    def _setup_pair_datasets(
        self,
        full_ds: RtmaHumidityPatchDataset,
        train_indices: list[int],
        val_indices: list[int],
    ) -> None:
        if not self.use_pairwise_loss or not self.pair_index:
            return

        pair_df = pd.read_parquet(self.pair_index)
        required = {"fid_i", "fid_j", "day"}
        missing = required - set(pair_df.columns)
        if missing:
            raise ValueError(f"pair_index missing required columns: {sorted(missing)}")

        pair_df = pair_df.copy()
        pair_df["fid_i"] = pair_df["fid_i"].astype(str)
        pair_df["fid_j"] = pair_df["fid_j"].astype(str)
        pair_df["day"] = pd.to_datetime(pair_df["day"], errors="coerce").dt.normalize()
        pair_df = pair_df.dropna(subset=["day"])

        if self.pair_dmax_km is not None and "dist_km" in pair_df.columns:
            pair_df = pair_df.loc[
                pd.to_numeric(pair_df["dist_km"], errors="coerce")
                <= float(self.pair_dmax_km)
            ].copy()

        base_df = full_ds.df.copy()
        base_df["fid"] = base_df["fid"].astype(str)
        base_df["day"] = pd.to_datetime(base_df["day"], errors="coerce").dt.normalize()
        base_df = base_df.reset_index(drop=False).rename(columns={"index": "ds_idx"})
        key_df = base_df[["fid", "day", "ds_idx"]].drop_duplicates(
            subset=["fid", "day"], keep="first"
        )
        key_df["k"] = key_df["fid"] + "|" + key_df["day"].dt.strftime("%Y-%m-%d")
        key_to_idx = key_df.set_index("k")["ds_idx"]

        pair_df["k_i"] = pair_df["fid_i"] + "|" + pair_df["day"].dt.strftime("%Y-%m-%d")
        pair_df["k_j"] = pair_df["fid_j"] + "|" + pair_df["day"].dt.strftime("%Y-%m-%d")
        pair_df["idx_i"] = pair_df["k_i"].map(key_to_idx)
        pair_df["idx_j"] = pair_df["k_j"].map(key_to_idx)
        pair_df = pair_df.dropna(subset=["idx_i", "idx_j"]).copy()
        pair_df["idx_i"] = pair_df["idx_i"].astype("int64")
        pair_df["idx_j"] = pair_df["idx_j"].astype("int64")

        train_set = set(int(i) for i in train_indices)
        val_set = set(int(i) for i in val_indices)

        train_mask = pair_df["idx_i"].isin(train_set) & pair_df["idx_j"].isin(train_set)
        val_mask = pair_df["idx_i"].isin(val_set) & pair_df["idx_j"].isin(val_set)

        train_pairs = pair_df.loc[train_mask].copy()
        val_pairs = pair_df.loc[val_mask].copy()
        cross_count = int(len(pair_df) - len(train_pairs) - len(val_pairs))

        if len(train_pairs) > 0:
            self.train_pair_ds = RtmaPairDataset(full_ds, train_pairs)
        if len(val_pairs) > 0:
            self.val_pair_ds = RtmaPairDataset(full_ds, val_pairs)

        print(
            f"  pair index: {len(pair_df):,} usable pairs, "
            f"{len(train_pairs):,} train / {len(val_pairs):,} val "
            f"(dropped cross/missing: {cross_count:,})",
            flush=True,
        )

    def setup(self, stage: str | None = None):
        if self.train_ds is not None:
            return
        extra = {}
        if self.rtma_channels is not None:
            extra["rtma_channels"] = self.rtma_channels
        if self.target_cols is not None:
            extra["target_cols"] = self.target_cols
        cfg = PatchDatasetConfig(
            tif_root=self.tif_root,
            patch_size=self.patch_size,
            preload=self.preload,
            terrain_tif=self.terrain_tif,
            rsun_tif=self.rsun_tif,
            landsat_tif=self.landsat_tif,
            target_col=self.target_col,
            start_date=self.start_date,
            end_date=self.end_date,
            decoded=self.decoded,
            **extra,
        )
        full_ds = RtmaHumidityPatchDataset(self.patch_index, cfg)
        self._in_channels = full_ds.in_channels

        n = len(full_ds)
        if self.val_mgrs_tiles is not None:
            df = full_ds.df
            tile_set = set(self.val_mgrs_tiles)
            val_mask = df["MGRS_TILE"].isin(tile_set)
            val_indices = df.index[val_mask].tolist()
            train_indices = df.index[~val_mask].tolist()
            n_val_sta = df.loc[val_mask, "fid"].nunique()
            n_train_sta = df.loc[~val_mask, "fid"].nunique()
            print(
                f"  spatial holdout: {len(train_indices):,} train / "
                f"{len(val_indices):,} val samples "
                f"({n_train_sta:,} / {n_val_sta:,} stations, "
                f"{len(tile_set)} val tiles)",
                flush=True,
            )
        else:
            gen = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(n, generator=gen).tolist()
            n_val = int(n * self.val_frac)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
        self.val_ds = Subset(full_ds, val_indices)
        self.train_ds = Subset(full_ds, train_indices)

        self._setup_pair_datasets(full_ds, train_indices, val_indices)

    def save_norm_stats(self, path: str) -> None:
        """Delegate to the underlying full dataset."""
        ds = self.train_ds.dataset if self.train_ds is not None else None
        if ds is None:
            raise RuntimeError("Call setup() before save_norm_stats()")
        ds.save_norm_stats(path)

    def _loader_kwargs(self, collate_fn=_collate) -> dict:
        kw = {
            "num_workers": self.num_workers,
            "pin_memory": True,
            "collate_fn": collate_fn,
        }
        if self.num_workers > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = 4
        return kw

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            **self._loader_kwargs(),
        )

    def train_pair_dataloader(self):
        if self.train_pair_ds is None:
            return None
        pair_bs = int(self.pair_sample_per_batch or self.batch_size)
        return DataLoader(
            self.train_pair_ds,
            batch_size=pair_bs,
            shuffle=True,
            **self._loader_kwargs(collate_fn=_pair_collate),
        )

    def val_pair_dataloader(self):
        if self.val_pair_ds is None:
            return None
        pair_bs = int(self.pair_sample_per_batch or self.batch_size)
        return DataLoader(
            self.val_pair_ds,
            batch_size=pair_bs,
            shuffle=False,
            **self._loader_kwargs(collate_fn=_pair_collate),
        )
