"""
Train the grid backbone on dense URMA − HRRR increments (Stage A).

Usage:
    uv run python -m models.hrrr_da.train_dense_pretrain \
        --config models/hrrr_da/configs/grid_pretrain_tmax.toml
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np

import lightning as L
import pandas as pd
import tomli_w
import tomllib
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader

from models.hrrr_da.dense_increment_dataset import (
    DenseIncrementDataset,
    collate_dense_increment,
)
from models.hrrr_da.lit_dense_pretrain import LitDensePretraining
from models.hrrr_da.train_hrrr_hetero import DayGroupedSampler
from prep.paths import MVP_ROOT


class DenseTileResamplingCallback(L.Callback):
    """Calls set_epoch on both the sampler and the dataset each epoch."""

    def __init__(
        self,
        sampler: DayGroupedSampler,
        train_dataset: DenseIncrementDataset,
    ) -> None:
        self.sampler = sampler
        self.train_dataset = train_dataset

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        epoch = trainer.current_epoch
        self.sampler.set_epoch(epoch)
        self.train_dataset.set_epoch(epoch)


@dataclass
class DensePretrainConfig:
    name: str = "grid_pretrain_tmax"
    description: str = "Dense URMA−HRRR pretraining for grid backbone (Stage A)"

    hidden_dim: int = 32
    lr: float = 3e-4
    tv_weight: float = 1e-4
    huber_delta: float = 2.0
    batch_size: int = 32
    epochs: int = 50
    seed: int = 42
    tiles_per_day: int = 8
    days_per_epoch: int = 250
    val_frac: float = 0.1
    num_workers: int = 4
    patch_size: int = 64

    target_names: list[str] = field(default_factory=lambda: ["delta_tmax"])
    increment_map: dict[str, str] = field(
        default_factory=lambda: {"tmax_c": "tmax_hrrr"}
    )

    background_dir: str = f"{MVP_ROOT}/hrrr_1km_pnw"
    background_pattern: str = "HRRR_1km_{date}.tif"
    teacher_dir: str = f"{MVP_ROOT}/urma_1km_pnw"
    teacher_pattern: str = "URMA_1km_{date}.tif"
    static_tifs: list[str] = field(
        default_factory=lambda: [f"{MVP_ROOT}/terrain_pnw_1km.tif"]
    )
    landsat_tif: str = f"{MVP_ROOT}/landsat_pnw_1km.tif"
    rsun_tif: str | None = None
    cdr_dir: str | None = None
    cdr_pattern: str = "CDR_005deg_{date}.tif"
    out_dir: str = f"{MVP_ROOT}/grid_pretrain_tmax"
    train_years: list[int] = field(
        default_factory=lambda: [2018, 2019, 2020, 2021, 2022, 2023]
    )
    drop_bands: list[str] = field(default_factory=lambda: ["n_hours"])

    device: str | None = None

    def save_toml(self, path: str) -> None:
        d = asdict(self)
        d = {k: v for k, v in d.items() if v is not None}
        with open(path, "wb") as f:
            tomli_w.dump(d, f)

    @classmethod
    def from_toml(cls, path: str) -> "DensePretrainConfig":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(**data)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train dense pretrain grid backbone.")
    p.add_argument("--config", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> DensePretrainConfig:
    cfg = (
        DensePretrainConfig.from_toml(args.config)
        if args.config
        else DensePretrainConfig()
    )
    for cli, attr in [
        ("out_dir", "out_dir"),
        ("batch_size", "batch_size"),
        ("epochs", "epochs"),
        ("lr", "lr"),
        ("device", "device"),
        ("num_workers", "num_workers"),
    ]:
        val = getattr(args, cli, None)
        if val is not None:
            setattr(cfg, attr, val)
    return cfg


def main() -> None:
    args = _parse_args()
    cfg = _build_config(args)

    L.seed_everything(cfg.seed, workers=True)
    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg.save_toml(os.path.join(cfg.out_dir, "experiment.toml"))

    # --- Discover available HRRR days in train years ---
    all_bg = sorted(os.listdir(cfg.background_dir))
    train_days: set[pd.Timestamp] = set()
    for fn in all_bg:
        if not fn.endswith(".tif"):
            continue
        date_str = fn.replace(".tif", "").split("_")[-1]
        if len(date_str) != 8:
            continue
        try:
            day = pd.Timestamp(date_str)
        except ValueError:
            continue
        if day.year in cfg.train_years:
            train_days.add(day.normalize())

    # Split train days into train/val subsets for Stage A monitoring
    sorted_days = sorted(train_days)
    rng = np.random.default_rng(cfg.seed)
    n_val = max(1, int(len(sorted_days) * cfg.val_frac))
    val_indices = set(rng.choice(len(sorted_days), n_val, replace=False))
    stage_a_val_days = {sorted_days[i] for i in val_indices}
    stage_a_train_days = train_days - stage_a_val_days
    print(
        f"Stage A split: {len(stage_a_train_days)} train days, "
        f"{len(stage_a_val_days)} val days (from train years)"
    )

    train_ds = DenseIncrementDataset(
        background_dir=cfg.background_dir,
        background_pattern=cfg.background_pattern,
        teacher_dir=cfg.teacher_dir,
        teacher_pattern=cfg.teacher_pattern,
        static_tifs=cfg.static_tifs,
        landsat_tif=cfg.landsat_tif,
        rsun_tif=cfg.rsun_tif,
        cdr_dir=cfg.cdr_dir,
        cdr_pattern=cfg.cdr_pattern,
        train_days=stage_a_train_days,
        increment_map=cfg.increment_map,
        drop_bands=cfg.drop_bands or None,
        patch_size=cfg.patch_size,
        tiles_per_day=cfg.tiles_per_day,
        base_seed=cfg.seed,
    )
    in_channels = train_ds.in_channels
    print(f"Train samples: {len(train_ds)}, in_channels: {in_channels}")

    # Save norm stats (canonical for all stages)
    norm_stats_path = os.path.join(cfg.out_dir, "norm_stats.json")
    with open(norm_stats_path, "w") as f:
        json.dump(
            {
                "norm_stats": train_ds.norm_stats,
                "feature_names": train_ds.feature_names,
                "target_names": train_ds.target_names,
                "in_channels": train_ds.in_channels,
            },
            f,
            indent=2,
        )
    print(f"Norm stats saved: {norm_stats_path}")

    with open(os.path.join(cfg.out_dir, "feature_manifest.json"), "w") as f:
        json.dump(train_ds.feature_names, f, indent=2)

    val_ds = DenseIncrementDataset(
        background_dir=cfg.background_dir,
        background_pattern=cfg.background_pattern,
        teacher_dir=cfg.teacher_dir,
        teacher_pattern=cfg.teacher_pattern,
        static_tifs=cfg.static_tifs,
        landsat_tif=cfg.landsat_tif,
        rsun_tif=cfg.rsun_tif,
        cdr_dir=cfg.cdr_dir,
        cdr_pattern=cfg.cdr_pattern,
        train_days=stage_a_val_days,
        increment_map=cfg.increment_map,
        drop_bands=cfg.drop_bands or None,
        norm_stats=train_ds.norm_stats,
        patch_size=cfg.patch_size,
        tiles_per_day=cfg.tiles_per_day,
        base_seed=cfg.seed,
    )
    print(f"Val samples: {len(val_ds)}")

    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty.")
    if len(val_ds) == 0:
        raise ValueError("Validation dataset is empty.")

    day_sampler = DayGroupedSampler(
        train_ds.samples,
        days_per_epoch=cfg.days_per_epoch,
        base_seed=cfg.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=day_sampler,
        num_workers=cfg.num_workers,
        persistent_workers=(cfg.num_workers > 0),
        collate_fn=collate_dense_increment,
    )
    n_val_days = val_ds.samples["day"].nunique()
    val_sampler = DayGroupedSampler(
        val_ds.samples,
        days_per_epoch=n_val_days,
        base_seed=cfg.seed,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        persistent_workers=(cfg.num_workers > 0),
        collate_fn=collate_dense_increment,
    )

    model = LitDensePretraining(
        in_channels=in_channels,
        target_names=cfg.target_names,
        hidden_dim=cfg.hidden_dim,
        lr=cfg.lr,
        tv_weight=cfg.tv_weight,
        huber_delta=cfg.huber_delta,
    )

    if cfg.device and cfg.device.startswith("cuda"):
        accelerator = "gpu"
        devices = [int(cfg.device.split(":")[1])] if ":" in cfg.device else [0]
    elif cfg.device == "cpu":
        accelerator = "cpu"
        devices = 1
    else:
        accelerator = "auto"
        devices = 1

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.out_dir,
            monitor="val_dense_loss",
            save_top_k=3,
            mode="min",
            filename="ckpt-{epoch:03d}-{val_dense_loss:.4f}",
        ),
        EarlyStopping(monitor="val_dense_loss", patience=20, mode="min", strict=False),
        LearningRateMonitor(logging_interval="epoch"),
        DenseTileResamplingCallback(day_sampler, train_ds),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed" if accelerator == "gpu" else 32,
        callbacks=callbacks,
        default_root_dir=cfg.out_dir,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)

    # Symlink best checkpoint to a stable name for downstream configs
    ckpt_cb = [c for c in callbacks if isinstance(c, ModelCheckpoint)][0]
    best = ckpt_cb.best_model_path
    if best:
        link = os.path.join(cfg.out_dir, "best.ckpt")
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.basename(best), link)
        print(f"Best checkpoint symlinked: {link} -> {os.path.basename(best)}")

    metrics = {
        k: v.item() if hasattr(v, "item") else v
        for k, v in trainer.callback_metrics.items()
    }
    registry = os.path.join(cfg.out_dir, "experiment_registry.jsonl")
    entry = {
        "timestamp": datetime.now().isoformat(),
        "config": asdict(cfg),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    with open(registry, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"Experiment logged: {registry}")


if __name__ == "__main__":
    main()
