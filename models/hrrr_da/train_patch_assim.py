"""
Train the E0 patch assimilation model.

Raster-patch HRRR bias correction supervised at all in-patch station locations.
No observation channels (E0 baseline).
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

from models.hrrr_da.lit_patch_assim import LitPatchAssim
from models.hrrr_da.patch_assim_dataset import HRRRPatchDataset, collate_patch
from models.hrrr_da.train_hrrr_hetero import DayGroupedSampler, DayResamplingCallback
from prep.paths import MVP_ROOT


@dataclass
class PatchAssimConfig:
    name: str = "patch_assim_e0"
    description: str = "Raster-patch HRRR bias correction (E0, no obs channels)"

    hidden_dim: int = 32
    lr: float = 3e-4
    tv_weight: float = 1e-3
    huber_delta: float = 1.0
    batch_size: int = 16
    epochs: int = 30
    seed: int = 42
    days_per_epoch: int = 250
    val_days_per_epoch: int | None = None
    num_workers: int = 4
    patch_size: int = 64

    target_names: list[str] = field(
        default_factory=lambda: [
            "delta_tmax",
            "delta_tmin",
            "delta_ea",
            "delta_rsds",
            "delta_w_par",
            "delta_w_perp",
        ]
    )

    table_path: str = f"{MVP_ROOT}/station_day_hrrr_pnw.parquet"
    background_dir: str = f"{MVP_ROOT}/hrrr_1km_pnw"
    background_pattern: str = "HRRR_1km_{date}.tif"
    static_tifs: list[str] = field(
        default_factory=lambda: [f"{MVP_ROOT}/terrain_pnw_1km.tif"]
    )
    landsat_tif: str = f"{MVP_ROOT}/landsat_pnw_1km.tif"
    rsun_tif: str | None = None
    cdr_dir: str | None = None
    cdr_pattern: str = "CDR_005deg_{date}.tif"
    out_dir: str = f"{MVP_ROOT}/patch_assim_e0"
    train_years: list[int] = field(
        default_factory=lambda: [2018, 2019, 2020, 2021, 2022, 2023]
    )
    val_years: list[int] = field(default_factory=lambda: [2024])
    holdout_fids_json: str | None = "artifacts/canonical_holdout_fids.json"
    benchmark_mode: bool = False
    drop_bands: list[str] = field(default_factory=list)

    pretrained_ckpt: str | None = None
    norm_stats_json: str | None = None

    device: str | None = None

    def save_toml(self, path: str) -> None:
        d = asdict(self)
        d = {k: v for k, v in d.items() if v is not None}
        with open(path, "wb") as f:
            tomli_w.dump(d, f)

    @classmethod
    def from_toml(cls, path: str) -> "PatchAssimConfig":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(**data)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train E0 patch assimilation model.")
    p.add_argument("--config", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> PatchAssimConfig:
    cfg = PatchAssimConfig.from_toml(args.config) if args.config else PatchAssimConfig()
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

    df = pd.read_parquet(cfg.table_path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["day"] = pd.to_datetime(df["day"])

    train_days = set(df[df["day"].dt.year.isin(cfg.train_years)]["day"].unique())
    val_days = set(df[df["day"].dt.year.isin(cfg.val_years)]["day"].unique())
    print(f"Train days: {len(train_days)}, Val days: {len(val_days)}")

    all_fids = set(df["fid"].astype(str).unique())
    holdout_fids: set[str] = set()
    if cfg.holdout_fids_json and os.path.exists(cfg.holdout_fids_json):
        with open(cfg.holdout_fids_json) as f:
            holdout_fids = set(str(x) for x in json.load(f))
        effective_holdout = holdout_fids & all_fids
        # Val-day holdout: fids present on val days
        val_df = df[df["day"].dt.year.isin(cfg.val_years)]
        val_holdout = effective_holdout & set(val_df["fid"].astype(str).unique())
        print(
            f"Holdout: {len(holdout_fids)} requested, "
            f"{len(effective_holdout)} in data, "
            f"{len(val_holdout)} on val days"
        )
    else:
        effective_holdout = set()
        val_holdout = set()
    with open(os.path.join(cfg.out_dir, "holdout_fids.json"), "w") as f:
        json.dump(sorted(holdout_fids), f)
    with open(os.path.join(cfg.out_dir, "effective_holdout_fids.json"), "w") as f:
        json.dump(sorted(effective_holdout), f)
    with open(os.path.join(cfg.out_dir, "val_holdout_fids.json"), "w") as f:
        json.dump(sorted(val_holdout), f)

    # Load canonical norm stats from Stage A if provided
    provided_norm_stats: dict | None = None
    if cfg.norm_stats_json:
        if not os.path.exists(cfg.norm_stats_json):
            raise FileNotFoundError(
                f"norm_stats_json does not exist: {cfg.norm_stats_json}"
            )
        with open(cfg.norm_stats_json) as f:
            provided_norm_stats = json.load(f).get("norm_stats")
        print(f"Loaded norm stats from {cfg.norm_stats_json}")

    train_ds = HRRRPatchDataset(
        table_path=cfg.table_path,
        background_dir=cfg.background_dir,
        background_pattern=cfg.background_pattern,
        static_tifs=cfg.static_tifs,
        landsat_tif=cfg.landsat_tif,
        target_names=cfg.target_names,
        train_days=train_days,
        target_exclude_fids=holdout_fids or None,
        supervision_exclude_fids=holdout_fids or None,
        rsun_tif=cfg.rsun_tif,
        cdr_dir=cfg.cdr_dir,
        cdr_pattern=cfg.cdr_pattern,
        holdout_fids=holdout_fids or None,
        drop_bands=cfg.drop_bands or None,
        norm_stats=provided_norm_stats,
        patch_size=cfg.patch_size,
    )
    in_channels = train_ds.in_channels
    print(f"Train samples: {len(train_ds)}, in_channels: {in_channels}")

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

    # Run-level metadata (replaces chip artifact manifests)
    with open(os.path.join(cfg.out_dir, "feature_manifest.json"), "w") as f:
        json.dump(train_ds.feature_names, f, indent=2)
    with open(os.path.join(cfg.out_dir, "target_manifest.json"), "w") as f:
        json.dump(cfg.target_names, f, indent=2)
    with open(os.path.join(cfg.out_dir, "tile_manifest.json"), "w") as f:
        json.dump(
            {
                "tile_size": cfg.patch_size,
                "stride": "station-centered (one tile per station-day)",
                "overlap": "variable (tiles centered on different stations may overlap)",
            },
            f,
            indent=2,
        )
    with open(os.path.join(cfg.out_dir, "query_sampling_manifest.json"), "w") as f:
        json.dump(
            {
                "method": "all in-patch stations supervised per tile",
                "scoring": "center-station-only for val/target_mae (deduplicated)",
                "holdout_gating": "sta_holdout mask per station",
            },
            f,
            indent=2,
        )
    with open(os.path.join(cfg.out_dir, "split_pointer.json"), "w") as f:
        json.dump(
            {
                "family": "grid-core-v0",
                "protocol": "benchmark" if cfg.benchmark_mode else "explore",
                "holdout_fids_json": cfg.holdout_fids_json,
                "train_years": cfg.train_years,
                "val_years": cfg.val_years,
            },
            f,
            indent=2,
        )

    # Subsample val days deterministically for consistent scoring each epoch
    all_val_days = sorted(val_days)
    if cfg.val_days_per_epoch and cfg.val_days_per_epoch < len(all_val_days):
        rng = np.random.default_rng(cfg.seed)
        chosen = rng.choice(len(all_val_days), cfg.val_days_per_epoch, replace=False)
        fixed_val_days = {all_val_days[i] for i in sorted(chosen)}
        print(
            f"Val day subsample: {len(fixed_val_days)} of {len(all_val_days)} "
            f"(deterministic, seed={cfg.seed})"
        )
    else:
        fixed_val_days = val_days

    val_ds = HRRRPatchDataset(
        table_path=cfg.table_path,
        background_dir=cfg.background_dir,
        background_pattern=cfg.background_pattern,
        static_tifs=cfg.static_tifs,
        landsat_tif=cfg.landsat_tif,
        target_names=cfg.target_names,
        train_days=fixed_val_days,
        target_include_fids=holdout_fids or None,
        rsun_tif=cfg.rsun_tif,
        cdr_dir=cfg.cdr_dir,
        cdr_pattern=cfg.cdr_pattern,
        holdout_fids=holdout_fids or None,
        drop_bands=cfg.drop_bands or None,
        norm_stats=train_ds.norm_stats,
        patch_size=cfg.patch_size,
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
        collate_fn=collate_patch,
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
        collate_fn=collate_patch,
    )

    model = LitPatchAssim(
        in_channels=in_channels,
        target_names=cfg.target_names,
        hidden_dim=cfg.hidden_dim,
        lr=cfg.lr,
        tv_weight=cfg.tv_weight,
        huber_delta=cfg.huber_delta,
        benchmark_mode=cfg.benchmark_mode,
    )

    if cfg.pretrained_ckpt:
        import torch

        if not os.path.exists(cfg.pretrained_ckpt):
            raise FileNotFoundError(
                f"pretrained_ckpt does not exist: {cfg.pretrained_ckpt}"
            )
        sd = torch.load(cfg.pretrained_ckpt, map_location="cpu")["state_dict"]
        backbone_sd = {
            k: v for k, v in sd.items() if "out." not in k and "heads." not in k
        }
        missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
        print(
            f"Loaded pretrained backbone from {cfg.pretrained_ckpt}\n"
            f"  missing keys (expected — new heads): {len(missing)}\n"
            f"  unexpected keys: {len(unexpected)}"
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

    ckpt_metric = "val/target_mae" if cfg.benchmark_mode else "val_loss"
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.out_dir,
            monitor=ckpt_metric,
            save_top_k=3,
            mode="min",
            filename="ckpt-{epoch:03d}-{" + ckpt_metric + ":.4f}",
        ),
        EarlyStopping(monitor=ckpt_metric, patience=20, mode="min", strict=False),
        LearningRateMonitor(logging_interval="epoch"),
        DayResamplingCallback(day_sampler),
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
