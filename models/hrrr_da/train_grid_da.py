"""
Train the grid backbone with DA fusion (Stage C).

Usage:
    uv run python -m models.hrrr_da.train_grid_da \
        --config models/hrrr_da/configs/grid_da_s_tmax.toml
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
import torch
import tomli_w
import tomllib
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader

from models.hrrr_da.grid_da_dataset import GridDAPatchDataset, collate_grid_da
from models.hrrr_da.lit_grid_da import LitGridDA
from models.hrrr_da.train_hrrr_hetero import DayGroupedSampler, DayResamplingCallback
from prep.paths import MVP_ROOT


@dataclass
class GridDAConfig:
    name: str = "grid_da_s_tmax"
    description: str = "Grid backbone + DA fusion, single-head tmax (Stage C)"

    hidden_dim: int = 32
    lr: float = 1e-4
    tv_weight: float = 1e-3
    huber_delta: float = 1.0
    dense_loss_weight: float = 0.0
    da_enabled: bool = True
    support_radius_px: int = 16
    da_gate_init_bias: float = -2.0
    batch_size: int = 16
    epochs: int = 100
    seed: int = 42
    days_per_epoch: int = 250
    val_days_per_epoch: int | None = None
    num_workers: int = 4
    patch_size: int = 64
    benchmark_mode: bool = True

    target_names: list[str] = field(default_factory=lambda: ["delta_tmax"])
    payload_cols: list[str] = field(
        default_factory=lambda: ["delta_tmax", "delta_tmin"]
    )

    # Optional dense teacher
    teacher_dir: str | None = None
    teacher_pattern: str = "URMA_1km_{date}.tif"
    increment_map: dict[str, str] = field(default_factory=dict)

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
    out_dir: str = f"{MVP_ROOT}/grid_da_s_tmax"
    train_years: list[int] = field(
        default_factory=lambda: [2018, 2019, 2020, 2021, 2022, 2023]
    )
    val_years: list[int] = field(default_factory=lambda: [2024])
    holdout_fids_json: str | None = "artifacts/canonical_holdout_fids.json"
    drop_bands: list[str] = field(default_factory=lambda: ["n_hours"])

    # Stage A/B artifacts
    pretrained_ckpt: str | None = None
    norm_stats_json: str | None = None

    device: str | None = None

    def save_toml(self, path: str) -> None:
        d = asdict(self)
        d = {k: v for k, v in d.items() if v is not None}
        with open(path, "wb") as f:
            tomli_w.dump(d, f)

    @classmethod
    def from_toml(cls, path: str) -> "GridDAConfig":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(**data)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train grid DA model (Stage C).")
    p.add_argument("--config", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> GridDAConfig:
    cfg = GridDAConfig.from_toml(args.config) if args.config else GridDAConfig()
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


def _load_stage_b_weights(model: LitGridDA, ckpt_path: str) -> None:
    """Load Stage B checkpoint into LitGridDA.

    Maps:
      Stage B  model.down1.* → backbone.down1.*
      Stage B  model.out.*   → bg_head.*
      Stage B  model.heads.0.* → bg_head.*
    """
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]

    backbone_sd = {}
    bg_head_sd = {}
    for k, v in sd.items():
        # Stage B stores UNet as model.model.* (LitPatchAssim.model = UNetSmall)
        if k.startswith("model.model."):
            inner_key = k[len("model.model.") :]
            if inner_key.startswith("out."):
                # Single-head output → bg_head
                bg_key = inner_key[len("out.") :]
                bg_head_sd[f"bg_head.{bg_key}"] = v
            elif inner_key.startswith("heads.0."):
                # Multi-head → bg_head (first head)
                bg_key = inner_key[len("heads.0.") :]
                bg_head_sd[f"bg_head.{bg_key}"] = v
            else:
                backbone_sd[f"backbone.{inner_key}"] = v
        elif k.startswith("model."):
            # Direct model.down1.* keys (if model IS the UNet)
            inner_key = k[len("model.") :]
            if inner_key.startswith("out."):
                bg_key = inner_key[len("out.") :]
                bg_head_sd[f"bg_head.{bg_key}"] = v
            elif inner_key.startswith("heads.0."):
                bg_key = inner_key[len("heads.0.") :]
                bg_head_sd[f"bg_head.{bg_key}"] = v
            else:
                backbone_sd[f"backbone.{inner_key}"] = v

    combined = {**backbone_sd, **bg_head_sd}
    missing, unexpected = model.load_state_dict(combined, strict=False)
    print(
        f"Loaded Stage B weights from {ckpt_path}\n"
        f"  backbone keys loaded: {len(backbone_sd)}\n"
        f"  bg_head keys loaded: {len(bg_head_sd)}\n"
        f"  missing (DA components, expected): {len(missing)}\n"
        f"  unexpected: {len(unexpected)}"
    )


def main() -> None:
    args = _parse_args()
    cfg = _build_config(args)

    L.seed_everything(cfg.seed, workers=True)
    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg.save_toml(os.path.join(cfg.out_dir, "experiment.toml"))

    # Load norm stats
    provided_norm_stats: dict | None = None
    if cfg.norm_stats_json:
        if not os.path.exists(cfg.norm_stats_json):
            raise FileNotFoundError(
                f"norm_stats_json does not exist: {cfg.norm_stats_json}"
            )
        with open(cfg.norm_stats_json) as f:
            provided_norm_stats = json.load(f).get("norm_stats")
        print(f"Loaded norm stats from {cfg.norm_stats_json}")

    # Station table
    df = pd.read_parquet(cfg.table_path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["day"] = pd.to_datetime(df["day"])

    train_days = set(df[df["day"].dt.year.isin(cfg.train_years)]["day"].unique())
    val_days = set(df[df["day"].dt.year.isin(cfg.val_years)]["day"].unique())
    print(f"Train days: {len(train_days)}, Val days: {len(val_days)}")

    # Holdout
    all_fids = set(df["fid"].astype(str).unique())
    holdout_fids: set[str] = set()
    if cfg.holdout_fids_json and os.path.exists(cfg.holdout_fids_json):
        with open(cfg.holdout_fids_json) as f:
            holdout_fids = set(str(x) for x in json.load(f))
        print(
            f"Holdout: {len(holdout_fids)} requested, {len(holdout_fids & all_fids)} in data"
        )
    with open(os.path.join(cfg.out_dir, "holdout_fids.json"), "w") as f:
        json.dump(sorted(holdout_fids), f)

    # Build datasets
    base_kwargs = dict(
        table_path=cfg.table_path,
        background_dir=cfg.background_dir,
        background_pattern=cfg.background_pattern,
        static_tifs=cfg.static_tifs,
        landsat_tif=cfg.landsat_tif,
        target_names=cfg.target_names,
        rsun_tif=cfg.rsun_tif,
        cdr_dir=cfg.cdr_dir,
        cdr_pattern=cfg.cdr_pattern,
        holdout_fids=holdout_fids or None,
        drop_bands=cfg.drop_bands or None,
        norm_stats=provided_norm_stats,
        patch_size=cfg.patch_size,
    )

    train_ds = GridDAPatchDataset(
        payload_cols=cfg.payload_cols,
        teacher_dir=cfg.teacher_dir,
        teacher_pattern=cfg.teacher_pattern,
        increment_map=cfg.increment_map or None,
        train_days=train_days,
        target_exclude_fids=holdout_fids or None,
        supervision_exclude_fids=holdout_fids or None,
        **base_kwargs,
    )
    in_channels = train_ds.in_channels
    source_ctx_dim = in_channels  # context extracted from x_patch
    source_pay_dim = train_ds.source_pay_dim
    print(
        f"Train samples: {len(train_ds)}, in_channels: {in_channels}, pay_dim: {source_pay_dim}"
    )

    # Save manifests
    with open(os.path.join(cfg.out_dir, "norm_stats.json"), "w") as f:
        json.dump(
            {"norm_stats": train_ds.norm_stats, "in_channels": in_channels},
            f,
            indent=2,
        )
    with open(os.path.join(cfg.out_dir, "feature_manifest.json"), "w") as f:
        json.dump(train_ds.feature_names, f, indent=2)
    with open(os.path.join(cfg.out_dir, "split_pointer.json"), "w") as f:
        json.dump(
            {
                "family": "grid-backbone",
                "stage": "C-DA" if cfg.da_enabled else "C-DA-off",
                "protocol": "benchmark" if cfg.benchmark_mode else "explore",
                "holdout_fids_json": cfg.holdout_fids_json,
                "train_years": cfg.train_years,
                "val_years": cfg.val_years,
            },
            f,
            indent=2,
        )

    # Val dataset
    all_val_days = sorted(val_days)
    if cfg.val_days_per_epoch and cfg.val_days_per_epoch < len(all_val_days):
        rng = np.random.default_rng(cfg.seed)
        chosen = rng.choice(len(all_val_days), cfg.val_days_per_epoch, replace=False)
        fixed_val_days = {all_val_days[i] for i in sorted(chosen)}
    else:
        fixed_val_days = val_days

    val_ds = GridDAPatchDataset(
        payload_cols=cfg.payload_cols,
        teacher_dir=cfg.teacher_dir if cfg.dense_loss_weight > 0 else None,
        teacher_pattern=cfg.teacher_pattern,
        increment_map=cfg.increment_map if cfg.dense_loss_weight > 0 else None,
        train_days=fixed_val_days,
        target_include_fids=holdout_fids or None,
        **base_kwargs,
    )
    print(f"Val samples: {len(val_ds)}")

    # Dataloaders
    day_sampler = DayGroupedSampler(
        train_ds.samples, days_per_epoch=cfg.days_per_epoch, base_seed=cfg.seed
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=day_sampler,
        num_workers=cfg.num_workers,
        persistent_workers=(cfg.num_workers > 0),
        collate_fn=collate_grid_da,
    )
    n_val_days = val_ds.samples["day"].nunique()
    val_sampler = DayGroupedSampler(
        val_ds.samples, days_per_epoch=n_val_days, base_seed=cfg.seed
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        persistent_workers=(cfg.num_workers > 0),
        collate_fn=collate_grid_da,
    )

    # Model
    model = LitGridDA(
        in_channels=in_channels,
        target_names=cfg.target_names,
        source_ctx_dim=source_ctx_dim,
        source_pay_dim=source_pay_dim,
        hidden_dim=cfg.hidden_dim,
        da_enabled=cfg.da_enabled,
        lr=cfg.lr,
        tv_weight=cfg.tv_weight,
        huber_delta=cfg.huber_delta,
        dense_loss_weight=cfg.dense_loss_weight,
        support_radius_px=cfg.support_radius_px,
        da_gate_init_bias=cfg.da_gate_init_bias,
        benchmark_mode=cfg.benchmark_mode,
    )

    # Load Stage B weights
    if cfg.pretrained_ckpt:
        if not os.path.exists(cfg.pretrained_ckpt):
            raise FileNotFoundError(
                f"pretrained_ckpt does not exist: {cfg.pretrained_ckpt}"
            )
        _load_stage_b_weights(model, cfg.pretrained_ckpt)

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

    # Symlink best checkpoint
    ckpt_cb = [c for c in callbacks if isinstance(c, ModelCheckpoint)][0]
    best = ckpt_cb.best_model_path
    if best:
        link = os.path.join(cfg.out_dir, "best.ckpt")
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.basename(best), link)

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
