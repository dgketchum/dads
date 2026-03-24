"""
Train the hetero HRRR grid/station data-assimilation GNN.

This trainer expects a daily gridded background raster product. The repo does
not yet build HRRR daily 1 km backgrounds, so this path is intentionally strict
about missing rasters rather than silently falling back to the station-only
workflow.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import torch
import lightning as L
import pandas as pd
import tomllib
import tomli_w
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import RandomSampler, Subset
from torch_geometric.loader import DataLoader

from models.hrrr_da.hetero_dataset import HRRRHeteroTileDataset
from models.hrrr_da.lit_hetero_gnn import LitHRRRHeteroGNN
from prep.paths import MVP_ROOT


@dataclass
class HRRRHeteroConfig:
    name: str = "hrrr_hetero_da_v1"
    description: str = "Grid/station hetero DA GNN over daily HRRR backgrounds"

    hidden_dim: int = 128
    n_hops: int = 1
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    huber_delta: float = 1.0
    batch_size: int = 8
    epochs: int = 100
    seed: int = 42
    grid_radius_cells: int = 2
    station_radius_km: float = 150.0
    max_neighbor_stations: int = 16

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
    terrain_tif: str = f"{MVP_ROOT}/terrain_pnw_1km.tif"
    out_dir: str = f"{MVP_ROOT}/hrrr_hetero_da_v1"
    train_years: list[int] = field(
        default_factory=lambda: [2018, 2019, 2020, 2021, 2022, 2023]
    )
    val_years: list[int] = field(default_factory=lambda: [2024])
    holdout_fids_json: str | None = "artifacts/canonical_holdout_fids.json"

    device: str | None = None
    num_workers: int = 0
    max_samples_per_epoch: int | None = None
    days_per_epoch: int | None = None

    def save_toml(self, path: str) -> None:
        d = asdict(self)
        d = {k: v for k, v in d.items() if v is not None}
        with open(path, "wb") as f:
            tomli_w.dump(d, f)

    @classmethod
    def from_toml(cls, path: str) -> "HRRRHeteroConfig":
        with open(path, "rb") as f:
            return cls(**tomllib.load(f))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train hetero HRRR DA GNN.")
    p.add_argument("--config", default=None)
    p.add_argument("--table-path", default=None)
    p.add_argument("--background-dir", default=None)
    p.add_argument("--background-pattern", default=None)
    p.add_argument("--terrain-tif", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--n-hops", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> HRRRHeteroConfig:
    cfg = HRRRHeteroConfig.from_toml(args.config) if args.config else HRRRHeteroConfig()
    cli_map = {
        "table_path": "table_path",
        "background_dir": "background_dir",
        "background_pattern": "background_pattern",
        "terrain_tif": "terrain_tif",
        "out_dir": "out_dir",
        "hidden_dim": "hidden_dim",
        "n_hops": "n_hops",
        "batch_size": "batch_size",
        "epochs": "epochs",
        "seed": "seed",
        "device": "device",
        "num_workers": "num_workers",
    }
    for cli_name, cfg_name in cli_map.items():
        val = getattr(args, cli_name, None)
        if val is not None:
            setattr(cfg, cfg_name, val)
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
    if cfg.holdout_fids_json:
        with open(cfg.holdout_fids_json) as f:
            holdout_fids = set(json.load(f)) & all_fids
        print(f"Spatial holdout: {len(holdout_fids)} stations")
    with open(os.path.join(cfg.out_dir, "holdout_fids.json"), "w") as f:
        json.dump(sorted(holdout_fids), f)

    train_ds = HRRRHeteroTileDataset(
        table_path=cfg.table_path,
        background_dir=cfg.background_dir,
        background_pattern=cfg.background_pattern,
        terrain_tif=cfg.terrain_tif,
        target_names=cfg.target_names,
        train_days=train_days,
        target_exclude_fids=holdout_fids,
        neighbor_exclude_fids=holdout_fids,
        grid_radius_cells=cfg.grid_radius_cells,
        station_radius_km=cfg.station_radius_km,
        max_neighbor_stations=cfg.max_neighbor_stations,
    )
    norm_stats_path = os.path.join(cfg.out_dir, "norm_stats.json")
    norm_stats_out = {
        "grid_norm_stats": train_ds.grid_norm_stats,
        "station_norm_stats": train_ds.station_norm_stats,
        "background_feature_names": train_ds.background_feature_names,
        "terrain_feature_names": train_ds.terrain_feature_names,
        "grid_feature_names": train_ds.grid_feature_names,
        "station_feature_cols": train_ds.station_feature_cols,
        "station_mask_cols": train_ds.station_mask_cols,
        "target_names": train_ds.target_names,
        "station_radius_km": train_ds.station_radius_km,
        "grid_node_dim": train_ds.grid_node_dim,
        "station_node_dim": train_ds.station_node_dim,
        "edge_dim": train_ds.edge_dim,
    }
    with open(norm_stats_path, "w") as f:
        json.dump(norm_stats_out, f, indent=2)
    print(f"Norm stats saved: {norm_stats_path}")
    val_ds = HRRRHeteroTileDataset(
        table_path=cfg.table_path,
        background_dir=cfg.background_dir,
        background_pattern=cfg.background_pattern,
        terrain_tif=cfg.terrain_tif,
        target_names=cfg.target_names,
        train_days=val_days,
        target_include_fids=holdout_fids if holdout_fids else None,
        neighbor_exclude_fids=holdout_fids,
        grid_radius_cells=cfg.grid_radius_cells,
        station_radius_km=cfg.station_radius_km,
        max_neighbor_stations=cfg.max_neighbor_stations,
    )

    if len(train_ds) == 0:
        raise ValueError("Training hetero dataset is empty.")
    if len(val_ds) == 0:
        raise ValueError("Validation hetero dataset is empty.")

    if cfg.days_per_epoch is not None:
        # Day-grouped subsetting: select random days, keep all their samples in
        # day-sorted order.  Consecutive indices stay within the same day, so each
        # DataLoader worker only opens O(days_per_epoch / num_workers) rasters per
        # epoch instead of opening a new raster for every sample.
        rng = np.random.default_rng(cfg.seed)
        all_days = train_ds.samples["day"].unique()
        n_days = min(cfg.days_per_epoch, len(all_days))
        selected = set(rng.choice(all_days, n_days, replace=False))
        day_to_idx: dict = {}
        for day, grp in train_ds.samples.groupby("day"):
            if day in selected:
                day_to_idx[day] = grp.index.tolist()
        day_order = list(selected)
        rng.shuffle(day_order)
        sorted_indices: list[int] = []
        for day in day_order:
            idxs = day_to_idx[day]
            rng.shuffle(idxs)
            sorted_indices.extend(idxs)
        if cfg.max_samples_per_epoch is not None:
            sorted_indices = sorted_indices[: cfg.max_samples_per_epoch]
        train_effective: torch.utils.data.Dataset = Subset(train_ds, sorted_indices)
        train_loader = DataLoader(
            train_effective,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            persistent_workers=(cfg.num_workers > 0),
        )
    elif cfg.max_samples_per_epoch is not None:
        train_sampler = RandomSampler(
            train_ds,
            num_samples=min(cfg.max_samples_per_epoch, len(train_ds)),
            replacement=False,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=train_sampler,
            num_workers=cfg.num_workers,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = LitHRRRHeteroGNN(
        grid_node_dim=train_ds.grid_node_dim,
        station_node_dim=train_ds.station_node_dim,
        edge_dim=train_ds.edge_dim,
        hidden_dim=cfg.hidden_dim,
        n_hops=cfg.n_hops,
        dropout=cfg.dropout,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        huber_delta=cfg.huber_delta,
        target_names=cfg.target_names,
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

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed" if accelerator == "gpu" else 32,
        callbacks=[
            ModelCheckpoint(
                dirpath=cfg.out_dir,
                monitor="val_loss",
                save_top_k=3,
                mode="min",
                filename="ckpt-{epoch:03d}-{val_loss:.4f}",
            ),
            EarlyStopping(monitor="val_loss", patience=20, mode="min"),
            LearningRateMonitor(logging_interval="epoch"),
        ],
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
