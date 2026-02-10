"""
Train wind bias correction GNN.

Supports TOML config + CLI overrides, following the pattern from
models/rtma_bias/train_patch_unet.py.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

import lightning as L
import numpy as np
import pandas as pd
import tomllib
import tomli_w
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch_geometric.loader import DataLoader

from models.wind_bias.lit_wind_gnn import LitWindGNN
from models.wind_bias.wind_dataset import WindGraphDataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class WindGNNConfig:
    name: str = "wind_gnn"
    description: str = ""

    # Model
    hidden_dim: int = 64
    n_hops: int = 1
    use_graph: bool = True
    use_sx: bool = True
    use_flow_terrain: bool = True

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-5
    huber_delta: float = 2.0
    calm_threshold: float = 2.0
    calm_min_weight: float = 0.1
    correction_penalty: float = 0.0
    batch_size: int = 16
    epochs: int = 100
    seed: int = 42
    k_neighbors: int = 16

    # Data
    table_path: str = "/nas/dads/mvp/station_day_wind_pnw_2018_2024.parquet"
    stations_csv: str = "artifacts/madis_pnw.csv"
    out_dir: str = "/nas/dads/mvp/wind_gnn_v1"
    train_years: list[int] = field(
        default_factory=lambda: [2018, 2019, 2020, 2021, 2022, 2023]
    )
    val_years: list[int] = field(default_factory=lambda: [2024])

    # Spatial holdout
    spatial_holdout_frac: float = 0.1
    spatial_holdout_seed: int = 99

    # Runtime
    device: str | None = None
    num_workers: int = 0

    def save_toml(self, path: str) -> None:
        d = asdict(self)
        d = {k: v for k, v in d.items() if v is not None}
        with open(path, "wb") as f:
            tomli_w.dump(d, f)

    @classmethod
    def from_toml(cls, path: str) -> WindGNNConfig:
        with open(path, "rb") as f:
            d = tomllib.load(f)
        return cls(**d)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train wind bias GNN.")
    p.add_argument("--config", default=None, help="TOML config path")
    p.add_argument("--table-path", default=None)
    p.add_argument("--stations-csv", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--n-hops", type=int, default=None)
    p.add_argument("--use-graph", type=int, default=None, help="0=MLP only, 1=GNN")
    p.add_argument("--use-sx", type=int, default=None, help="0=no Sx, 1=with Sx")
    p.add_argument("--use-flow-terrain", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--k-neighbors", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> WindGNNConfig:
    if args.config:
        cfg = WindGNNConfig.from_toml(args.config)
    else:
        cfg = WindGNNConfig()

    cli_map = {
        "table_path": "table_path",
        "stations_csv": "stations_csv",
        "out_dir": "out_dir",
        "hidden_dim": "hidden_dim",
        "n_hops": "n_hops",
        "lr": "lr",
        "batch_size": "batch_size",
        "epochs": "epochs",
        "k_neighbors": "k_neighbors",
        "seed": "seed",
        "device": "device",
        "num_workers": "num_workers",
    }
    for cli_name, cfg_name in cli_map.items():
        val = getattr(args, cli_name, None)
        if val is not None:
            setattr(cfg, cfg_name, val)

    # Boolean flags via int
    if args.use_graph is not None:
        cfg.use_graph = bool(args.use_graph)
    if args.use_sx is not None:
        cfg.use_sx = bool(args.use_sx)
    if args.use_flow_terrain is not None:
        cfg.use_flow_terrain = bool(args.use_flow_terrain)

    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    cfg = _build_config(args)

    L.seed_everything(cfg.seed, workers=True)
    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg.save_toml(os.path.join(cfg.out_dir, "experiment.toml"))

    # Load table to determine train/val splits
    print("Loading station-day table...")
    df = pd.read_parquet(cfg.table_path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["day"] = pd.to_datetime(df["day"])

    # Temporal split
    train_days = set(df[df["day"].dt.year.isin(cfg.train_years)]["day"].unique())
    val_days = set(df[df["day"].dt.year.isin(cfg.val_years)]["day"].unique())
    print(f"Train days: {len(train_days)}, Val days: {len(val_days)}")

    # Spatial holdout: stratified by elevation quartile
    fid_elev = df.groupby("fid")["elevation"].first().dropna()
    fids_with_elev = fid_elev.index.tolist()
    rng = np.random.default_rng(cfg.spatial_holdout_seed)
    n_holdout = max(1, int(len(fids_with_elev) * cfg.spatial_holdout_frac))
    quartiles = pd.qcut(fid_elev, q=4, labels=False, duplicates="drop")
    holdout_fids: set[str] = set()
    for q in sorted(quartiles.unique()):
        q_fids = quartiles[quartiles == q].index.tolist()
        n_q = max(1, round(n_holdout * len(q_fids) / len(fids_with_elev)))
        chosen = rng.choice(q_fids, size=min(n_q, len(q_fids)), replace=False)
        holdout_fids.update(chosen)
    print(f"Spatial holdout: {len(holdout_fids)} stations (elevation-stratified)")

    # Save holdout list for reproducibility
    with open(os.path.join(cfg.out_dir, "holdout_fids.json"), "w") as f:
        json.dump(sorted(holdout_fids), f)

    print("Building training dataset...")
    train_ds = WindGraphDataset(
        table_path=cfg.table_path,
        stations_csv=cfg.stations_csv,
        k=cfg.k_neighbors,
        use_graph=cfg.use_graph,
        use_sx=cfg.use_sx,
        use_flow_terrain=cfg.use_flow_terrain,
        train_days=train_days,
        exclude_fids=holdout_fids,
    )
    train_ds.save_norm_stats(os.path.join(cfg.out_dir, "norm_stats.json"))

    print("Building validation dataset...")
    val_ds = WindGraphDataset(
        table_path=cfg.table_path,
        stations_csv=cfg.stations_csv,
        k=cfg.k_neighbors,
        use_graph=cfg.use_graph,
        use_sx=cfg.use_sx,
        use_flow_terrain=cfg.use_flow_terrain,
        norm_stats=train_ds.norm_stats,
        train_days=val_days,
    )

    print(f"Train: {len(train_ds)} days, Val: {len(val_ds)} days")
    print(f"Node features: {train_ds.node_dim}, Edge features: {train_ds.edge_dim}")

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

    model = LitWindGNN(
        node_dim=train_ds.node_dim,
        edge_dim=train_ds.edge_dim,
        hidden_dim=cfg.hidden_dim,
        n_hops=cfg.n_hops,
        use_graph=cfg.use_graph,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        huber_delta=cfg.huber_delta,
        calm_threshold=cfg.calm_threshold,
        calm_min_weight=cfg.calm_min_weight,
        correction_penalty=cfg.correction_penalty,
    )

    # Accelerator
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
        gradient_clip_val=1.0,
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

    # Log results
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
