"""
Train scalar bias-correction GNN (tmax, EA, etc.).

TOML config + CLI overrides. Uses precomputed per-day .pt graphs
built by prep/build_graphs.py.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

import lightning as L
import pandas as pd
import tomllib
import tomli_w
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch_geometric.loader import DataLoader

from models.rtma_bias.lit_scalar_gnn import LitScalarGNN
from models.rtma_bias.scalar_gnn_dataset import PrecomputedGraphDataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ScalarGNNConfig:
    name: str = "e2_tmax_gnn"
    description: str = ""

    # Model
    hidden_dim: int = 128
    n_hops: int = 1
    use_graph: bool = True

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-5
    huber_delta: float = 1.0
    dropout: float = 0.0
    batch_size: int = 16
    epochs: int = 100
    seed: int = 42

    # Data
    graph_dir: str = "/nas/dads/mvp/tmax_graphs_pnw_2024"
    val_graph_dir: str | None = None
    out_dir: str = "/nas/dads/mvp/e2_tmax_gnn"

    # Spatial holdout (MGRS tile-based)
    val_mgrs_tiles: list[str] = field(default_factory=list)
    stations_csv: str = "/nas/dads/met/stations/madis_02JULY2025_mgrs.csv"

    # Runtime
    device: str | None = None
    num_workers: int = 0

    def save_toml(self, path: str) -> None:
        d = asdict(self)
        d = {k: v for k, v in d.items() if v is not None}
        with open(path, "wb") as f:
            tomli_w.dump(d, f)

    @classmethod
    def from_toml(cls, path: str) -> ScalarGNNConfig:
        with open(path, "rb") as f:
            d = tomllib.load(f)
        return cls(**d)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train scalar bias-correction GNN.")
    p.add_argument("--config", default=None, help="TOML config path")
    p.add_argument("--graph-dir", default=None)
    p.add_argument("--val-graph-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--n-hops", type=int, default=None)
    p.add_argument("--use-graph", type=int, default=None, help="0=MLP only, 1=GNN")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> ScalarGNNConfig:
    if args.config:
        cfg = ScalarGNNConfig.from_toml(args.config)
    else:
        cfg = ScalarGNNConfig()

    cli_map = {
        "graph_dir": "graph_dir",
        "val_graph_dir": "val_graph_dir",
        "out_dir": "out_dir",
        "hidden_dim": "hidden_dim",
        "n_hops": "n_hops",
        "lr": "lr",
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

    if args.use_graph is not None:
        cfg.use_graph = bool(args.use_graph)

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

    # ---- MGRS-based spatial holdout ----
    holdout_fids: set[str] = set()
    if cfg.val_mgrs_tiles:
        stations = pd.read_csv(cfg.stations_csv)
        id_col = "station_id" if "station_id" in stations.columns else "fid"
        stations[id_col] = stations[id_col].astype(str)
        val_tiles = set(cfg.val_mgrs_tiles)
        holdout_fids = set(
            stations.loc[stations["MGRS_TILE"].isin(val_tiles), id_col].values
        )
        print(
            f"MGRS spatial holdout: {len(holdout_fids)} stations "
            f"from {len(val_tiles)} tiles"
        )

    with open(os.path.join(cfg.out_dir, "holdout_fids.json"), "w") as f:
        json.dump(sorted(holdout_fids), f)

    # ---- Build datasets (transductive spatial holdout) ----
    # All nodes stay in every graph for message passing; loss_mask selects
    # which nodes contribute to the loss.
    val_gdir = cfg.val_graph_dir or cfg.graph_dir
    print(f"Loading training graphs from {cfg.graph_dir}")
    if cfg.val_graph_dir:
        print(f"Loading validation graphs from {val_gdir}")

    # Determine train loss fids (all fids minus holdout)
    train_loss_fids = None
    if holdout_fids:
        print("Building training dataset (transductive, all nodes in graph)...")
        train_ds = PrecomputedGraphDataset(
            graph_dir=cfg.graph_dir,
            use_graph=cfg.use_graph,
        )
        all_fids = set()
        for g in train_ds._graphs:
            all_fids.update(g.fids)
        train_loss_fids = all_fids - holdout_fids
        train_ds.loss_fids = train_loss_fids
        print(
            f"  {len(train_loss_fids)} train loss fids, "
            f"{len(holdout_fids)} holdout fids, "
            f"{len(all_fids)} total"
        )
    else:
        print("Building training dataset...")
        train_ds = PrecomputedGraphDataset(
            graph_dir=cfg.graph_dir,
            use_graph=cfg.use_graph,
        )
    train_ds.save_norm_stats(os.path.join(cfg.out_dir, "norm_stats.json"))

    print("Building validation dataset (transductive, loss on holdout only)...")
    val_ds = PrecomputedGraphDataset(
        graph_dir=val_gdir,
        use_graph=cfg.use_graph,
        norm_stats=train_ds.norm_stats,
        loss_fids=holdout_fids if holdout_fids else None,
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

    model = LitScalarGNN(
        node_dim=train_ds.node_dim,
        edge_dim=train_ds.edge_dim,
        hidden_dim=cfg.hidden_dim,
        n_hops=cfg.n_hops,
        use_graph=cfg.use_graph,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        huber_delta=cfg.huber_delta,
        dropout=cfg.dropout,
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
