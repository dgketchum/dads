"""
Train DA benchmark GNN (da-graph-v0).

Uses pre-built HeteroData graphs with source/query separation.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

import lightning as L
import pandas as pd
import tomli_w
import tomllib
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch_geometric.loader import DataLoader

from models.rtma_bias.da_graph_dataset import DAGraphDataset
from models.rtma_bias.lit_da_gnn import LitDAGNN


@dataclass
class DAGNNConfig:
    name: str = "da_s_tmax"
    description: str = ""

    hidden_dim: int = 64
    n_hops: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-3
    huber_delta: float = 1.0
    dropout: float = 0.3
    batch_size: int = 16
    epochs: int = 100
    seed: int = 42
    task: str = "scalar"
    target_names: list[str] = field(default_factory=list)
    target_index: int | None = None
    da_version: str = "v1"
    disable_payload: bool = False
    da_gate_init_bias: float = -2.0
    source_edge_dropout: float = 0.0

    da_split_enabled: bool = False
    da_source_fraction: float = 0.5
    da_exclude_radius_km: float = 20.0
    da_target_source_k: int | None = 16
    split_seed: int = 42

    da_mixed_local_enabled: bool = False
    da_mixed_local_graph_fraction: float = 0.0
    da_mixed_local_max_edges_per_query: int = 0

    train_years: list[int] = field(default_factory=list)
    val_years: list[int] = field(default_factory=list)

    graph_dir: str = ""
    out_dir: str = ""
    holdout_fids_json: str = ""
    device: str | None = None
    num_workers: int = 0

    def save_toml(self, path: str) -> None:
        d = asdict(self)
        d = {k: v for k, v in d.items() if v is not None}
        with open(path, "wb") as f:
            tomli_w.dump(d, f)

    @classmethod
    def from_toml(cls, path: str) -> DAGNNConfig:
        with open(path, "rb") as f:
            d = tomllib.load(f)
        return cls(**d)


class DASplitEpochCallback(L.Callback):
    """Call set_epoch and log split diagnostics each epoch."""

    def __init__(self, train_dataset: DAGraphDataset, n_audit: int = 10) -> None:
        self.train_dataset = train_dataset
        self.n_audit = n_audit

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        import numpy as _np

        epoch = trainer.current_epoch
        self.train_dataset.set_epoch(epoch)

        # Audit first n_audit graphs to report split geometry
        ds = self.train_dataset
        n = min(self.n_audit, len(ds))
        n_src_list, n_qry_list, n_holdout_list = [], [], []
        edges_before_list, edges_after_list = [], []
        min_dist_list = []
        src_per_qry = []
        dist_mean = ds._sq_edge_norm.get("dist_mean", 0.0)
        dist_std = ds._sq_edge_norm.get("dist_std", 1.0)

        for i in range(n):
            g_raw = ds._graphs[i]
            # Edge count before split
            raw_ei = g_raw["source", "influences", "query"].edge_index
            edges_before_list.append(raw_ei.shape[1] if raw_ei.numel() > 0 else 0)

            # Holdout count
            q_fids = list(g_raw["query"].fids)
            n_hold = sum(1 for f in q_fids if f not in (ds.loss_fids or set()))
            n_holdout_list.append(n_hold)

            out = ds[i]
            n_src_list.append(out["source"].context_x.shape[0])
            n_qry_list.append(int(out["query"].loss_mask.sum()))
            sq_ei = out["source", "influences", "query"].edge_index
            edges_after_list.append(sq_ei.shape[1] if sq_ei.numel() > 0 else 0)

            # Nearest source distance per supervised query
            qry_idx = _np.where(out["query"].loss_mask.numpy())[0]
            if len(qry_idx) > 0:
                dst_all = (
                    sq_ei[1].numpy() if sq_ei.numel() > 0 else _np.array([], dtype=int)
                )
                counts = _np.array(
                    [(dst_all == qi).sum() for qi in qry_idx], dtype=float
                )
                src_per_qry.extend(counts.tolist())
            if sq_ei.numel() > 0:
                ea = out["source", "influences", "query"].edge_attr
                d_km = (ea[:, 0] * dist_std + dist_mean).numpy()
                dst = sq_ei[1].numpy()
                for qi in qry_idx:
                    d = d_km[dst == qi]
                    if len(d) > 0:
                        min_dist_list.append(float(d.min()))

        md = _np.array(min_dist_list) if min_dist_list else _np.array([0.0])
        spq = _np.array(src_per_qry) if src_per_qry else _np.array([0.0])
        full_k = (
            float((spq >= ds.da_target_source_k).mean())
            if ds.da_target_source_k is not None and spq.size > 0
            else 0.0
        )
        base_msg = (
            f"  [epoch {epoch}] split audit ({n} graphs):\n"
            f"    src={_np.mean(n_src_list):.0f}, "
            f"qry={_np.mean(n_qry_list):.0f}, "
            f"holdout={_np.mean(n_holdout_list):.0f}\n"
            f"    edges: before={_np.mean(edges_before_list):.0f}, "
            f"after={_np.mean(edges_after_list):.0f}\n"
            f"    src_per_qry: mean={spq.mean():.1f} med={_np.median(spq):.1f} "
            f"zero={(spq == 0).mean():.1%} full_k={full_k:.1%}\n"
            f"    nearest_src_km: "
            f"min={md.min():.1f} p25={_np.percentile(md, 25):.1f} "
            f"med={_np.median(md):.1f} p75={_np.percentile(md, 75):.1f} "
            f"p95={_np.percentile(md, 95):.1f} max={md.max():.1f}"
        )

        # Mixed-local audit: report local vs far edge counts using
        # already-collected nearest-source distances (no extra ds[i] calls)
        if ds.da_mixed_local_enabled and min_dist_list:
            local_thresh = ds.da_exclude_radius_km
            md_arr = _np.array(min_dist_list)
            n_with_local = int((md_arr < local_thresh).sum())
            n_total = len(md_arr)
            base_msg += (
                f"\n    mixed-local: qry_with_local={n_with_local / n_total:.1%} "
                f"local_edges={n_with_local}/{n_total}"
            )

        print(base_msg, flush=True)


def _parse_args():
    p = argparse.ArgumentParser(description="Train DA benchmark GNN.")
    p.add_argument("--config", required=True)
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = DAGNNConfig.from_toml(args.config)

    L.seed_everything(cfg.seed, workers=True)
    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg.save_toml(os.path.join(cfg.out_dir, "experiment.toml"))

    # Holdout
    with open(cfg.holdout_fids_json) as f:
        holdout_fids = set(str(x) for x in json.load(f))
    print(f"Holdout fids: {len(holdout_fids)}")

    # Temporal split
    from pathlib import Path

    all_pt = sorted(Path(cfg.graph_dir).glob("*.pt"))
    all_dates = [pd.Timestamp(p.stem) for p in all_pt]
    train_days = {d for d in all_dates if d.year in set(cfg.train_years)}
    val_days = {d for d in all_dates if d.year in set(cfg.val_years)}
    print(f"Temporal split: {len(train_days)} train days, {len(val_days)} val days")

    # Discover all fids for train_loss_fids
    probe = DAGraphDataset(graph_dir=cfg.graph_dir, train_days=train_days)
    all_fids = set()
    for g in probe._graphs:
        all_fids.update(g["query"].fids)
    effective_holdout = holdout_fids & all_fids
    train_loss_fids = all_fids - holdout_fids
    del probe

    # Build datasets
    print("Building training dataset...")
    train_ds = DAGraphDataset(
        graph_dir=cfg.graph_dir,
        train_days=train_days,
        loss_fids=train_loss_fids,
        target_index=cfg.target_index,
        is_train=True,
        da_split_enabled=cfg.da_split_enabled,
        da_source_fraction=cfg.da_source_fraction,
        da_exclude_radius_km=cfg.da_exclude_radius_km,
        da_target_source_k=cfg.da_target_source_k,
        split_seed=cfg.split_seed,
        da_mixed_local_enabled=cfg.da_mixed_local_enabled,
        da_mixed_local_graph_fraction=cfg.da_mixed_local_graph_fraction,
        da_mixed_local_max_edges_per_query=cfg.da_mixed_local_max_edges_per_query,
    )
    train_ds.save_norm_stats(os.path.join(cfg.out_dir, "norm_stats.json"))

    print("Building validation dataset...")
    val_ds = DAGraphDataset(
        graph_dir=cfg.graph_dir,
        train_days=val_days,
        loss_fids=holdout_fids,
        norm_stats=train_ds.norm_stats,
        target_index=cfg.target_index,
    )

    # Effective holdout reporting
    val_fids = set()
    for g in val_ds._graphs:
        val_fids.update(g["query"].fids)
    val_holdout = holdout_fids & val_fids
    print(
        f"Holdout: {len(holdout_fids)} requested, "
        f"{len(effective_holdout)} in graphs, "
        f"{len(val_holdout)} on val days"
    )
    if len(val_holdout) < 100:
        raise ValueError(f"Effective val holdout is only {len(val_holdout)} stations.")

    with open(os.path.join(cfg.out_dir, "holdout_fids.json"), "w") as f:
        json.dump(sorted(holdout_fids), f)
    with open(os.path.join(cfg.out_dir, "effective_holdout_fids.json"), "w") as f:
        json.dump(sorted(effective_holdout), f)
    with open(os.path.join(cfg.out_dir, "val_holdout_fids.json"), "w") as f:
        json.dump(sorted(val_holdout), f)

    print(f"Train: {len(train_ds)} days, Val: {len(val_ds)} days")
    print(f"Effective family: {train_ds.effective_family}")
    if cfg.da_split_enabled:
        print(
            f"DA split: enabled, source_frac={cfg.da_source_fraction}, "
            f"exclude_radius={cfg.da_exclude_radius_km}km, "
            f"target_source_k={cfg.da_target_source_k}, seed={cfg.split_seed}"
        )
        print(
            f"Candidate source graph: source_k={train_ds.source_candidate_k}, "
            f"source_radius={train_ds.source_candidate_radius_km}km"
        )
        if cfg.da_mixed_local_enabled:
            print(
                f"Mixed-local: graph_frac={cfg.da_mixed_local_graph_fraction}, "
                f"max_local/qry={cfg.da_mixed_local_max_edges_per_query}, "
                f"local_thresh=exclude_radius={cfg.da_exclude_radius_km}km"
            )

    # Record effective family in run metadata
    with open(os.path.join(cfg.out_dir, "split_pointer.json"), "w") as f:
        json.dump(
            {
                "artifact_family": train_ds._family,
                "effective_family": train_ds.effective_family,
                "da_split_enabled": cfg.da_split_enabled,
                "da_source_fraction": cfg.da_source_fraction,
                "da_exclude_radius_km": cfg.da_exclude_radius_km,
                "da_target_source_k": cfg.da_target_source_k,
                "split_seed": cfg.split_seed,
                "da_mixed_local_enabled": cfg.da_mixed_local_enabled,
                "da_mixed_local_graph_fraction": cfg.da_mixed_local_graph_fraction,
                "da_mixed_local_max_edges_per_query": cfg.da_mixed_local_max_edges_per_query,
                "graph_dir": cfg.graph_dir,
                "holdout_fids_json": cfg.holdout_fids_json,
                "train_years": cfg.train_years,
                "val_years": cfg.val_years,
            },
            f,
            indent=2,
        )
    print(
        f"Query features: {train_ds.query_node_dim}, "
        f"Source context: {train_ds.source_context_dim}, "
        f"Source payload: {train_ds.source_payload_dim}"
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    # Target scales for multitask — fit on train-year non-holdout query nodes only.
    # Use raw graphs (before __getitem__ normalization) with manual loss_mask.
    target_scales = None
    if cfg.task == "multitask" and cfg.target_names:
        import torch

        # Load raw train graphs for scale computation
        raw_probe = DAGraphDataset(graph_dir=cfg.graph_dir, train_days=train_days)
        scale_vals = []
        for ti, tn in enumerate(cfg.target_names):
            ys = []
            for g in raw_probe._graphs:
                y_g = g["query"].y[:, ti] if g["query"].y.ndim == 2 else g["query"].y
                vm = (
                    g["query"].valid_mask[:, ti]
                    if hasattr(g["query"], "valid_mask")
                    and g["query"].valid_mask.ndim == 2
                    else torch.ones(y_g.shape[0], dtype=torch.bool)
                )
                # Manual loss_mask: non-holdout fids only
                lm = torch.tensor(
                    [f in train_loss_fids for f in g["query"].fids], dtype=torch.bool
                )
                mask = vm & lm
                if mask.any():
                    ys.append(y_g[mask])
            all_y = torch.cat(ys)
            mad = (all_y - all_y.median()).abs().median().item()
            scale_vals.append(max(mad * 1.4826, 1e-6))
        target_scales = scale_vals
        del raw_probe
        print(f"Target scales (MAD): {dict(zip(cfg.target_names, target_scales))}")

    # Scalar: checkpoint on val/target_mae. Multitask: val_loss is already
    # MAD-normalized when target_scales is set, so it serves as the normalized
    # 2-head selection metric required by the policy.
    ckpt_metric = "val/target_mae" if cfg.task == "scalar" else "val_loss"

    model = LitDAGNN(
        query_node_dim=train_ds.query_node_dim,
        source_context_dim=train_ds.source_context_dim,
        source_payload_dim=train_ds.source_payload_dim,
        edge_dim=train_ds.edge_dim,
        hidden_dim=cfg.hidden_dim,
        n_hops=cfg.n_hops,
        dropout=cfg.dropout,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        huber_delta=1.0 if target_scales else cfg.huber_delta,
        task=cfg.task,
        target_names=cfg.target_names if cfg.task == "multitask" else None,
        target_scales=target_scales,
        disable_payload=cfg.disable_payload,
        da_gate_init_bias=cfg.da_gate_init_bias,
        source_edge_dropout=cfg.source_edge_dropout,
        da_version=cfg.da_version,
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
            monitor=ckpt_metric,
            save_top_k=3,
            mode="min",
            filename="ckpt-{epoch:03d}-{" + ckpt_metric + ":.4f}",
        ),
        EarlyStopping(monitor=ckpt_metric, patience=20, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if cfg.da_split_enabled:
        callbacks.append(DASplitEpochCallback(train_ds))

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed" if accelerator == "gpu" else 32,
        gradient_clip_val=1.0,
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
