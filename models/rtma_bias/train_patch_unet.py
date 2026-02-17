from __future__ import annotations

import argparse
import os

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from models.rtma_bias.data_module import RtmaPatchDataModule
from models.rtma_bias.experiment import ExperimentConfig, log_experiment
from models.rtma_bias.lit_unet import LitPatchUNet

# Default RTMA channels when no --config and no --rtma-channels provided.
_DEFAULT_RTMA = (
    "tmp_c",
    "dpt_c",
    "ugrd",
    "vgrd",
    "pres_kpa",
    "tcdc_pct",
    "prcp_mm",
    "ea_kpa",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a patch-based RTMA humidity correction U-Net (MVP)."
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to experiment TOML config (optional; CLI args override TOML values)",
    )
    p.add_argument(
        "--patch-index",
        default=None,
        help="Parquet from prep/build_rtma_patch_index.py",
    )
    p.add_argument(
        "--tif-root",
        default=None,
        help="Directory containing RTMA_YYYYMMDD.tif daily COGs",
    )
    p.add_argument(
        "--out-dir", default=None, help="Output directory for checkpoints/config"
    )
    p.add_argument("--patch-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--tv-weight", type=float, default=None)
    p.add_argument("--base", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default=None, help="Override device (e.g., cuda:0, cpu)")
    p.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pre-load all COGs into memory (default: True; use --no-preload to disable)",
    )
    p.add_argument(
        "--val-frac", type=float, default=None, help="Fraction of data for validation"
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    p.add_argument(
        "--terrain-tif",
        default=None,
        help="6-band terrain GeoTIFF (from prep/build_terrain_grid.py)",
    )
    p.add_argument(
        "--rsun-tif",
        default=None,
        help="365-band RSUN GeoTIFF (from prep/build_terrain_grid.py)",
    )
    p.add_argument(
        "--landsat-tif",
        default=None,
        help="35-band Landsat composite GeoTIFF (from prep/build_landsat_grid.py)",
    )
    p.add_argument(
        "--target-col",
        default=None,
        help="Target column in patch index (default: delta_log_ea)",
    )
    p.add_argument(
        "--rtma-channels",
        default=None,
        help="Comma-separated RTMA channel names (default: all 8)",
    )
    p.add_argument(
        "--start-date",
        default=None,
        help="Filter patch index to days >= this date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--end-date",
        default=None,
        help="Filter patch index to days <= this date (YYYY-MM-DD)",
    )
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build an ExperimentConfig from TOML (if provided) + CLI overrides."""
    if args.config:
        cfg = ExperimentConfig.from_toml(args.config)
    else:
        cfg = ExperimentConfig()

    # Map CLI arg names → ExperimentConfig field names.
    _cli_map = {
        "patch_index": "patch_index",
        "tif_root": "tif_root",
        "out_dir": "out_dir",
        "patch_size": "patch_size",
        "batch_size": "batch_size",
        "epochs": "epochs",
        "lr": "lr",
        "tv_weight": "tv_weight",
        "base": "base",
        "num_workers": "num_workers",
        "device": "device",
        "preload": "preload",
        "val_frac": "val_frac",
        "seed": "seed",
        "terrain_tif": "terrain_tif",
        "rsun_tif": "rsun_tif",
        "landsat_tif": "landsat_tif",
        "target_col": "target_col",
        "start_date": "start_date",
        "end_date": "end_date",
        "decoded": "decoded",
    }
    for cli_name, cfg_name in _cli_map.items():
        val = getattr(args, cli_name, None)
        if val is not None:
            setattr(cfg, cfg_name, val)

    # --rtma-channels overrides features with explicit channel list.
    if args.rtma_channels is not None:
        channels = [c.strip() for c in args.rtma_channels.split(",")]
        cfg.features = channels

    # When no --config was given AND features still at default, apply legacy
    # defaults: check if auxiliary TIFs were provided to decide feature set.
    if not args.config and cfg.features == ExperimentConfig().features:
        cfg.features = list(_DEFAULT_RTMA)
        if cfg.terrain_tif:
            cfg.features.append("terrain")
        if cfg.rsun_tif:
            cfg.features.append("rsun")
        if cfg.landsat_tif:
            cfg.features.append("landsat")

    # Apply legacy defaults for fields that are still unset.
    _legacy_defaults = {
        "patch_size": 64,
        "batch_size": 64,
        "epochs": 5,
        "lr": 3e-4,
        "tv_weight": 1e-3,
        "base": 32,
        "num_workers": 2,
        "preload": True,
        "val_frac": 0.2,
        "seed": 42,
        "target_col": "delta_log_ea",
    }
    for k, v in _legacy_defaults.items():
        if getattr(cfg, k) is None:
            setattr(cfg, k, v)

    # Validate required paths.
    if not cfg.patch_index:
        raise SystemExit("Error: --patch-index is required (via CLI or config TOML)")
    if not cfg.tif_root:
        raise SystemExit("Error: --tif-root is required (via CLI or config TOML)")
    if not cfg.out_dir:
        raise SystemExit("Error: --out-dir is required (via CLI or config TOML)")

    return cfg


def main() -> None:
    a = _parse_args()
    cfg = _build_config(a)

    L.seed_everything(cfg.seed, workers=True)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Save experiment config for reproducibility.
    cfg.save_toml(os.path.join(cfg.out_dir, "experiment.toml"))

    rtma_channels = cfg.rtma_channels or None

    dm = RtmaPatchDataModule(
        patch_index=cfg.patch_index,
        tif_root=cfg.tif_root,
        patch_size=cfg.patch_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        val_frac=cfg.val_frac,
        seed=cfg.seed,
        preload=cfg.preload,
        terrain_tif=cfg.terrain_tif if cfg.use_terrain else None,
        rsun_tif=cfg.rsun_tif if cfg.use_rsun else None,
        landsat_tif=cfg.landsat_tif if cfg.use_landsat else None,
        target_col=cfg.target_col,
        rtma_channels=rtma_channels,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        decoded=cfg.decoded,
    )
    dm.setup()

    # Save normalisation stats for inference.
    norm_path = os.path.join(cfg.out_dir, "norm_stats.json")
    dm.save_norm_stats(norm_path)

    model = LitPatchUNet(
        in_channels=dm.in_channels,
        out_channels=1,
        base=cfg.base,
        lr=cfg.lr,
        tv_weight=cfg.tv_weight,
    )

    # Determine accelerator / devices from --device flag.
    if cfg.device and cfg.device.startswith("cuda"):
        accelerator = "gpu"
        if ":" in cfg.device:
            devices = [int(cfg.device.split(":")[1])]
        else:
            devices = 1
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
                save_top_k=5,
                mode="min",
                filename="ckpt-{epoch:03d}-{val_loss:.4f}",
            ),
            EarlyStopping(monitor="val_loss", patience=20, mode="min"),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        default_root_dir=cfg.out_dir,
        log_every_n_steps=50,
    )
    trainer.fit(model, dm)

    # Log experiment results.
    metrics = {
        k: v.item() if hasattr(v, "item") else v
        for k, v in trainer.callback_metrics.items()
    }
    log_experiment(cfg, metrics)


if __name__ == "__main__":
    main()
