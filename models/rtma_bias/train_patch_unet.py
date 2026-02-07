from __future__ import annotations

import argparse
import os

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from models.rtma_bias.data_module import RtmaPatchDataModule
from models.rtma_bias.lit_unet import LitPatchUNet


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a patch-based RTMA humidity correction U-Net (MVP)."
    )
    p.add_argument(
        "--patch-index",
        required=True,
        help="Parquet from prep/build_rtma_patch_index.py",
    )
    p.add_argument(
        "--tif-root",
        required=True,
        help="Directory containing RTMA_YYYYMMDD.tif daily COGs",
    )
    p.add_argument(
        "--out-dir", required=True, help="Output directory for checkpoints/config"
    )
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tv-weight", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None, help="Override device (e.g., cuda:0, cpu)")
    p.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pre-load all COGs into memory (default: True; use --no-preload to disable)",
    )
    p.add_argument(
        "--val-frac", type=float, default=0.2, help="Fraction of data for validation"
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
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
    return p.parse_args()


def main() -> None:
    a = _parse_args()

    L.seed_everything(a.seed, workers=True)

    os.makedirs(a.out_dir, exist_ok=True)

    dm = RtmaPatchDataModule(
        patch_index=a.patch_index,
        tif_root=a.tif_root,
        patch_size=a.patch_size,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        val_frac=a.val_frac,
        seed=a.seed,
        preload=a.preload,
        terrain_tif=a.terrain_tif,
        rsun_tif=a.rsun_tif,
    )
    dm.setup()

    # Save normalisation stats for inference.
    norm_path = os.path.join(a.out_dir, "norm_stats.json")
    dm.save_norm_stats(norm_path)

    model = LitPatchUNet(
        in_channels=dm.in_channels,
        out_channels=1,
        base=a.base,
        lr=a.lr,
        tv_weight=a.tv_weight,
    )

    # Determine accelerator / devices from --device flag.
    if a.device and a.device.startswith("cuda"):
        accelerator = "gpu"
        # e.g. "cuda:0" → devices=[0]; plain "cuda" → devices=1
        if ":" in a.device:
            devices = [int(a.device.split(":")[1])]
        else:
            devices = 1
    elif a.device == "cpu":
        accelerator = "cpu"
        devices = 1
    else:
        accelerator = "auto"
        devices = 1

    trainer = L.Trainer(
        max_epochs=a.epochs,
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed" if accelerator == "gpu" else 32,
        callbacks=[
            ModelCheckpoint(
                dirpath=a.out_dir,
                monitor="val_loss",
                save_top_k=3,
                mode="min",
                filename="ckpt-{epoch:03d}-{val_loss:.4f}",
            ),
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ],
        default_root_dir=a.out_dir,
        log_every_n_steps=50,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
