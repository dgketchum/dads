from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.rtma_bias.patch_dataset import PatchDatasetConfig, RtmaHumidityPatchDataset
from models.rtma_bias.unet import UNetSmall


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    # x: (B, C, H, W)
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    return dx + dy


def center_pixel(x: torch.Tensor) -> torch.Tensor:
    # x: (B, C, H, W) -> (B, C)
    h = x.shape[-2]
    w = x.shape[-1]
    cy = h // 2
    cx = w // 2
    return x[:, :, cy, cx]


def train(
    patch_index: str,
    tif_root: str,
    out_dir: str,
    patch_size: int = 64,
    batch_size: int = 16,
    epochs: int = 5,
    lr: float = 3e-4,
    tv_weight: float = 1e-3,
    base: int = 32,
    num_workers: int = 0,
    device: str | None = None,
    preload: bool = True,
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    cfg = PatchDatasetConfig(tif_root=tif_root, patch_size=int(patch_size), preload=bool(preload))
    ds = RtmaHumidityPatchDataset(patch_index, cfg)
    def _collate(batch):
        xb, yb, meta = zip(*batch)
        return torch.stack(list(xb), dim=0), torch.stack(list(yb), dim=0), list(meta)

    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=bool(preload),
        collate_fn=_collate,
    )

    dev = str(device).lower() if device is not None else None
    if dev is None:
        # Default to CPU for MVP reproducibility; fall back from CUDA if the local
        # GPU/driver combo is incompatible with the installed torch wheel.
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cuda":
            try:
                _ = torch.zeros(1, device="cuda")
            except Exception:
                dev = "cpu"
    model = UNetSmall(in_channels=ds.in_channels, out_channels=1, base=int(base)).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))
    huber = nn.HuberLoss(delta=1.0)

    # Save config for provenance.
    with open(os.path.join(out_dir, "train_config.json"), "w") as f:
        import json
        json.dump(
            {
                "dataset": asdict(cfg),
                "train": {
                    "patch_index": patch_index,
                    "batch_size": int(batch_size),
                    "epochs": int(epochs),
                    "lr": float(lr),
                    "tv_weight": float(tv_weight),
                    "base": int(base),
                    "device": dev,
                },
            },
            f,
            indent=2,
            default=str,
        )

    import time

    model.train()
    step = 0
    n_batches = (len(ds) + int(batch_size) - 1) // int(batch_size)
    print(f"Training: {len(ds)} samples, {n_batches} batches/epoch, {epochs} epochs, device={dev}", flush=True)
    for ep in range(int(epochs)):
        t_ep = time.perf_counter()
        losses = []
        for bi, (xb, yb, _meta) in enumerate(dl):
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)  # (B,1)

            pred_patch = model(xb)  # (B,1,H,W)
            pred_center = center_pixel(pred_patch)  # (B,1)

            loss_fit = huber(pred_center, yb)
            loss_tv = tv_loss(pred_patch)
            loss = loss_fit + float(tv_weight) * loss_tv

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu().item()))
            step += 1

        elapsed = time.perf_counter() - t_ep
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"  epoch {ep:3d}/{epochs}  loss={mean_loss:.6f}  {elapsed:.1f}s", flush=True)

        # Epoch checkpoint
        ckpt = {
            "epoch": ep,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "loss_mean": mean_loss,
        }
        torch.save(ckpt, os.path.join(out_dir, f"ckpt_epoch_{ep:03d}.pt"))

    final_path = os.path.join(out_dir, "ckpt_final.pt")
    torch.save({"model_state": model.state_dict(), "dataset_in_channels": ds.in_channels}, final_path)
    return final_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a patch-based RTMA humidity correction U-Net (MVP).")
    p.add_argument("--patch-index", required=True, help="Parquet from prep/build_rtma_patch_index.py")
    p.add_argument("--tif-root", required=True, help="Directory containing RTMA_YYYYMMDD.tif daily COGs")
    p.add_argument("--out-dir", required=True, help="Output directory for checkpoints/config")
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tv-weight", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None, help="Override device (e.g., cuda, cpu)")
    p.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True,
                   help="Pre-load all COGs into memory (default: True; use --no-preload to disable)")
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    train(
        patch_index=a.patch_index,
        tif_root=a.tif_root,
        out_dir=a.out_dir,
        patch_size=int(a.patch_size),
        batch_size=int(a.batch_size),
        epochs=int(a.epochs),
        lr=float(a.lr),
        tv_weight=float(a.tv_weight),
        base=int(a.base),
        num_workers=int(a.num_workers),
        device=a.device,
        preload=bool(a.preload),
    )


if __name__ == "__main__":
    main()
