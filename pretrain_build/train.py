"""
Pre-training loop for DADS on gridded data.

Key differences from models/dads/train.py:
    1. Rebuilds graph structure each epoch via EpochSampler
    2. Uses PretrainDataset instead of DadsDataset
    3. Saves checkpoints compatible with fine-tuning on observations
    4. Supports regional holdout validation for spatial generalization

The pre-training objective is to learn spatial meteorological transfer
patterns from dense gridded data that can later be fine-tuned on sparse
observational data.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from torch.utils.data import DataLoader

from models.dads.dads_gnn import DadsMetGNN
from models.scalers import MinMaxScaler
from pretrain_build.config import PretrainConfig, GridSource
from pretrain_build.grid_index import GridIndex
from pretrain_build.sampler import EpochSampler, ValidationSampler
from pretrain_build.sequences import SequenceExtractor, CachedSequenceExtractor
from pretrain_build.dataset import PretrainDataset, pretrain_collate_fn


def pretrain(
    config: PretrainConfig,
    output_dir: Path,
    variable: str = "tmax",
    max_epochs: int = 50,
    batch_size: int = 512,
    num_workers: int = 8,
    learning_rate: float = 1e-3,
    hidden_dim: int = 256,
    tcn_channels: int = 128,
    tcn_out_dim: int = 256,
    val_holdout_bounds: Optional[tuple] = None,
    resume_checkpoint: Optional[str] = None,
    windows_per_cell: int = 100,
    val_windows_per_cell: int = 20,
    seed: int = 42,
) -> DadsMetGNN:
    """
    Pre-train DADS on gridded data.

    Workflow:
        1. Load/build spatial index
        2. Initialize model and sampler
        3. For each epoch:
            a. Generate new graph sample (if resample_each_epoch)
            b. Build dataset from sample
            c. Train one epoch
        4. Save checkpoint for fine-tuning

    Args:
        config: PretrainConfig with grid sources and parameters
        output_dir: Directory for outputs (checkpoints, logs)
        variable: Target variable (e.g., 'tmax', 'tmin', 'rsds')
        max_epochs: Maximum training epochs
        batch_size: Training batch size
        num_workers: DataLoader workers
        learning_rate: Initial learning rate
        hidden_dim: GNN hidden dimension
        tcn_channels: TCN channel width
        tcn_out_dim: TCN output dimension
        val_holdout_bounds: Optional (w, s, e, n) for regional validation
        resume_checkpoint: Optional path to resume from
        windows_per_cell: Temporal windows per cell for training
        val_windows_per_cell: Temporal windows per cell for validation
        seed: Random seed

    Returns:
        Trained DadsMetGNN model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(seed)

    # Load or build spatial index
    index_path = output_dir / "grid_index.zarr"
    if not index_path.exists():
        if not config.sources:
            raise ValueError("No grid sources configured")

        print(f"[Pretrain] Building spatial index from {config.sources[0].name}")
        grid_index = GridIndex.build_from_source(
            config.sources[0],
            config,
            index_path,
            dem_path=config.dem_path,
        )
    else:
        print(f"[Pretrain] Loading spatial index from {index_path}")
        grid_index = GridIndex(index_path)

    print(f"[Pretrain] Grid index contains {len(grid_index)} cells")

    # Initialize sequence extractor
    sources = {s.name: s for s in config.sources}
    seq_extractor = CachedSequenceExtractor(sources, config, cache_size=10000)

    # Build or load scaler from gridded data
    scaler_path = output_dir / f"{variable}_scaler.json"
    if scaler_path.exists():
        print(f"[Pretrain] Loading scaler from {scaler_path}")
        with open(scaler_path, "r") as f:
            scaler_params = json.load(f)
        scaler = MinMaxScaler()
        scaler.bias = np.array(scaler_params["bias"]).reshape(1, -1)
        scaler.scale = np.array(scaler_params["scale"]).reshape(1, -1)
    else:
        print("[Pretrain] Building scaler from gridded data")
        scaler = _build_scaler_from_grid(
            grid_index, seq_extractor, config, variable, seed=seed
        )
        # Save scaler
        scaler_params = {
            "bias": scaler.bias.tolist(),
            "scale": scaler.scale.tolist(),
        }
        with open(scaler_path, "w") as f:
            json.dump(scaler_params, f)

    # Initialize sampler
    if val_holdout_bounds:
        print(f"[Pretrain] Using regional holdout validation: {val_holdout_bounds}")
        val_sampler = ValidationSampler(grid_index, config, val_holdout_bounds)
        epoch_sampler = EpochSampler(grid_index, config)
    else:
        epoch_sampler = EpochSampler(grid_index, config)
        val_sampler = None

    # Generate initial sample to determine dimensions
    print("[Pretrain] Generating initial sample for dimension inference")
    initial_sample = epoch_sampler.new_epoch(seed=seed)
    temp_dataset = PretrainDataset(
        initial_sample,
        grid_index,
        seq_extractor,
        config,
        variable,
        scaler,
        windows_per_cell=1,
        seed=seed,
        inject_missingness=False,
    )

    # Infer dimensions from dataset
    terrain_dim = temp_dataset.terrain_dim
    emb_dim = temp_dataset.emb_dim
    exog_dim = temp_dataset.exog_dim
    seq_in_channels = temp_dataset.seq_in_channels
    edge_dim = temp_dataset.edge_dim

    print(
        f"[Pretrain] Dimensions: terrain={terrain_dim}, emb={emb_dim}, "
        f"exog={exog_dim}, seq_channels={seq_in_channels}, edge={edge_dim}"
    )

    # Initialize model
    model = DadsMetGNN(
        hidden_dim=hidden_dim,
        n_nodes=config.n_neighbors,
        output_dim=1,
        edge_attr_dim=edge_dim,
        emb_dim=emb_dim,
        exog_dim=exog_dim,
        tcn_in_channels=seq_in_channels,
        tcn_out_dim=tcn_out_dim,
        tcn_channels=tcn_channels,
        learning_rate=learning_rate,
        scaler=scaler,
        column_indices=(
            0,
            1,
            2,
            seq_in_channels,
        ),  # (y_idx, comparator_idx, feat_start, tensor_width)
    )

    if resume_checkpoint:
        print(f"[Pretrain] Resuming from {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="pretrain-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        default_root_dir=output_dir,
        enable_progress_bar=True,
    )

    # Training loop with epoch-wise resampling
    print(f"[Pretrain] Starting training for {max_epochs} epochs")

    for epoch in range(max_epochs):
        print(f"\n[Pretrain] Epoch {epoch + 1}/{max_epochs}")

        # Generate new graph samples
        if config.resample_each_epoch or epoch == 0:
            train_sample = epoch_sampler.new_epoch(seed=seed + epoch * 2)
            if val_sampler:
                val_sample = val_sampler.create_val_sample(seed=seed + epoch * 2 + 1)
            else:
                val_sample = epoch_sampler.new_epoch(seed=seed + epoch * 2 + 1)

            print(f"  Train sample: {len(train_sample)} targets")
            print(f"  Val sample: {len(val_sample)} targets")

        # Build datasets
        train_dataset = PretrainDataset(
            train_sample,
            grid_index,
            seq_extractor,
            config,
            variable,
            scaler,
            windows_per_cell=windows_per_cell,
            seed=seed + epoch,
            inject_missingness=True,
        )

        val_dataset = PretrainDataset(
            val_sample,
            grid_index,
            seq_extractor,
            config,
            variable,
            scaler,
            windows_per_cell=val_windows_per_cell,
            seed=seed + epoch + 10000,
            inject_missingness=False,
        )

        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=pretrain_collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=pretrain_collate_fn,
            pin_memory=True,
        )

        # Train one epoch
        trainer.fit_loop.max_epochs = epoch + 1
        trainer.fit(model, train_loader, val_loader)

        # Clear sequence cache between epochs
        seq_extractor.clear_cache()

        # Check for early stopping
        if trainer.should_stop:
            print("[Pretrain] Early stopping triggered")
            break

    # Save final checkpoint with metadata
    final_ckpt = output_dir / "pretrain_final.ckpt"
    trainer.save_checkpoint(final_ckpt)
    print(f"[Pretrain] Saved final checkpoint to {final_ckpt}")

    # Save config for reproducibility
    config_dict = {
        "bounds": config.bounds,
        "n_cells_per_epoch": config.n_cells_per_epoch,
        "sampling_strategy": config.sampling_strategy,
        "n_neighbors": config.n_neighbors,
        "seq_len": config.seq_len,
        "date_range": config.date_range,
        "variable": variable,
        "hidden_dim": hidden_dim,
        "tcn_channels": tcn_channels,
        "tcn_out_dim": tcn_out_dim,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "seed": seed,
    }
    with open(output_dir / "pretrain_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Clean up
    seq_extractor.close()

    return model


def _build_scaler_from_grid(
    grid_index: GridIndex,
    seq_extractor: SequenceExtractor,
    config: PretrainConfig,
    variable: str,
    n_samples: int = 5000,
    seed: int = 42,
) -> MinMaxScaler:
    """
    Fit MinMaxScaler on sample of gridded data.

    Args:
        grid_index: Spatial index
        seq_extractor: Sequence extractor
        config: Configuration
        variable: Target variable
        n_samples: Number of samples to use for fitting
        seed: Random seed

    Returns:
        Fitted MinMaxScaler
    """
    rng = np.random.default_rng(seed)

    # Sample cells
    n_cells = min(1000, len(grid_index))
    cell_indices = rng.choice(len(grid_index.cell_ids), size=n_cells, replace=False)

    # Sample dates
    start_str, end_str = config.date_range
    start_ord = np.datetime64(start_str, "D").astype(int) + config.seq_len
    end_ord = np.datetime64(end_str, "D").astype(int)

    sequences = []
    samples_collected = 0

    for ci in cell_indices:
        if samples_collected >= n_samples:
            break

        lat, lon = grid_index.coords[ci]
        terrain = grid_index.terrain[ci]
        row, col = grid_index.grid_indices[ci]

        # Sample a few dates per cell
        n_dates = min(10, n_samples - samples_collected)
        end_dates = rng.integers(start_ord, end_ord, size=n_dates)

        for ed in end_dates:
            end_date = np.datetime64(ed, "D")
            seq, valid = seq_extractor.extract_sequence(
                lat,
                lon,
                end_date,
                variable,
                terrain,
                row=int(row),
                col=int(col),
            )
            if valid:
                sequences.append(seq)
                samples_collected += 1
                if samples_collected >= n_samples:
                    break

    if not sequences:
        raise ValueError("Could not extract any valid sequences for scaler fitting")

    print(f"[Scaler] Collected {len(sequences)} sequences for fitting")

    # Stack and fit scaler
    data = np.vstack(sequences)
    scaler = MinMaxScaler()
    scaler.fit(data)

    return scaler


def load_pretrained_model(
    checkpoint_path: str,
    config_path: Optional[str] = None,
) -> DadsMetGNN:
    """
    Load a pre-trained model for fine-tuning.

    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Optional path to config JSON

    Returns:
        Loaded DadsMetGNN model
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract model hyperparameters from checkpoint
    hparams = checkpoint.get("hyper_parameters", {})

    # Create model
    model = DadsMetGNN(
        hidden_dim=hparams.get("hidden_dim", 256),
        n_nodes=hparams.get("n_nodes", 10),
        output_dim=hparams.get("output_dim", 1),
        edge_attr_dim=hparams.get("edge_attr_dim", 20),
        emb_dim=hparams.get("emb_dim", 7),
        exog_dim=hparams.get("exog_dim", 10),
        tcn_in_channels=hparams.get("tcn_in_channels", 11),
        tcn_out_dim=hparams.get("tcn_out_dim", 256),
        tcn_channels=hparams.get("tcn_channels", 128),
        learning_rate=hparams.get("learning_rate", 1e-3),
    )

    # Load weights
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    return model


def prepare_for_finetuning(
    pretrained_model: DadsMetGNN,
    observation_scaler: MinMaxScaler,
    freeze_encoder: bool = False,
    differential_lr: bool = True,
    base_lr: float = 1e-4,
) -> Dict[str, Any]:
    """
    Prepare a pre-trained model for fine-tuning on observations.

    Args:
        pretrained_model: Pre-trained DadsMetGNN
        observation_scaler: Scaler fit on observation data
        freeze_encoder: Whether to freeze TCN and early GNN layers
        differential_lr: Whether to use different LR for pretrained layers
        base_lr: Base learning rate

    Returns:
        Dict with model and optimizer configuration
    """
    # Update scaler
    pretrained_model.scaler = observation_scaler

    if freeze_encoder:
        # Freeze TCN and first GNN layer
        for param in pretrained_model.tcn.parameters():
            param.requires_grad = False
        for param in pretrained_model.gnn_layer.parameters():
            param.requires_grad = False

    if differential_lr:
        # Create parameter groups with different learning rates
        pretrained_params = []
        new_params = []

        pretrained_names = {"tcn", "gnn_layer", "node_proj", "pre_norm"}

        for name, param in pretrained_model.named_parameters():
            if not param.requires_grad:
                continue

            is_pretrained = any(pn in name for pn in pretrained_names)
            if is_pretrained:
                pretrained_params.append(param)
            else:
                new_params.append(param)

        optimizer_groups = [
            {"params": pretrained_params, "lr": base_lr * 0.1},
            {"params": new_params, "lr": base_lr},
        ]
    else:
        optimizer_groups = pretrained_model.parameters()

    return {
        "model": pretrained_model,
        "optimizer_groups": optimizer_groups,
        "base_lr": base_lr,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-train DADS on gridded data")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--prism_zarr", type=str, default=None, help="Path to PRISM Zarr store"
    )
    parser.add_argument(
        "--gridmet_zarr", type=str, default=None, help="Path to GridMET Zarr store"
    )
    parser.add_argument("--dem_path", type=str, default=None, help="Path to DEM raster")
    parser.add_argument("--variable", type=str, default="tmax", help="Target variable")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Build config
    sources = []
    if args.prism_zarr:
        sources.append(
            GridSource(
                name="prism",
                zarr_path=Path(args.prism_zarr),
                variables=["tmax", "tmin", "ppt"],
                resolution_deg=0.04166667,
            )
        )
    if args.gridmet_zarr:
        sources.append(
            GridSource(
                name="gridmet",
                zarr_path=Path(args.gridmet_zarr),
                variables=["srad", "vpd", "vs"],
                resolution_deg=0.04166667,
            )
        )

    if not sources:
        print(
            "Error: At least one data source (--prism_zarr or --gridmet_zarr) required"
        )
        exit(1)

    config = PretrainConfig(
        sources=sources,
        dem_path=Path(args.dem_path) if args.dem_path else None,
        cache_dir=Path(args.output_dir),
        n_cells_per_epoch=8000,
        n_neighbors=10,
        seq_len=12,
    )

    model = pretrain(
        config=config,
        output_dir=Path(args.output_dir),
        variable=args.variable,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    print("Pre-training complete!")
