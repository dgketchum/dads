from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

import tomllib

import tomli_w

# ── Feature group definitions ──────────────────────────────────────────
# Each group maps to a set of individual channel names used by the dataset.

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    # URMA-based groups (11-band daily COGs from prep/build_urma_1km.py)
    "urma_weather": (
        "tmp_c",
        "tmax_c",
        "tmin_c",
        "ugrd",
        "vgrd",
        "gust",
        "pres_kpa",
        "tcdc_pct",
    ),
    "urma_humidity": ("dpt_c", "spfh", "ea_kpa"),
    "urma_all": (
        "tmp_c",
        "tmax_c",
        "tmin_c",
        "dpt_c",
        "ugrd",
        "vgrd",
        "gust",
        "spfh",
        "pres_kpa",
        "tcdc_pct",
        "ea_kpa",
    ),
    "terrain": (
        "elevation",
        "slope",
        "aspect_sin",
        "aspect_cos",
        "tpi_4",
        "tpi_10",
    ),
    "rsun": ("rsun",),
    "landsat": (
        "ls_b2",
        "ls_b3",
        "ls_b4",
        "ls_b5",
        "ls_b6",
        "ls_b7",
        "ls_b10",
    ),
}

# All individual channel names recognised by the system.
ALL_CHANNELS = set()
for _grp in FEATURE_GROUPS.values():
    ALL_CHANNELS.update(_grp)


def resolve_channels(features: list[str]) -> tuple[str, ...]:
    """Expand a mixed list of group names and individual channel names.

    Returns a deduped tuple preserving first-seen order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for feat in features:
        if feat in FEATURE_GROUPS:
            for ch in FEATURE_GROUPS[feat]:
                if ch not in seen:
                    seen.add(ch)
                    result.append(ch)
        elif feat in ALL_CHANNELS:
            if feat not in seen:
                seen.add(feat)
                result.append(feat)
        else:
            raise ValueError(
                f"Unknown feature '{feat}'. "
                f"Valid groups: {sorted(FEATURE_GROUPS)} | "
                f"Valid channels: {sorted(ALL_CHANNELS)}"
            )
    return tuple(result)


# ── Which channels belong to which data source ─────────────────────────
_RTMA_CHANNELS = set(FEATURE_GROUPS["urma_all"])
_TERRAIN_CHANNELS = set(FEATURE_GROUPS["terrain"])
_RSUN_CHANNELS = set(FEATURE_GROUPS["rsun"])
_LANDSAT_CHANNELS = set(FEATURE_GROUPS["landsat"])


@dataclass
class ExperimentConfig:
    """Everything needed to reproduce a training run."""

    # Identity
    name: str = "default"
    description: str = ""

    # Features
    features: list[str] = field(
        default_factory=lambda: ["urma_all", "terrain", "rsun", "landsat"]
    )

    # Model / training
    base: int = 32
    lr: float = 3e-4
    tv_weight: float = 1e-3
    batch_size: int = 64
    epochs: int = 5

    # Multi-task
    n_heads: int = 1
    target_cols: list[str] | None = None  # overrides target_col when set
    task_weights: list[float] | None = None
    physics_weight: float = 0.0
    task: str = "ea"  # primary task for single-head mode: ea|tmax
    pair_head_idx: int | None = None

    # Data
    patch_size: int = 64
    val_frac: float = 0.2
    seed: int = 42
    val_mgrs_tiles: list[str] | None = None
    target_col: str = "delta_log_ea"
    start_date: str | None = None
    end_date: str | None = None
    use_pairwise_loss: bool = False

    # Paths
    patch_index: str = ""
    pair_index: str | None = None
    tif_root: str = ""
    out_dir: str = ""
    terrain_tif: str | None = None
    rsun_tif: str | None = None
    landsat_tif: str | None = None

    # Runtime
    device: str | None = None
    num_workers: int = 2
    preload: bool = True
    decoded: bool = False
    model: str = "unet"
    pair_k: int = 8
    pair_dmax_km: float = 35.0
    pair_loss_weight: float = 0.3
    pair_sample_per_batch: int | None = None
    pair_distance_bins: list[float] | None = None

    # ── Derived properties ──────────────────────────────────────────────

    @property
    def rtma_channels(self) -> tuple[str, ...]:
        """RTMA channels selected by the current feature list."""
        resolved = resolve_channels(self.features)
        return tuple(ch for ch in resolved if ch in _RTMA_CHANNELS)

    @property
    def use_terrain(self) -> bool:
        resolved = resolve_channels(self.features)
        return any(ch in _TERRAIN_CHANNELS for ch in resolved)

    @property
    def use_rsun(self) -> bool:
        resolved = resolve_channels(self.features)
        return any(ch in _RSUN_CHANNELS for ch in resolved)

    @property
    def use_landsat(self) -> bool:
        resolved = resolve_channels(self.features)
        return any(ch in _LANDSAT_CHANNELS for ch in resolved)

    # ── Serialisation ───────────────────────────────────────────────────

    def save_toml(self, path: str) -> None:
        d = asdict(self)
        # tomli_w cannot serialise None; drop None-valued keys.
        d = {k: v for k, v in d.items() if v is not None}
        with open(path, "wb") as f:
            tomli_w.dump(d, f)

    @classmethod
    def from_toml(cls, path: str) -> ExperimentConfig:
        with open(path, "rb") as f:
            d = tomllib.load(f)
        return cls(**d)

    def config_hash(self) -> str:
        """Short deterministic hash of the config (for dedup / naming)."""
        d = asdict(self)
        d.pop("description", None)
        blob = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def log_experiment(config: ExperimentConfig, metrics: dict) -> None:
    """Append config + final metrics to a JSONL registry in out_dir."""
    registry = os.path.join(config.out_dir, "experiment_registry.jsonl")
    entry = {
        "timestamp": datetime.now().isoformat(),
        "config_hash": config.config_hash(),
        "config": asdict(config),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    with open(registry, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"Experiment logged to {registry}", flush=True)
