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
    "rtma_weather": ("tmp_c", "ugrd", "vgrd", "pres_kpa", "tcdc_pct", "prcp_mm"),
    "rtma_humidity": ("dpt_c", "ea_kpa"),
    "rtma_all": (
        "tmp_c",
        "dpt_c",
        "ugrd",
        "vgrd",
        "pres_kpa",
        "tcdc_pct",
        "prcp_mm",
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
_RTMA_CHANNELS = set(FEATURE_GROUPS["rtma_all"])
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
        default_factory=lambda: ["rtma_all", "terrain", "rsun", "landsat"]
    )

    # Model / training
    base: int = 32
    lr: float = 3e-4
    tv_weight: float = 1e-3
    batch_size: int = 64
    epochs: int = 5

    # Data
    patch_size: int = 64
    val_frac: float = 0.2
    seed: int = 42
    target_col: str = "delta_log_ea"
    start_date: str | None = None
    end_date: str | None = None

    # Paths
    patch_index: str = ""
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
