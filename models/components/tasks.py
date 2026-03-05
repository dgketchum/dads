from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskAdapter:
    """Task metadata used by training/evaluation plumbing."""

    name: str
    target_col: str
    metric_prefix: str
    default_head_idx: int
    supports_ea_space_metrics: bool = False
    supports_cc_physics: bool = False


_TASKS: dict[str, TaskAdapter] = {
    "ea": TaskAdapter(
        name="ea",
        target_col="delta_log_ea",
        metric_prefix="ea",
        default_head_idx=0,
        supports_ea_space_metrics=True,
        supports_cc_physics=True,
    ),
    "tmax": TaskAdapter(
        name="tmax",
        target_col="delta_tmax",
        metric_prefix="tmax",
        default_head_idx=1,
        supports_ea_space_metrics=False,
        supports_cc_physics=False,
    ),
    "tmin": TaskAdapter(
        name="tmin",
        target_col="delta_tmin",
        metric_prefix="tmin",
        default_head_idx=2,
        supports_ea_space_metrics=False,
        supports_cc_physics=False,
    ),
    "wind": TaskAdapter(
        name="wind",
        target_col="delta_wind",
        metric_prefix="wind",
        default_head_idx=3,
        supports_ea_space_metrics=False,
        supports_cc_physics=False,
    ),
}


def get_task_adapter(task: str) -> TaskAdapter:
    key = str(task).strip().lower()
    if key not in _TASKS:
        raise ValueError(f"Unknown task '{task}'. Valid tasks: {sorted(_TASKS)}")
    return _TASKS[key]
