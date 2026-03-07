"""Thin wrapper for checkpoint-loading backward compatibility."""

from models.components.lit_dads_gnn import LitDadsGNN


class LitWindGNN(LitDadsGNN):
    def __init__(self, **kwargs):
        kwargs.setdefault("task", "wind")
        super().__init__(**kwargs)
