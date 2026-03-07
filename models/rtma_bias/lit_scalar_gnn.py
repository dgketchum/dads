"""Thin wrapper for checkpoint-loading backward compatibility."""

from models.components.lit_dads_gnn import LitDadsGNN


class LitScalarGNN(LitDadsGNN):
    def __init__(self, **kwargs):
        kwargs.setdefault("task", "scalar")
        super().__init__(**kwargs)
