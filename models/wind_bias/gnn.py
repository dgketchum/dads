"""
Wind bias correction GNN — edge-gated neighbor attention.

Architecture:
  1) Node encoder:  MLP(x_i) -> h_i  (hidden_dim)
  2) Edge-gated attention:
     a_ji = MLP_att([h_i; h_j; e_ji]) -> scalar
     alpha_ji = softmax_j(a_ji)
  3) Context: c_i = sum_j(alpha_ji * h_j)
  4) Output: MLP_out([h_i; c_i]) -> (delta_w_par, delta_w_perp)

Optionally runs a second attention hop.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class EdgeGatedAttention(MessagePassing):
    """Single hop of edge-gated attention."""

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__(aggr="add")
        self.att_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.propagate(edge_index, h=h, edge_attr=edge_attr)

    def message(
        self, h_i: Tensor, h_j: Tensor, edge_attr: Tensor, index: Tensor
    ) -> Tensor:
        att_input = torch.cat([h_i, h_j, edge_attr], dim=-1)
        att_score = self.att_mlp(att_input).squeeze(-1)
        alpha = softmax(att_score, index)
        return alpha.unsqueeze(-1) * h_j


class WindBiasGNN(nn.Module):
    """Edge-gated neighbor attention GNN for wind bias correction.

    Parameters
    ----------
    node_dim : int
        Number of input node features.
    edge_dim : int
        Number of edge features.
    hidden_dim : int
        Hidden dimension.
    n_hops : int
        Number of attention hops (1 or 2).
    use_graph : bool
        If False, skip attention — pure local MLP (experiment step 1).
    out_dim : int
        Number of output scalars per node (default 2 for wind par/perp).
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 7,
        hidden_dim: int = 64,
        n_hops: int = 1,
        use_graph: bool = True,
        out_dim: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_graph = use_graph
        self.n_hops = n_hops

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if use_graph:
            self.attention_layers = nn.ModuleList(
                [EdgeGatedAttention(hidden_dim, edge_dim) for _ in range(n_hops)]
            )
            # After each hop, merge h_i and context
            self.merge_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    )
                    for _ in range(n_hops)
                ]
            )
            out_input_dim = 2 * hidden_dim
        else:
            out_input_dim = hidden_dim

        self.output_head = nn.Sequential(
            nn.Linear(out_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        h = self.node_encoder(x)

        if self.use_graph and edge_index is not None:
            h_local = h
            for att, merge in zip(self.attention_layers, self.merge_layers):
                ctx = att(h, edge_index, edge_attr)
                h = merge(torch.cat([h, ctx], dim=-1))
            out = self.output_head(torch.cat([h_local, h], dim=-1))
        else:
            out = self.output_head(h)

        return out
