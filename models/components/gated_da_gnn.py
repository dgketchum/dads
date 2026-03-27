"""
Gated DA GNN for da-graph-v1.

Splits source information into context (for attention scoring) and payload
(for message values). The background prediction pathway is source-free.
A bounded gate controls how much DA residual is added.

Information flow:
  1. Query encoder → h_query
  2. Query→query propagation (source-free) → h_query_bg
  3. Source context encoder → h_src_ctx (for attention scoring)
  4. Source payload encoder → h_src_pay (for message values)
  5. Source→query payload attention → da_ctx
  6. Background head(h_query_local, h_query_bg) → bg_pred
  7. DA gate(h_query_bg, da_ctx) * DA head(h_query_bg, da_ctx) → da_residual
  8. pred = bg_pred + da_residual
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class PayloadGatedAttention(MessagePassing):
    """Edge-gated attention that scores from context but propagates payload.

    Attention score: MLP([h_dst_i; h_src_context_j; edge_attr_ji])
    Message value: alpha * h_src_payload_j
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__(aggr="add")
        self.hidden_dim = hidden_dim
        self.att_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_src_context: Tensor,
        h_src_payload: Tensor,
        h_dst: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        if edge_index.numel() == 0:
            return h_dst.new_zeros((h_dst.size(0), self.hidden_dim))
        return self.propagate(
            edge_index,
            h_src_context=h_src_context,
            h_src_payload=h_src_payload,
            h_dst=h_dst,
            edge_attr=edge_attr,
        )

    def message(
        self,
        h_dst_i: Tensor,
        h_src_context_j: Tensor,
        h_src_payload_j: Tensor,
        edge_attr: Tensor,
        index: Tensor,
    ) -> Tensor:
        # Score from context + query + geometry
        att_input = torch.cat([h_dst_i, h_src_context_j, edge_attr], dim=-1)
        att_score = self.att_mlp(att_input).squeeze(-1)
        alpha = softmax(att_score, index)
        # Propagate payload only
        return alpha.unsqueeze(-1) * h_src_payload_j


class CrossEdgeGatedAttention(MessagePassing):
    """Standard edge-gated attention for query→query propagation."""

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__(aggr="add")
        self.hidden_dim = hidden_dim
        self.att_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x_src: Tensor, x_dst: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tensor:
        if edge_index.numel() == 0:
            return x_dst.new_zeros((x_dst.size(0), self.hidden_dim))
        return self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr)

    def message(
        self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, index: Tensor
    ) -> Tensor:
        att_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        att_score = self.att_mlp(att_input).squeeze(-1)
        alpha = softmax(att_score, index)
        return alpha.unsqueeze(-1) * x_j


class GatedDAGNN(nn.Module):
    """Gated DA GNN with source-free background pathway.

    Parameters
    ----------
    query_node_dim : int
        Query node feature dimension (37 for core-graph-v0).
    source_context_dim : int
        Source context feature dimension (37, same as query).
    source_payload_dim : int
        Source payload feature dimension (4: deltas + valid flags).
    edge_dim : int
        Edge feature dimension.
    hidden_dim : int
        Hidden state dimension for all encoders.
    n_hops : int
        Number of query→query propagation hops.
    out_dim : int
        Output dimension (1 for scalar, 2 for 2-head).
    dropout : float
        Dropout probability.
    da_gate_init_bias : float
        Initial bias for the DA gate (negative → conservative start).
    """

    def __init__(
        self,
        query_node_dim: int,
        source_context_dim: int,
        source_payload_dim: int,
        edge_dim: int = 7,
        hidden_dim: int = 64,
        n_hops: int = 1,
        out_dim: int = 1,
        dropout: float = 0.3,
        da_gate_init_bias: float = -2.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_hops = n_hops

        # Encoders
        self.query_encoder = nn.Sequential(
            nn.Linear(query_node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.source_context_encoder = nn.Sequential(
            nn.Linear(source_context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.source_payload_encoder = nn.Sequential(
            nn.Linear(source_payload_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Query→query propagation (source-free background track)
        self.qq_layers = nn.ModuleList(
            [CrossEdgeGatedAttention(hidden_dim, edge_dim) for _ in range(n_hops)]
        )
        self.qq_merge = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(n_hops)
            ]
        )

        # Source→query payload attention
        self.sq_attention = PayloadGatedAttention(hidden_dim, edge_dim)

        # Background prediction head (source-free)
        self.background_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        # DA gate: sigmoid output, initialized conservatively
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # Initialize gate bias negative so DA starts small
        with torch.no_grad():
            self.gate_mlp[-1].bias.fill_(da_gate_init_bias)

        # DA residual head
        self.da_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        data,
        disable_payload: bool = False,
        source_edge_dropout: float = 0.0,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Returns (pred, da_gate_mean) where da_gate_mean is logged for monitoring.
        """
        query_x = data["query"].x
        source_context_x = data["source"].context_x
        source_payload_x = data["source"].payload_x

        # Encode
        h_query = self.query_encoder(query_x)
        h_src_ctx = self.source_context_encoder(source_context_x)
        h_src_pay = self.source_payload_encoder(source_payload_x)

        # DA-off: zero payload encoding
        if disable_payload:
            h_src_pay = torch.zeros_like(h_src_pay)

        h_query_local = h_query

        # Background track: query→query only (source-free)
        qq_edge_index = data["query", "neighbors", "query"].edge_index
        qq_edge_attr = data["query", "neighbors", "query"].edge_attr

        for g2g, merge in zip(self.qq_layers, self.qq_merge):
            ctx = g2g(h_query, h_query, qq_edge_index, qq_edge_attr)
            h_query = merge(torch.cat([h_query, ctx], dim=-1))
        h_query_bg = h_query

        # DA track: source→query payload messages
        sq_edge_index = data["source", "influences", "query"].edge_index
        sq_edge_attr = data["source", "influences", "query"].edge_attr

        # Source-edge dropout during training
        if self.training and source_edge_dropout > 0 and sq_edge_index.numel() > 0:
            n_edges = sq_edge_index.shape[1]
            keep = (
                torch.rand(n_edges, device=sq_edge_index.device) > source_edge_dropout
            )
            sq_edge_index = sq_edge_index[:, keep]
            sq_edge_attr = sq_edge_attr[keep]

        da_ctx = self.sq_attention(
            h_src_ctx, h_src_pay, h_query_bg, sq_edge_index, sq_edge_attr
        )

        # Background prediction (source-free)
        bg_pred = self.background_head(torch.cat([h_query_local, h_query_bg], dim=-1))

        # Gated DA residual
        gate_input = torch.cat([h_query_bg, da_ctx], dim=-1)
        da_gate = torch.sigmoid(self.gate_mlp(gate_input))
        da_residual = self.da_head(gate_input)

        pred = bg_pred + da_gate * da_residual

        return pred, da_gate.mean()
