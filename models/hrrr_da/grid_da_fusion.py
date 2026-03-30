"""
Sparse station→pixel DA fusion for the grid backbone (Stage C).

For each source station at pixel (r, c) within a patch, scatter payload
messages to all query pixels within ``support_radius_px``.  Attention is
normalised per query pixel over incoming source stations (same convention
as ``PayloadGatedAttention`` in ``gated_da_gnn.py``).
"""

from __future__ import annotations


import torch
from torch import Tensor, nn
from torch_geometric.utils import softmax as pyg_softmax


class GridDAFusion(nn.Module):
    """Sparse station→pixel DA context injection.

    Parameters
    ----------
    grid_latent_dim : int
        Channel depth of the UNet decoder output (``base`` in UNetSmall).
    source_ctx_dim : int
        Dimension of per-source context features (extracted from ``x_patch``).
    source_pay_dim : int
        Dimension of per-source payload (innovations + valid flags).
    hidden_dim : int
        Internal hidden size for encoders and attention MLP.
    support_radius_px : int
        Maximum pixel distance for source→query edges (1 px = 1 km).
    geom_dim : int
        Geometry feature dimension per edge (distance, sin, cos, delta_elev).
    """

    def __init__(
        self,
        grid_latent_dim: int,
        source_ctx_dim: int,
        source_pay_dim: int,
        hidden_dim: int = 32,
        support_radius_px: int = 16,
        geom_dim: int = 4,
    ):
        super().__init__()
        self.support_radius_px = support_radius_px
        self.hidden_dim = hidden_dim

        self.source_ctx_enc = nn.Sequential(
            nn.Linear(source_ctx_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.source_pay_enc = nn.Sequential(
            nn.Linear(source_pay_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.query_proj = nn.Sequential(
            nn.Linear(grid_latent_dim, hidden_dim),
            nn.ReLU(),
        )
        # Attention: score from (h_query, h_src_ctx, geometry)
        self.att_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + geom_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        F_grid: Tensor,
        src_rows: Tensor,
        src_cols: Tensor,
        src_ctx: Tensor,
        src_pay: Tensor,
        src_valid: Tensor,
        raw_elev_patch: Tensor,
        src_elev: Tensor,
    ) -> Tensor:
        """Compute DA context map.

        Parameters
        ----------
        F_grid : (B, C_latent, H, W)
            Grid backbone latent features.
        src_rows, src_cols : (B, N_src)
            Pixel positions of source stations within the patch (padded).
        src_ctx : (B, N_src, C_ctx)
            Source context features.
        src_pay : (B, N_src, C_pay)
            Source payload (innovations + valid flags).
        src_valid : (B, N_src)
            Bool mask — which source slots are real (not padding).
        raw_elev_patch : (B, 1, H, W)
            Raw elevation in metres (un-normalised).
        src_elev : (B, N_src)
            Raw elevation at source pixels.

        Returns
        -------
        da_ctx : (B, hidden_dim, H, W)
            Dense DA context map (zero where no source is within radius).
        coverage_mask : (B, 1, H, W)
            Binary mask: 1.0 where at least one source contributed, else 0.0.
            Used by ``LitGridDA`` to hard-zero the DA residual outside the
            support disk and when no sources are valid.
        """
        B, C, H, W = F_grid.shape
        device = F_grid.device
        da_ctx = F_grid.new_zeros(B, self.hidden_dim, H, W)
        coverage_mask = F_grid.new_zeros(B, 1, H, W)
        R = self.support_radius_px

        for b in range(B):
            valid_mask = src_valid[b]
            if not valid_mask.any():
                continue

            n_valid = int(valid_mask.sum())
            s_rows = src_rows[b][valid_mask].long()
            s_cols = src_cols[b][valid_mask].long()

            # Encode sources
            h_ctx = self.source_ctx_enc(src_ctx[b][valid_mask])  # (n_valid, H_dim)
            h_pay = self.source_pay_enc(src_pay[b][valid_mask])  # (n_valid, H_dim)
            s_elev = src_elev[b][valid_mask]  # (n_valid,)

            # Build edge list: for each source, enumerate query pixels in radius
            edge_src_list = []
            edge_qr_list = []
            edge_qc_list = []
            for j in range(n_valid):
                sr, sc = int(s_rows[j]), int(s_cols[j])
                r_lo = max(0, sr - R)
                r_hi = min(H, sr + R + 1)
                c_lo = max(0, sc - R)
                c_hi = min(W, sc + R + 1)
                qr = torch.arange(r_lo, r_hi, device=device)
                qc = torch.arange(c_lo, c_hi, device=device)
                qr_grid, qc_grid = torch.meshgrid(qr, qc, indexing="ij")
                qr_flat = qr_grid.reshape(-1)
                qc_flat = qc_grid.reshape(-1)
                # Filter to circular radius
                dr = (qr_flat - sr).float()
                dc = (qc_flat - sc).float()
                dist = torch.sqrt(dr * dr + dc * dc)
                in_radius = dist <= R
                edge_src_list.append(
                    torch.full((in_radius.sum(),), j, device=device, dtype=torch.long)
                )
                edge_qr_list.append(qr_flat[in_radius])
                edge_qc_list.append(qc_flat[in_radius])

            if not edge_src_list:
                continue

            e_src = torch.cat(edge_src_list)  # (E,)
            e_qr = torch.cat(edge_qr_list)  # (E,)
            e_qc = torch.cat(edge_qc_list)  # (E,)

            # Query features at edge query pixels
            h_query_all = self.query_proj(
                F_grid[b, :, e_qr, e_qc].T  # (E, C_latent) -> (E, H_dim)
            )

            # Geometry features per edge
            dr = e_qr.float() - s_rows[e_src].float()  # pixel diff = km
            dc = e_qc.float() - s_cols[e_src].float()
            dist_km = torch.sqrt(dr * dr + dc * dc).clamp(min=1e-6)
            bearing = torch.atan2(dc, -dr)  # bearing from source to query
            q_elev = raw_elev_patch[b, 0, e_qr, e_qc]
            delta_elev = q_elev - s_elev[e_src]

            geom = torch.stack(
                [
                    dist_km / self.support_radius_px,  # normalised distance
                    torch.sin(bearing),
                    torch.cos(bearing),
                    delta_elev / 1000.0,  # delta elev in km
                ],
                dim=-1,
            )  # (E, 4)

            # Attention scores
            att_input = torch.cat(
                [h_query_all, h_ctx[e_src], geom], dim=-1
            )  # (E, 2*H + 4)
            att_score = self.att_mlp(att_input).squeeze(-1)  # (E,)

            # Softmax per query pixel (destination-normalised)
            q_flat = e_qr * W + e_qc  # unique query pixel index
            alpha = pyg_softmax(att_score, q_flat)  # (E,)

            # Weighted payload messages
            messages = alpha.unsqueeze(-1) * h_pay[e_src]  # (E, H_dim)

            # Scatter-add into da_ctx
            q_flat_exp = q_flat.unsqueeze(-1).expand_as(messages)
            flat_ctx = da_ctx[b].view(self.hidden_dim, H * W)
            flat_ctx.scatter_add_(1, q_flat_exp.T, messages.T)

            # Mark covered pixels in the coverage mask
            unique_q = q_flat.unique()
            coverage_mask[b, 0].view(-1)[unique_q] = 1.0

        return da_ctx, coverage_mask
