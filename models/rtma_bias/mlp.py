from __future__ import annotations

import torch
from torch import nn


class PointMLP(nn.Module):
    """Center-pixel MLP baseline for bias correction.

    Takes the same (B, C, H, W) input as UNetSmall but only uses the center
    pixel.  Outputs match UNetSmall's interface: a single (B, 1, 1, 1) tensor
    or a list of (B, 1, 1, 1) tensors when n_heads > 1.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        n_heads: int = 1,
    ):
        super().__init__()
        self.n_heads = int(n_heads)

        layers: list[nn.Module] = []
        prev = in_channels
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)

        if self.n_heads > 1:
            self.heads = nn.ModuleList(
                [nn.Linear(prev, 1) for _ in range(self.n_heads)]
            )
        else:
            self.out = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        # Extract center pixel: (B, C, H, W) -> (B, C)
        cy = x.shape[-2] // 2
        cx = x.shape[-1] // 2
        xc = x[:, :, cy, cx]

        h = self.backbone(xc)  # (B, hidden[-1])

        if self.n_heads > 1:
            return [head(h).unsqueeze(-1).unsqueeze(-1) for head in self.heads]
        return self.out(h).unsqueeze(-1).unsqueeze(-1)
