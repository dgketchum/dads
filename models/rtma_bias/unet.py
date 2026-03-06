from __future__ import annotations

import torch
from torch import nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSmall(nn.Module):
    """
    Minimal U-Net for patch-based correction fields.

    This is intentionally small for MVP iteration speed.

    When *n_heads* > 1, the output layer is replaced by a ModuleList of
    independent 1×1 conv heads.  ``forward()`` returns a list of tensors
    ``[(B, 1, H, W), ...]`` in that case, or a single ``(B, out_channels, H, W)``
    tensor when ``n_heads == 1`` (backward compatible).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base: int = 32,
        n_heads: int = 1,
    ):
        super().__init__()
        b = int(base)
        self.n_heads = int(n_heads)
        self.down1 = _ConvBlock(in_channels, b)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = _ConvBlock(b, 2 * b)
        self.pool2 = nn.MaxPool2d(2)

        self.mid = _ConvBlock(2 * b, 4 * b)

        self.up2 = nn.ConvTranspose2d(4 * b, 2 * b, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(4 * b, 2 * b)
        self.up1 = nn.ConvTranspose2d(2 * b, b, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(2 * b, b)

        if self.n_heads > 1:
            self.heads = nn.ModuleList(
                [nn.Conv2d(b, 1, kernel_size=1) for _ in range(self.n_heads)]
            )
        else:
            self.out = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        xm = self.mid(self.pool2(x2))

        x = self.up2(xm)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        if self.n_heads > 1:
            return [head(x) for head in self.heads]
        return self.out(x)
