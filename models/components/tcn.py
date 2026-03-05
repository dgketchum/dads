import torch.nn as nn
import torch.nn.functional as F


class _CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, bias=True):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        # x: [B, C, T]; causal left padding only
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1, groups=8):
        super().__init__()
        c_in, c_out = channels
        self.conv1 = _CausalConv1d(
            c_in, c_out, kernel_size=kernel_size, dilation=dilation
        )
        self.norm1 = nn.GroupNorm(num_groups=min(groups, c_out), num_channels=c_out)
        self.conv2 = _CausalConv1d(
            c_out, c_out, kernel_size=kernel_size, dilation=dilation
        )
        self.norm2 = nn.GroupNorm(num_groups=min(groups, c_out), num_channels=c_out)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.downsample = None
        if c_in != c_out:
            self.downsample = nn.Conv1d(c_in, c_out, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
        return out + residual


class TemporalConvEncoder(nn.Module):
    """Compact causal TCN for 12-day windows.

    Inputs
    - x: [B, C_in, T]
    Outputs
    - [B, out_dim]
    """

    def __init__(
        self,
        in_channels: int,
        channels: int = 128,
        kernel_size: int = 3,
        dilations=(1, 2, 4, 8),
        dropout: float = 0.1,
        out_dim: int = 256,
    ):
        super().__init__()
        layers = []
        c_prev = in_channels
        for d in dilations:
            layers.append(
                TemporalBlock(
                    (c_prev, channels),
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )
            c_prev = channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels, out_dim)

    def forward(self, x):
        # x: [B, C, T]
        h = self.tcn(x)
        # last timestep readout
        h_last = h[:, :, -1]
        out = self.head(h_last)
        return out
