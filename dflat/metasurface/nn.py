import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SirenNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, channels=(256, 256), w0=1.0, w0_initial=30
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        chi = in_channels
        for i, ch in enumerate(channels):
            layer = Siren(
                chi,
                ch,
                w0=w0_initial if i == 0 else w0,
                is_first=True if i == 0 else False,
            )
            self.blocks.append(layer)
            chi = ch

        self.blocks.append(Siren(chi, out_channels, w0))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class VanillaMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        chi = in_channels
        for ch in channels:
            layer = []
            layer.append(nn.Linear(chi, ch))
            layer.append(nn.LeakyReLU(0.01))
            self.blocks.append(nn.Sequential(*layer))
            chi = ch

        self.blocks.append(nn.Linear(chi, out_channels))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=1.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
        dropout=0.0,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        return out
