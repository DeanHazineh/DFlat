import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        self.blocks.append(nn.Linear(in_channels, channels[0]))
        chi = channels[0]

        for ch in channels:
            layer = []
            layer.append(nn.Linear(chi, ch))
            layer.append(nn.GELU())
            self.blocks.append(nn.Sequential(*layer))
            chi = ch

        self.blocks.append(nn.Linear(chi, out_channels))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
