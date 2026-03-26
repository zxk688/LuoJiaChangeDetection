"""Identity Module"""
from luojianet import nn


class Identity(nn.Module):
    """Identity"""

    def forward(self, x):
        return x
