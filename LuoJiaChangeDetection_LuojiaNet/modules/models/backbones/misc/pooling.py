""" GlobalAvgPooling Module"""
from luojianet import nn, ops


class GlobalAvgPooling(nn.Module):
    """
    GlobalAvgPooling, same as torch.nn.AdaptiveAvgPool2d when output shape is 1
    """

    def __init__(self, keep_dims: bool = False) -> None:
        super().__init__()
        self.keep_dims = keep_dims

    def forward(self, x):
        x = ops.mean(x, axis=(2, 3), keep_dims=self.keep_dims)
        return x
