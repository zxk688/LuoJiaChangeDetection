"""
MindSpore implementation of `VGGNet`.
Refer to SqueezeNet: Very Deep Convolutional Networks for Large-Scale Image Recognition.
"""

import math
from typing import Dict, List, Union

import mindspore.common.initializer as init
from mindspore import Tensor, nn
from mindspore import ops as F

from .misc import *

__all__ = [
    "VGG",
    "vgg11",
    "vgg13",
    "vgg16",
    "vgg19",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "features.0",
        "classifier": "classifier.6",
        **kwargs,
    }


default_cfgs = {
    "vgg11": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/vgg/vgg11-ef31d161.ckpt"),
    "vgg13": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/vgg/vgg13-da805e6e.ckpt"),
    "vgg16": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/vgg/vgg16-95697531.ckpt"),
    "vgg19": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/vgg/vgg19-bedee7b6.ckpt"),
}


cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _make_layers(
    cfg: List[Union[str, int]],
    batch_norm: bool = False,
    in_channels: int = 3,
) -> nn.SequentialCell:
    """define the basic block of VGG"""
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, pad_mode="pad", padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v

    return nn.SequentialCell(layers)


class VGG(nn.Cell):
    r"""VGGNet model class, based on
    `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/abs/1409.1556>`_

    Args:
        model_name: name of the architecture. 'vgg11', 'vgg13', 'vgg16' or 'vgg19'.
        batch_norm: use batch normalization or not. Default: False.
        num_classes: number of classification classes. Default: 1000.
        in_channels: number the channels of the input. Default: 3.
        drop_rate: dropout rate of the classifier. Default: 0.5.
    """

    def __init__(
        self,
        model_name: str,
        batch_norm: bool = False,
        num_classes: int = 1000,
        in_channels: int = 3,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        cfg = cfgs[model_name]
        self.features = _make_layers(cfg, batch_norm=batch_norm, in_channels=in_channels)
        self.flatten = nn.Flatten()
        self.classifier = nn.SequentialCell([
            nn.Dense(512 * 7 * 7, 4096),
            nn.ReLU(),
            Dropout(p=drop_rate),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            Dropout(p=drop_rate),
            nn.Dense(4096, num_classes),
        ])
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(init.HeNormal(math.sqrt(5), mode="fan_out", nonlinearity="relu"),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer("zeros", cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.Normal(0.01), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def vgg11(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> VGG:
    """Get 11 layers VGG model.
    Refer to the base class `models.VGG` for more details.
    """
    default_cfg = default_cfgs["vgg11"]
    model = VGG(model_name="vgg11", num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def vgg13(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> VGG:
    """Get 13 layers VGG model.
    Refer to the base class `models.VGG` for more details.
    """
    default_cfg = default_cfgs["vgg13"]
    model = VGG(model_name="vgg13", num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def vgg16(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> VGG:
    """Get 16 layers VGG model.
    Refer to the base class `models.VGG` for more details.
    """
    default_cfg = default_cfgs["vgg16"]
    model = VGG(model_name="vgg16", num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def vgg19(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> VGG:
    """Get 19 layers VGG model.
    Refer to the base class `models.VGG` for more details.
    """
    default_cfg = default_cfgs["vgg19"]
    model = VGG(model_name="vgg19", num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


class vgg16_base(nn.Cell):
    def __init__(self):
        super(vgg16_base, self).__init__()
        features = list(vgg16(pretrained=True).features)[:30]
        self.features = nn.CellList(features)

    def construct(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15, 22, 29}:
                results.append(x)
        return results


class VggEncoder(nn.Cell):
    def __init__(self):
        super().__init__()
        vggnet = vgg16_base()
        self.t1_base = vggnet
        self.t2_base = vggnet
        
    def construct(self,x):
        t1_input = x[:,0]  # 2x3x512x512
        t2_input = x[:,1] 
        t1_list = self.t1_base(t1_input)
        t2_list = self.t2_base(t2_input)
        #t1_list:(4, 64, 256, 256) (4, 128, 128, 128) (4, 256, 64, 64) (4, 512, 32, 32) (4, 512, 16, 16)
        return t1_list, t2_list