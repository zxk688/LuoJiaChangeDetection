"""
MindSpore implementation of `ResNet`.
Refer to Deep Residual Learning for Image Recognition.
"""

from typing import List, Optional, Type, Union

import luojianet.common.initializer as init
from luojianet import Tensor, nn
from luojianet import ops as F

# from ..misc.helpers import build_model_with_cfg
# from ...misc.pooling import GlobalAvgPooling
# from .misc.registry import register_model
from ...models.backbones.misc.helpers import build_model_with_cfg
from ...models.backbones.misc.pooling import GlobalAvgPooling
from ...models.backbones.misc.registry import register_model
__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x4d",
    "resnext101_64x4d",
    "resnext152_64x4d",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "conv1",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "resnet18": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet18-1e65cd21.ckpt"),
    "resnet34": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet34-f297d27e.ckpt"),
    "resnet50": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet50-e0733ab8.ckpt"),
    "resnet101": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet101-689c5e77.ckpt"),
    "resnet152": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet152-beb689d8.ckpt"),
    "resnext50_32x4d": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnext/resnext50_32x4d-af8aba16.ckpt"),
    "resnext101_32x4d": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/resnext/resnext101_32x4d-3c1e9c51.ckpt"
    ),
    "resnext101_64x4d": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/resnext/resnext101_64x4d-8929255b.ckpt"
    ),
    "resnext152_64x4d": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/resnext/resnext152_64x4d-3aba275c.ckpt"
    ),
}

class BasicBlock(nn.Module):
    """define the basic block of resnet"""
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Module] = None,
        down_sample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d
        assert groups == 1, "BasicBlock only supports groups=1"
        assert base_width == 64, "BasicBlock only supports base_width=64"

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3,
                               stride=stride, padding=1, pad_mode="pad")
        self.bn1 = norm(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=1, padding=1, pad_mode="pad")
        self.bn2 = norm(channels)
        self.down_sample = down_sample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck here places the stride for downsampling at 3x3 convolution(self.conv2) as torchvision does,
    while original implementation places the stride at the first 1x1 convolution(self.conv1)
    """
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Module] = None,
        down_sample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d

        width = int(channels * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1)
        self.bn1 = norm(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, pad_mode="pad", group=groups)
        self.bn2 = norm(width)
        self.conv3 = nn.Conv2d(width, channels * self.expansion,
                               kernel_size=1, stride=1)
        self.bn3 = norm(channels * self.expansion)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    r"""ResNet model class, based on
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_

    Args:
        block: block of resnet.
        layers: number of layers of each stage.
        num_classes: number of classification classes. Default: 1000.
        in_channels: number the channels of the input. Default: 3.
        groups: number of groups for group conv in blocks. Default: 1.
        base_width: base width of pre group hidden channel in blocks. Default: 64.
        norm: normalization layer in blocks. Default: None.
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        in_channels: int = 3,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d

        self.norm: nn.Module = norm  # add type hints to make pylint happy
        self.input_channels = 64
        self.groups = groups
        self.base_with = base_width

        self.conv1 = nn.Conv2d(in_channels, self.input_channels, kernel_size=7,
                               stride=2, pad_mode="pad", padding=3)
        self.bn1 = norm(self.input_channels)
        self.relu = nn.ReLU()
        self.feature_info = [dict(chs=self.input_channels, reduction=2, name="relu")]

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,pad_mode="pad")
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.feature_info.append(dict(chs=block.expansion * 64, reduction=4, name="layer1"))

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.feature_info.append(dict(chs=block.expansion * 128, reduction=8, name="layer2"))

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.feature_info.append(dict(chs=block.expansion * 256, reduction=16, name="layer3"))

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.feature_info.append(dict(chs=block.expansion * 512, reduction=32, name="layer4"))

        self.pool = GlobalAvgPooling()
        self.num_features = 512 * block.expansion
        self.classifier = nn.Dense(self.num_features, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(init.HeNormal(mode='fan_out', nonlinearity='relu'),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer('zeros', cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(init.initializer('ones', cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(init.initializer('zeros', cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.HeUniform(mode='fan_in', nonlinearity='sigmoid'),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer('zeros', cell.bias.shape, cell.bias.dtype))

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        channels: int,
        block_nums: int,
        stride: int = 1,
    ) -> nn.SequentialCell:
        """build model depending on cfgs"""
        down_sample = None

        if stride != 1 or self.input_channels != channels * block.expansion:
            down_sample = nn.SequentialCell([
                nn.Conv2d(self.input_channels, channels * block.expansion, kernel_size=1, stride=stride),
                self.norm(channels * block.expansion)
            ])

        layers = []
        layers.append(
            block(
                self.input_channels,
                channels,
                stride=stride,
                down_sample=down_sample,
                groups=self.groups,
                base_width=self.base_with,
                norm=self.norm,
            )
        )
        self.input_channels = channels * block.expansion

        for _ in range(1, block_nums):
            layers.append(
                block(
                    self.input_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_with,
                    norm=self.norm
                )
            )

        return nn.SequentialCell(layers)

    def forward_features(self, x: Tensor) -> Tensor:
        """Network forward feature extraction."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_resnet(pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet, pretrained, **kwargs)


@register_model
def resnet18(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 18 layers ResNet model.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["resnet18"]
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels,
                      **kwargs)
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))

@register_model
def resnet34(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 34 layers ResNet model.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["resnet34"]
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels,
                      **kwargs)
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnet50(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 50 layers ResNet model.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["resnet50"]
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels,
                      **kwargs)
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnet101(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 101 layers ResNet model.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["resnet101"]
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], num_classes=num_classes, in_channels=in_channels,
                      **kwargs)
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnet152(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 152 layers ResNet model.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["resnet152"]
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], num_classes=num_classes, in_channels=in_channels,
                      **kwargs)
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnext50_32x4d(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 50 layers ResNeXt model with 32 groups of GPConv.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["resnext50_32x4d"]
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], groups=32, base_width=4, num_classes=num_classes,
                      in_channels=in_channels, **kwargs)
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnext101_32x4d(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 101 layers ResNeXt model with 32 groups of GPConv.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["resnext101_32x4d"]
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], groups=32, base_width=4, num_classes=num_classes,
                      in_channels=in_channels, **kwargs)
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnext101_64x4d(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 101 layers ResNeXt model with 64 groups of GPConv.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["resnext101_64x4d"]
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], groups=64, base_width=4, num_classes=num_classes,
                      in_channels=in_channels, **kwargs)
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnext152_64x4d(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["resnext152_64x4d"]
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], groups=64, base_width=4, num_classes=num_classes,
                      in_channels=in_channels, **kwargs)
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))


class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        self.resnet = resnet18(pretrained=False)
        expand = 1
        self.upsamplex2 = nn.Upsample(scale_factor=2.0,mode='bilinear',recompute_scale_factor=True)
        self.upsamplex4 = nn.Upsample(scale_factor=4.0,mode='bilinear',recompute_scale_factor=True)
        self.upsamplex8 = nn.Upsample(scale_factor=8.0,mode='bilinear',recompute_scale_factor=True)
        self.resnet_stages_num = 5

        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1,pad_mode='pad')

    def forward(self,x):
        split_tensors = F.Split(1,2)(x)
        img1 = split_tensors[0]  # 10x3x256x256
        img2 = split_tensors[1]
        x1 = self.forward_single(img1)
        x2 = self.forward_single(img2)
        return x1, x2
    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128
        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256
        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError
        x = self.upsamplex8(x_8)
        x = self.conv_pred(x)# output layers([2, 32, 256, 256]) x_1
        return x

class ResnetMultilevelEncoder(nn.Module):
    def __init__(self):
        super(ResnetMultilevelEncoder, self).__init__()
        self.resnet = resnet18(pretrained=False)
    def forward(self, x):
        split_tensors = F.Split(1,2)(x)
        imgs1 = split_tensors[0]  # 10x3x256x256
        imgs2 = split_tensors[1] 
        
        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)
        
        return (c0, c1, c2, c3,), (c0_img2, c1_img2, c2_img2, c3_img2,)

    

class AERNetEncoder(nn.Module):
    def __init__(self):
        super(AERNetEncoder, self).__init__()
        self.resnet = resnet34(pretrained=True)
    def forward(self,x):
        split_tensors = F.Split(1,2)(x)
        x1 = split_tensors[0]  # 10x3x256x256
        x2 = split_tensors[1] 
        
        x1= self.resnet.conv1(x1)
        x1= self.resnet.bn1(x1)
        x1_0 = self.resnet.relu(x1)
        x1= self.resnet.maxpool(x1_0)
        x1_1 = self.resnet.layer1(x1)
        x1_2 = self.resnet.layer2(x1_1)
        x1_3 = self.resnet.layer3(x1_2)
        x1_4 = self.resnet.layer4(x1_3)

        x2= self.resnet.conv1(x2)
        x2= self.resnet.bn1(x2)
        x2_0 = self.resnet.relu(x2)
        x2= self.resnet.maxpool(x2_0)
        x2_1 = self.resnet.layer1(x2)
        x2_2 = self.resnet.layer2(x2_1)
        x2_3 = self.resnet.layer3(x2_2)
        x2_4 = self.resnet.layer4(x2_3)
        
        return (x1_0,x1_1,x1_2,x1_3,x1_4,),(x2_0,x2_1,x2_2,x2_3,x2_4,)