import mindspore
import mindspore.nn as nn
import mindspore.ops as F
from mindspore import Tensor

from .resnet import resnet50

class _PSPModule(nn.Cell):
    def __init__(self, in_channels, bin_sizes):
        super(_PSPModule, self).__init__()

        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.CellList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])
        self.bottleneck = nn.SequentialCell(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), out_channels, 
                                    kernel_size=3, padding=1, has_bias=False,pad_mode='pad'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU()
        return nn.SequentialCell(prior, conv, bn, relu)
    
    def construct(self, features):
        h, w = features.shape[2], features.shape[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=False) for stage in self.stages])
        output = self.bottleneck(F.cat(pyramids, axis=1))
        return output


class SemiEncoder(nn.Cell):
    def __init__(self):
        super(SemiEncoder, self).__init__()
        model = resnet50(pretrained=False,replace_stride_with_dilation=[False,True,True])
        self.base = nn.SequentialCell(
            nn.SequentialCell(model.conv1, model.maxpool),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
        self.psp = _PSPModule(2048, bin_sizes=[1, 2, 3, 6])

    def construct(self, AB):
        split_tensors = F.Split(1,2)(AB)
        A = split_tensors[0]  # 10x3x256x256
        B = split_tensors[1] 
        a = self.base(A)
        b = self.base(B)
        diff = F.abs(a-b)
        x = self.psp(diff)
        return x
