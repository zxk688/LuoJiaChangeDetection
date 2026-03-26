import luojianet
import luojianet.nn as nn
import luojianet.ops as F

from collections import OrderedDict


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True,
                 bias=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        depthwise = nn.Conv2d(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, group=inplanes,has_bias=bias,pad_mode='pad')
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, has_bias=bias)
        bn_point = norm_layer(planes)

        if relu_first:
            self.block = nn.SequentialCell(OrderedDict([('relu', nn.ReLU()),
                                                    ('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point)
                                                    ]))
        else:
            self.block = nn.SequentialCell(OrderedDict([('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('relu1', nn.ReLU()),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point),
                                                    ('relu2', nn.ReLU())
                                                    ]))

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1,pad_mode='pad'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2,pad_mode='pad'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3,pad_mode='pad'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True,recompute_scale_factor=True)
        else:
            self.up = nn.Conv2dTranspose(in_channels // 2, in_channels // 2, kernel_size=2, stride=2, pad_mode='pad')

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # diffY = luojianet.tensor([x2.size()[2] - x1.size()[2]])
        # diffX = luojianet.tensor([x2.size()[3] - x1.size()[3]])
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = F.cat([x2, x1], axis=1)
        return self.conv(x)
    
class FCNPPDecoder(nn.Module):
    def __init__(self,bilinear,up):
        super(FCNPPDecoder, self).__init__()
        self.up1 = Up(up[4], up[2], bilinear)
        self.up2 = Up(up[3], up[1], bilinear)
        self.up3 = Up(up[2], up[0], bilinear)
        self.up4 = Up(up[1], up[0], bilinear)
        
    def forward(self,b):

        x = self.up1(b[0], b[1])
        x = self.up2(x, b[2])
        x = self.up3(x, b[3])
        x = self.up4(x, b[4])

        return x



