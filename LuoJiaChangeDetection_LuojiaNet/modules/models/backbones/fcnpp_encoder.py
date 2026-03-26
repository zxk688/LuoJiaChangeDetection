import luojianet.nn as nn
import luojianet.ops as F
from collections import OrderedDict

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True,
                 bias=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        depthwise = nn.Conv2d(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, has_bias=bias, pad_mode='pad')
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, has_bias=bias, pad_mode='pad')
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
            #nn.ReLU(inplace=True),
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


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.SequentialCell(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class _ASPP(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256):
        super().__init__()

        dilations = [6, 12, 18]

        self.aspp0 = nn.SequentialCell(OrderedDict([('conv', nn.Conv2d(in_channels, out_channels, 1, has_bias=False)),
                                                ('bn', nn.BatchNorm2d(out_channels)),
                                                ('relu', nn.ReLU())]))
        self.aspp1 = SeparableConv2d(in_channels, out_channels, dilation=dilations[0], relu_first=False)
        self.aspp2 = SeparableConv2d(in_channels, out_channels, dilation=dilations[1], relu_first=False)
        self.aspp3 = SeparableConv2d(in_channels, out_channels, dilation=dilations[2], relu_first=False)

        self.image_pooling = nn.SequentialCell(OrderedDict([('gap', nn.AdaptiveAvgPool2d((1, 1))),
                                                        ('conv', nn.Conv2d(in_channels, out_channels, 1, has_bias=False)),
                                                        ('bn', nn.BatchNorm2d(out_channels)),
                                                        ('relu', nn.ReLU())]))

        self.conv = nn.Conv2d(out_channels * 5, out_channels, 1, has_bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        pool = self.image_pooling(x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=True)

        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x = F.cat((pool, x0, x1, x2, x3), axis=1)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x
    
class FCNPPEncoder(nn.Module):
    def __init__(self,in_channels,DC_out, mlp):
        super(FCNPPEncoder, self).__init__()
        self.inc = DoubleConv(in_channels, DC_out)
        self.down1 = Down(mlp[0][0], mlp[0][1])
        self.down2 = Down(mlp[1][0], mlp[1][1])
        self.down3 = Down(mlp[2][0], mlp[2][1])
        self.aspp = _ASPP(mlp[3][0], mlp[3][1])
        
    def forward(self, inputs):
        x1 = inputs[:,0]# 10x3x256x256
        x2 = inputs[:,1]
        x1 = F.abs(x1-x2)
        x1 = self.inc(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.aspp(x4)

        return (x5,x4,x3,x2,x1,)
