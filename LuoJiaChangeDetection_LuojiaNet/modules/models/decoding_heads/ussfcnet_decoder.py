import numpy as np
import math
import luojianet
import luojianet.nn as nn
import luojianet.ops as F


class SSFC(nn.Module):
    def __init__(self, in_ch):
        super(SSFC, self).__init__()

        # self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)  # generate k by conv

    def forward(self, x):
        _, _, h, w = x.shape

        q = F.mean(x, axis=[2, 3], keep_dims=True)
        # k = self.proj(x)
        k = x
        square = F.pow(k - q, 2)
        sigma = F.sum(square, dim=[2, 3], keepdim=True) / (h * w)
        att_score = square / (2 * sigma + F.scalar_to_array(np.finfo(np.float32).eps)) + 0.5
        att_weight = nn.Sigmoid()(att_score)
        # print(sigma)
        return x * att_weight

class CMConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=3, group=1, dilation_set=4,
                 bias=False):
        super(CMConv, self).__init__()
        self.prim = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=dilation, dilation=dilation,
                              group=group * dilation_set, has_bias=bias,pad_mode='pad')
        self.prim_shift = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=2 * dilation, dilation=2 * dilation,
                                    group=group * dilation_set, has_bias=bias,pad_mode='pad')
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad_mode='pad', padding=1, group=group, has_bias=bias)

        def backward_hook(grad):
            out = grad.clone()
            out[self.mask] = 0
            return out

        self.mask = F.zeros(self.conv.weight.shape,dtype=luojianet.int64)
        _in_channels = in_ch // (group * dilation_set)
        _out_channels = out_ch // (group * dilation_set)
        for i in range(dilation_set):
            for j in range(group):
                self.mask[(i + j * group) * _out_channels: (i + j * group + 1) * _out_channels,
                i * _in_channels: (i + 1) * _in_channels, :, :] = 1
                self.mask[((i + dilation_set // 2) % dilation_set + j * group) *
                          _out_channels: ((i + dilation_set // 2) % dilation_set + j *  + 1) * _out_channels,
                i * _in_channels: (i + 1) * _in_channels, :, :] = 1
        self.conv.weight.data[self.mask] = 0
        # self.conv.weight.register_hook(backward_hook)
        self.group = group

    def forward(self, x):
        x_split = (z.chunk(2, axis=1) for z in x.chunk(self.group, axis=1))
        x_merge = F.cat(tuple(F.cat((x2, x1), axis=1) for (x1, x2) in x_split), axis=1)
        x_shift = self.prim_shift(x_merge)
        return self.prim(x) + self.conv(x) + x_shift
    
class MSDConv_SSFC(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, ratio=2, aux_k=3, dilation=3):
        super(MSDConv_SSFC, self).__init__()
        self.out_ch = out_ch
        native_ch = math.ceil(out_ch / ratio)
        aux_ch = native_ch * (ratio - 1)

        # native feature maps
        self.native = nn.SequentialCell(
            nn.Conv2d(in_ch, native_ch, kernel_size, stride, padding=padding, dilation=1, has_bias=False,pad_mode='pad'),
            nn.BatchNorm2d(native_ch),
            nn.ReLU(),
        )

        # auxiliary feature maps
        self.aux = nn.SequentialCell(
            CMConv(native_ch, aux_ch, aux_k, 1, padding=1, group=int(native_ch / 4), dilation=dilation,
                   bias=False),
            nn.BatchNorm2d(aux_ch),
            nn.ReLU(),
        )

        self.att = SSFC(aux_ch)

    def forward(self, x):
        x1 = self.native(x)
        x2 = self.att(self.aux(x1))
        out = F.cat([x1, x2], axis=1)
        return out[:, :self.out_ch, :, :]
    
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.Conv = nn.SequentialCell(
            MSDConv_SSFC(in_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            MSDConv_SSFC(out_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, input):
        return self.Conv(input)

class CNNAttentionDecoder(nn.Module):
    def __init__(self, out_ch,mlp,ratio):
        super(CNNAttentionDecoder, self).__init__()
        self.Up5 = nn.Conv2dTranspose(int(mlp[4] * ratio), int(mlp[3] * ratio), 2, stride=2)
        self.Up_conv5 = DoubleConv(int(mlp[4] * ratio), int(mlp[3] * ratio))

        self.Up4 = nn.Conv2dTranspose(int(mlp[3] * ratio), int(mlp[2] * ratio), 2, stride=2)
        self.Up_conv4 = DoubleConv(int(mlp[3] * ratio), int(mlp[2] * ratio))

        self.Up3 = nn.Conv2dTranspose(int(mlp[2] * ratio), int(mlp[1] * ratio), 2, stride=2)
        self.Up_conv3 = DoubleConv(int(mlp[2] * ratio), int(mlp[1] * ratio))

        self.Up2 = nn.Conv2dTranspose(int(mlp[1] * ratio), int(mlp[0] * ratio), 2, stride=2)
        self.Up_conv2 = DoubleConv(int(mlp[1] * ratio), int(mlp[0] * ratio))

        self.Conv_1x1 = nn.Conv2d(int(mlp[0] * ratio), out_ch, kernel_size=1, stride=1, padding=0)
        
    def forward(self, feaslist): 
        # decoding
        d5 = self.Up5(feaslist[4])
        d5 = F.cat((feaslist[3], d5), axis=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = F.cat((feaslist[2], d4), axis=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = F.cat((feaslist[1], d3), axis=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = F.cat((feaslist[0], d2), axis=1)
        d2 = self.Up_conv2(d2)
        return d2