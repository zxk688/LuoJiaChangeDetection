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
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad_mode='pad',padding=1, dilation=1, group=group, has_bias=bias)
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
                          _out_channels: ((i + dilation_set // 2) % dilation_set + j * group + 1) * _out_channels,
                i * _in_channels: (i + 1) * _in_channels, :, :] = 1
        self.conv.weight.data[self.mask] = 0
        # self.conv.weight.register_backward_hook(backward_hook)
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
            nn.Conv2d(in_ch, native_ch, kernel_size, stride, padding=padding, dilation=1, has_bias=False),
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
    
class First_DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(First_DoubleConv, self).__init__()
        self.conv = nn.SequentialCell(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1,pad_mode='pad'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1,pad_mode='pad'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, input):
        return self.conv(input)


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
    
class CNNAttentionEncoder(nn.Module):
    def __init__(self,in_ch, mlp, ratio):
        super(CNNAttentionEncoder, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_1 = First_DoubleConv(in_ch, int(mlp[0] * ratio))
        self.Conv1_2 = First_DoubleConv(in_ch, int(mlp[0] * ratio))
        self.Conv2_1 = DoubleConv(int(mlp[0] * ratio), int(mlp[1] * ratio))
        self.Conv2_2 = DoubleConv(int(mlp[0] * ratio), int(mlp[1] * ratio))
        self.Conv3_1 = DoubleConv(int(mlp[1] * ratio), int(mlp[2] * ratio))
        self.Conv3_2 = DoubleConv(int(mlp[1] * ratio), int(mlp[2] * ratio))
        self.Conv4_1 = DoubleConv(int(mlp[2] * ratio), int(mlp[3] * ratio))
        self.Conv4_2 = DoubleConv(int(mlp[2] * ratio), int(mlp[3] * ratio))
        self.Conv5_1 = DoubleConv(int(mlp[3] * ratio), int(mlp[4] * ratio))
        self.Conv5_2 = DoubleConv(int(mlp[3] * ratio), int(mlp[4] * ratio))

    def forward(self, x):
        x1 = x[:,0]  # 10x3x256x256
        x2 = x[:,1] 
        c1_1 = self.Conv1_1(x1)
        c1_2 = self.Conv1_2(x2)

        c2_1 = self.Maxpool(c1_1)
        c2_1 = self.Conv2_1(c2_1)
        c2_2 = self.Maxpool(c1_2)
        c2_2 = self.Conv2_2(c2_2)

        c3_1 = self.Maxpool(c2_1)
        c3_1 = self.Conv3_1(c3_1)
        c3_2 = self.Maxpool(c2_2)
        c3_2 = self.Conv3_2(c3_2)

        c4_1 = self.Maxpool(c3_1)
        c4_1 = self.Conv4_1(c4_1)
        c4_2 = self.Maxpool(c3_2)
        c4_2 = self.Conv4_2(c4_2)

        c5_1 = self.Maxpool(c4_1)
        c5_1 = self.Conv5_1(c5_1)
        c5_2 = self.Maxpool(c4_2)
        c5_2 = self.Conv5_2(c5_2)
        return (c1_1,c2_1,c3_1,c4_1,c5_1,), (c1_2,c2_2,c3_2,c4_2,c5_2,)
    