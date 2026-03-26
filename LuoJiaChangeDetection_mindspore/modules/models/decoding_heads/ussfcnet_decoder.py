import numpy as np
import math
import mindspore
import mindspore.nn as nn
from mindspore import ops as F 
from mindspore.common.initializer import initializer, HeNormal
from mindspore.common.tensor import Tensor 

class SSFC(nn.Cell):
    """
    自相关特征通道注意力模块
    """
    def __init__(self, in_ch):
        super(SSFC, self).__init__()
        self.eps = Tensor(float(np.finfo(np.float32).eps), mindspore.float32)

    def construct(self, x):
        _, _, h, w = x.shape

        q = F.mean(x, (2, 3), True) 
        k = x
        
        square = F.pow(k - q, 2)      
        sigma = F.sum(square, (2, 3), True) / (h * w)
        
        att_score = square / (2 * sigma + self.eps) + 0.5
        att_weight = nn.Sigmoid()(att_score)
        
        return x * att_weight

class CMConv(nn.Cell):
    """
    上下文掩码卷积 (Context-Masked Convolution)
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=3, group=1, dilation_set=4,
                 bias=False):
        super(CMConv, self).__init__()
        
        self.prim = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=dilation, dilation=dilation,
                              group=group * dilation_set, has_bias=bias, pad_mode='pad')
        self.prim_shift = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=2 * dilation, dilation=2 * dilation,
                                    group=group * dilation_set, has_bias=bias, pad_mode='pad')
        
        weight_shape = (out_ch, in_ch // group, kernel_size, kernel_size)
        initial_weight = initializer(HeNormal(mode='fan_in'), shape=weight_shape, dtype=mindspore.float32)
        mask = F.zeros(weight_shape, mindspore.int64)
        _in_channels = in_ch // (group * dilation_set)
        _out_channels = out_ch // (group * dilation_set)
        
        for i in range(dilation_set):
            for j in range(group):
                mask[(i + j * group) * _out_channels: (i + j * group + 1) * _out_channels,
                     i * _in_channels: (i + 1) * _in_channels, :, :] = 1
                mask[((i + dilation_set // 2) % dilation_set + j * group) * _out_channels: 
                     ((i + dilation_set // 2) % dilation_set + j * group + 1) * _out_channels,
                     i * _in_channels: (i + 1) * _in_channels, :, :] = 1

        final_weight_data = F.masked_fill(initial_weight, mask.astype(mindspore.bool_), 0.0)
        
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad_mode='pad', padding=1, 
                              group=group, has_bias=bias, weight_init=final_weight_data)
        self.group = group

    def construct(self, x):
        x_chunks_group = F.chunk(x, self.group, axis=1)
        
        x_split = [F.chunk(z, 2, axis=1) for z in x_chunks_group]
        
        inner_cats = tuple(F.cat((x2, x1), axis=1) for (x1, x2) in x_split)
        x_merge = F.cat(inner_cats, axis=1)
        
        x_shift = self.prim_shift(x_merge)
        return self.prim(x) + self.conv(x) + x_shift
    
class MSDConv_SSFC(nn.Cell):
    """
    多尺度空洞卷积 + SSFC 注意力
    """
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, ratio=2, aux_k=3, dilation=3):
        super(MSDConv_SSFC, self).__init__()
        self.out_ch = out_ch

        _native_ch_base = math.ceil(out_ch / ratio)
        self.native_ch = (int(_native_ch_base) // 4 + 1) * 4 if _native_ch_base % 4 != 0 else int(_native_ch_base)

        if self.native_ch == 0:
             raise ValueError(f"Calculated native_ch is 0 for out_ch={out_ch} and ratio={ratio}. This is not allowed.")

        self.aux_ch = self.native_ch * (ratio - 1)

        self.native = nn.SequentialCell(
            nn.Conv2d(in_ch, self.native_ch, kernel_size, stride, padding=padding, dilation=1, has_bias=False, pad_mode='pad'),
            nn.BatchNorm2d(self.native_ch),
            nn.ReLU(),
        )

        # 仅当需要辅助分支时才创建它
        if self.aux_ch > 0:
            self.aux = nn.SequentialCell(
                CMConv(self.native_ch, self.aux_ch, aux_k, 1, padding=1, group=self.native_ch // 4, dilation=dilation,
                       bias=False),
                nn.BatchNorm2d(self.aux_ch),
                nn.ReLU(),
            )
            self.att = SSFC(self.aux_ch)
        else:
            self.aux = None

    def construct(self, x):
        x1 = self.native(x)
        # 如果没有辅助分支，直接返回 native 特征
        if self.aux is None:
            return x1[:, :self.out_ch, :, :]

        x2 = self.att(self.aux(x1))
        out = F.cat((x1, x2), axis=1)
        return out[:, :self.out_ch, :, :]
    
class DoubleConv(nn.Cell):
    """
    双重卷积块
    """
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

    def construct(self, x):
        return self.Conv(x)

class CNNAttentionDecoder(nn.Cell):
    """
    CNN 注意力解码器
    """
    def __init__(self, out_ch, mlp, ratio):
        super(CNNAttentionDecoder, self).__init__()
        mlp_r = [int(ch * ratio) for ch in mlp]

        self.Up5 = nn.Conv2dTranspose(mlp_r[4], mlp_r[3], 2, stride=2)
        self.Up_conv5 = DoubleConv(mlp_r[4], mlp_r[3]) # PyTorch cat([N,C1,H,W], [N,C2,H,W]) -> [N, C1+C2, H, W], so in_ch is mlp_r[3]+mlp_r[3]
        # MindSpore is the same. 
        # Original: d5 = cat((feaslist[3], d5)) -> feaslist[3] is mlp_r[3], d5 is mlp_r[3]. so in_ch for Up_conv5 should be 2 * mlp_r[3].
        self.Up_conv5 = DoubleConv(mlp_r[3] + mlp_r[3], mlp_r[3])

        self.Up4 = nn.Conv2dTranspose(mlp_r[3], mlp_r[2], 2, stride=2)
        self.Up_conv4 = DoubleConv(mlp_r[2] + mlp_r[2], mlp_r[2])

        self.Up3 = nn.Conv2dTranspose(mlp_r[2], mlp_r[1], 2, stride=2)
        self.Up_conv3 = DoubleConv(mlp_r[1] + mlp_r[1], mlp_r[1])

        self.Up2 = nn.Conv2dTranspose(mlp_r[1], mlp_r[0], 2, stride=2)
        self.Up_conv2 = DoubleConv(mlp_r[0] + mlp_r[0], mlp_r[0])

        self.Conv_1x1 = nn.Conv2d(mlp_r[0], out_ch, kernel_size=1, stride=1, padding=0, pad_mode='valid')
        
    def construct(self, feaslist): 
        d5_up = self.Up5(feaslist[4])
        d5 = F.cat((feaslist[3], d5_up), axis=1)
        d5 = self.Up_conv5(d5)

        d4_up = self.Up4(d5)
        d4 = F.cat((feaslist[2], d4_up), axis=1)
        d4 = self.Up_conv4(d4)

        d3_up = self.Up3(d4)
        d3 = F.cat((feaslist[1], d3_up), axis=1)
        d3 = self.Up_conv3(d3)

        d2_up = self.Up2(d3)
        d2 = F.cat((feaslist[0], d2_up), axis=1)
        d2 = self.Up_conv2(d2)
        
        output = self.Conv_1x1(d2)
        return output