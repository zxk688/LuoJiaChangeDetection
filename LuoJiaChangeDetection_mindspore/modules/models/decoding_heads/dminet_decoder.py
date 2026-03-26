import mindspore
import mindspore.nn as nn
import mindspore.ops as F
from mindspore import Tensor

class decode(nn.Cell):
    def __init__(self, in_channel_left, in_channel_down, out_channel,norm_layer=nn.BatchNorm2d):
        super(decode, self).__init__()
        self.conv_d1 = nn.Conv2d(in_channel_down, out_channel, kernel_size=3, stride=1, padding=1,pad_mode='pad')
        self.conv_l = nn.Conv2d(in_channel_left, out_channel, kernel_size=3, stride=1, padding=1,pad_mode='pad')
        self.conv3 = nn.Conv2d(out_channel*2, out_channel, kernel_size=3, stride=1, padding=1,pad_mode='pad')
        self.bn3 = norm_layer(out_channel)

    def construct(self, left, down):
        down_mask = self.conv_d1(down)
        left_mask = self.conv_l(left)
        if down.shape[2:] != left.shape[2:]:
            down_ = F.interpolate(down, size=left.shape[2:], mode='bilinear')
            z1 = F.relu(left_mask * down_)
        else:
            z1 = F.relu(left_mask * down)

        if down_mask.shape[2:] != left.shape[2:]:
            down_mask = F.interpolate(down_mask, size=left.shape[2:], mode='bilinear')

        z2 = F.relu(down_mask * left)

        out = F.cat((z1, z2), axis=1)
        return F.relu(self.bn3(self.conv3(out)))

class BasicConv2d(nn.Cell):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, has_bias=False,pad_mode='pad')
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CrossAtt(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels

        self.query1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key1   = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.value1 = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)

        self.query2 = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key2   = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.value2 = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)

        self.gamma = Tensor(F.zeros(1)) 
        self.softmax = nn.Softmax(axis = -1)

        self.conv_cat = nn.SequentialCell(nn.Conv2d(in_channels*2, out_channels, 3, padding=1, has_bias=False,pad_mode='pad'),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU()) # conv_f

    def construct(self, input1, input2):
        batch_size, channels, height, width = input1.shape
        q1 = self.query1(input1)
        k1 = self.key1(input1).view(batch_size, -1, height * width)
        v1 = self.value1(input1).view(batch_size, -1, height * width)

        q2 = self.query2(input2) 
        k2 = self.key2(input2).view(batch_size, -1, height * width)
        v2 = self.value2(input2).view(batch_size, -1, height * width)

        q = F.cat([q1,q2],1).view(batch_size, -1, height * width).permute(0, 2, 1)
        attn_matrix1 = F.bmm(q, k1)  
        attn_matrix1 = self.softmax(attn_matrix1)
        out1 = F.bmm(v1, attn_matrix1.permute(0, 2, 1)) 
        out1 = out1.view(*input1.shape)
        out1 = self.gamma * out1 + input1

        attn_matrix2 = F.bmm(q, k2) 
        attn_matrix2 = self.softmax(attn_matrix2)
        out2 = F.bmm(v2, attn_matrix2.permute(0, 2, 1))  
        out2 = out2.view(*input2.shape)
        out2 = self.gamma * out2 + input2

        feat_sum = self.conv_cat(F.cat([out1,out2],1))
        return feat_sum, out1, out2

class DMINetDecoder(nn.Cell):
    def __init__(self):
        super(DMINetDecoder, self).__init__()
        self.cross2 = CrossAtt(256, 256) 
        self.cross3 = CrossAtt(128, 128) 
        self.cross4 = CrossAtt(64, 64) 

        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.fam32_1 = decode(128,128,128) # AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(128,64,1)
        self.fam43_1 = decode(64,64,64) # AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(256,128,1)
        self.fam32_2 = decode(128,128,128)
        self.Translayer3_2 = BasicConv2d(128,64,1)
        self.fam43_2 = decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4.0, mode='bilinear',align_corners=True,recompute_scale_factor=True)
        self.upsamplex8 = nn.Upsample(scale_factor=8.0, mode='bilinear',align_corners=True,recompute_scale_factor=True)
    def construct(self, inputs):
        feas1, feas2 = inputs
        cross_result2, cur1_2, cur2_2 = self.cross2(feas1[-1], feas2[-1])
        cross_result3, cur1_3, cur2_3 = self.cross3(feas1[-2], feas2[-2])
        cross_result4, cur1_4, cur2_4 = self.cross4(feas1[-3], feas2[-3]) 

        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        out3_2 = self.fam32_2(F.abs(cur1_3-cur2_3), self.Translayer2_2(F.abs(cur1_2-cur2_2)))
        out4_2 = self.fam43_2(F.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))

        out4_up = self.upsamplex4(out4)
        #return (out4_up,)
        return out4_up