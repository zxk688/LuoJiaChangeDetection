import mindspore.numpy as np
import mindspore.nn as nn
import mindspore.ops as F

class conv_block_nested(nn.Cell):
    def __init__(self, channel, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(channel, mid_ch, kernel_size=3, padding=1, has_bias=True, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, has_bias=True, pad_mode='pad')
        self.bn2 = nn.BatchNorm2d(out_ch)

    def construct(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output

class ChannelAttention(nn.Cell):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,has_bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,has_bias=False)
        self.sigmod = nn.Sigmoid()
    def construct(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)
    
class up(nn.Cell):
    def __init__(self, channel, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True,recompute_scale_factor=True)
        else:
            self.up = nn.Conv2dTranspose(channel, channel, 2, stride=2)

    def construct(self, x):

        x = self.up(x)
        return x
    
class SNUNetDecoder(nn.Cell):
    def __init__(self, filters):
        super(SNUNetDecoder, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up1_0 = up(filters[1])
        self.Up2_0 = up(filters[2])
        self.Up3_0 = up(filters[3])
        self.Up4_0 = up(filters[4])

        self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.Up1_1 = up(filters[1])
        self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.Up2_1 = up(filters[2])
        self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3])
        self.Up3_1 = up(filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.Up1_2 = up(filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.Up2_2 = up(filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.Up1_3 = up(filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])

        self.ca = ChannelAttention(filters[0] * 4, ratio=16)
        self.ca1 = ChannelAttention(filters[0], ratio=16 // 4)

    def construct(self, b,d):
        x0_1 = self.conv0_1(F.cat([d[0], self.Up1_0(b[1])], 1))
        x1_1 = self.conv1_1(F.cat([d[1], self.Up2_0(b[2])], 1))
        x0_2 = self.conv0_2(F.cat([d[0], x0_1, self.Up1_1(x1_1)], 1))

        x2_1 = self.conv2_1(F.cat([d[2], self.Up3_0(b[3])], 1))
        x1_2 = self.conv1_2(F.cat([d[1], x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(F.cat([d[0], x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(F.cat([d[3], self.Up4_0(b[4])], 1))
        x2_2 = self.conv2_2(F.cat([d[2], x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(F.cat([d[1], x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(F.cat([d[0], x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))

        out = F.cat([x0_1, x0_2, x0_3, x0_4], 1)
        intra = F.sum(np.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
        
        ca1 = self.ca1(intra)
        ca1_repeated = F.tile(ca1, (1, 4, 1, 1))
        out = self.ca(out) * (out + ca1_repeated)

        return out