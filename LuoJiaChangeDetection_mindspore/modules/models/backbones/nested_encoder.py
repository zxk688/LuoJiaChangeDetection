import mindspore
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

class NestedCNNEncoder(nn.Cell):
    def __init__(self,in_channels,filters):
        super(NestedCNNEncoder, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv0_0 = conv_block_nested(in_channels, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

    def construct(self,x):
        xA = x[:,0]  # 10x3x256x256
        xB = x[:,1] 
        '''xA'''
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        x4_0A = self.conv4_0(self.pool(x3_0A))
        '''xB'''
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        return (x0_0A,x1_0A,x2_0A,x3_0A,x4_0A,), (x0_0B,x1_0B,x2_0B,x3_0B,x4_0B,)