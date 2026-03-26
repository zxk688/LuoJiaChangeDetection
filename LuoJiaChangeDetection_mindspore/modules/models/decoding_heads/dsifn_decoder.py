import mindspore
import mindspore.nn as nn
import mindspore.ops as F


class ChannelAttention(nn.Cell):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, has_bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, has_bias=False)
        self.sigmod = nn.Sigmoid()

    def construct(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)


class SpatialAttention(nn.Cell):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, has_bias=False,pad_mode='pad')
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        avg_out = F.mean(x, axis=1, keep_dims=True)
        max_out = F.max(x, axis=1, keepdims=True)[0]

        x = F.cat([avg_out, max_out], axis=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def conv2d_bn(in_channels, out_channels):
    return nn.SequentialCell(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,pad_mode='pad'),
        nn.PReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Dropout(p=0.6),
    )

      
class CNNDecoderDeepSuper(nn.Cell):
    def __init__(self):
        super().__init__()
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa5 = SpatialAttention()

        self.sigmoid = nn.Sigmoid()

        # branch1
        self.ca1 = ChannelAttention(in_channels=1024)
        self.bn_ca1 = nn.BatchNorm2d(1024)
        self.o1_conv1 = conv2d_bn(1024, 512)
        self.o1_conv2 = conv2d_bn(512, 512)
        self.bn_sa1 = nn.BatchNorm2d(512)
        self.o1_conv3 = nn.Conv2d(512, 1, 1)
        self.trans_conv1 = nn.Conv2dTranspose(512, 512, kernel_size=2, stride=2)

        # branch 2
        self.ca2 = ChannelAttention(in_channels=1536)
        self.bn_ca2 = nn.BatchNorm2d(1536)
        self.o2_conv1 = conv2d_bn(1536, 512)
        self.o2_conv2 = conv2d_bn(512, 256)
        self.o2_conv3 = conv2d_bn(256, 256)
        self.bn_sa2 = nn.BatchNorm2d(256)
        self.o2_conv4 = nn.Conv2d(256, 1, 1)
        self.trans_conv2 = nn.Conv2dTranspose(256, 256, kernel_size=2, stride=2)

        # branch 3
        self.ca3 = ChannelAttention(in_channels=768)
        self.o3_conv1 = conv2d_bn(768, 256)
        self.o3_conv2 = conv2d_bn(256, 128)
        self.o3_conv3 = conv2d_bn(128, 128)
        self.bn_sa3 = nn.BatchNorm2d(128)
        self.o3_conv4 = nn.Conv2d(128, 1, 1)
        self.trans_conv3 = nn.Conv2dTranspose(128, 128, kernel_size=2, stride=2)

        # branch 4
        self.ca4 = ChannelAttention(in_channels=384)
        self.o4_conv1 = conv2d_bn(384, 128)
        self.o4_conv2 = conv2d_bn(128, 64)
        self.o4_conv3 = conv2d_bn(64, 64)
        self.bn_sa4 = nn.BatchNorm2d(64)
        self.o4_conv4 = nn.Conv2d(64, 1, 1)
        self.trans_conv4 = nn.Conv2dTranspose(64, 64, kernel_size=2, stride=2)

        # branch 5
        self.ca5 = ChannelAttention(in_channels=192)
        self.o5_conv1 = conv2d_bn(192, 64)
        self.o5_conv2 = conv2d_bn(64, 32)
        self.o5_conv3 = conv2d_bn(32, 16)
        self.bn_sa5 = nn.BatchNorm2d(16)
        self.o5_conv4 = nn.Conv2d(16, 1, 1)
        
    def construct(self, feaslist):
 
        x = feaslist[4]

        x = self.o1_conv1(x)
        x = self.o1_conv2(x)
        x = self.sa1(x) * x
        x = self.bn_sa1(x)

        branch_1_out = self.sigmoid(self.o1_conv3(x))

        x = self.trans_conv1(x)
        x = F.cat((x, feaslist[3]), axis=1)
        x = self.ca2(x) * x
 
        x = self.o2_conv1(x)
        x = self.o2_conv2(x)
        x = self.o2_conv3(x)
        x = self.sa2(x) * x
        x = self.bn_sa2(x)

        branch_2_out = self.sigmoid(self.o2_conv4(x))

        x = self.trans_conv2(x)
        x = F.cat((x, feaslist[2]), axis=1)
        x = self.ca3(x) * x
        x = self.o3_conv1(x)
        x = self.o3_conv2(x)
        x = self.o3_conv3(x)
        x = self.sa3(x) * x
        x = self.bn_sa3(x)

        branch_3_out = self.sigmoid(self.o3_conv4(x))

        x = self.trans_conv3(x)
        x = F.cat((x, feaslist[1]), axis=1)
        x = self.ca4(x) * x
        x = self.o4_conv1(x)
        x = self.o4_conv2(x)
        x = self.o4_conv3(x)
        x = self.sa4(x) * x
        x = self.bn_sa4(x)

        branch_4_out = self.sigmoid(self.o4_conv4(x))

        x = self.trans_conv4(x)
        x = F.cat((x, feaslist[0]), axis=1)
        x = self.ca5(x) * x
        x = self.o5_conv1(x)
        x = self.o5_conv2(x)
        x = self.o5_conv3(x)
        x = self.sa5(x) * x
        x = self.bn_sa5(x)

        branch_5_out = self.sigmoid(self.o5_conv4(x))

        #return branch_5_out, branch_4_out, branch_3_out, branch_2_out, branch_1_out
        return branch_5_out


 