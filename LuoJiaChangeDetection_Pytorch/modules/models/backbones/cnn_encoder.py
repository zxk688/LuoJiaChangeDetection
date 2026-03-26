import torch
import torch.nn as nn

class FEC(nn.Module):
    """feature extraction cell"""
    #convolutional block
    def __init__(self, in_ch, mid_ch, out_ch):
        super(FEC, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1,bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channel=40):
        super(CNNEncoder, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = base_channel  # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        self.conv0_0 = nn.Conv2d(in_channels, n1, kernel_size=5, padding=2, stride=1)
        self.conv0 = FEC(filters[0], filters[0], filters[0])
        self.conv2 = FEC(filters[0], filters[1], filters[1])
        self.conv4 = FEC(filters[1], filters[2], filters[2])
        self.conv5 = FEC(filters[2], filters[3], filters[3])
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    def forward(self,x1,x2):  
        x1 = self.conv0(self.conv0_0(x1)) # Output of the first scale
        x3 = self.conv2(self.pool(x1))
        x4 = self.conv4(self.pool(x3))
        A_F4 = self.conv5(self.pool(x4))

        x2 = self.conv0(self.conv0_0(x2))
        x5 = self.conv2(self.pool(x2))
        x6 = self.conv4(self.pool(x5))
        A_F8 = self.conv5(self.pool(x6))
        return (A_F4, x4, x3, x1, ), (A_F8, x6, x5, x2, )
