import luojianet.nn as nn

class FEC(nn.Module):
    """feature extraction cell"""
    #convolutional block
    def __init__(self, in_ch, mid_ch, out_ch):
        super(FEC, self).__init__()
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1,has_bias=True,pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, has_bias=False,pad_mode='pad')
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
    def __init__(self, in_channels,base_channel):
        super(CNNEncoder, self).__init__()
        nn.Module.dump_patches = True
        filters = [base_channel, base_channel * 2, base_channel * 4, base_channel * 8]
        self.conv0_0 = nn.Conv2d(in_channels, base_channel, kernel_size=5, padding=2, stride=1,pad_mode='pad')
        self.conv0 = FEC(filters[0], filters[0], filters[0])
        self.conv2 = FEC(filters[0], filters[1], filters[1])
        self.conv4 = FEC(filters[1], filters[2], filters[2])
        self.conv5 = FEC(filters[2], filters[3], filters[3])
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    def forward(self,x): 
        # split_tensors = F.Split(1,2)(x)
        # x1 = split_tensors[0]  # 10x3x256x256
        # x2 = split_tensors[1] 
        x1 = x[:,0]
        x2 = x[:,1]
        x1 = self.conv0(self.conv0_0(x1)) # Output of the first scale
        x3 = self.conv2(self.pool(x1))
        x4 = self.conv4(self.pool(x3))
        A_F4 = self.conv5(self.pool(x4))

        x2 = self.conv0(self.conv0_0(x2))
        x5 = self.conv2(self.pool(x2))
        x6 = self.conv4(self.pool(x5))
        A_F8 = self.conv5(self.pool(x6))
        return (A_F4, x4, x3, x1,), (A_F8, x6, x5, x2, )
