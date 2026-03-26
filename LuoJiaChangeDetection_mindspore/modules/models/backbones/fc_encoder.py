import mindspore.nn as nn
import mindspore.ops as F

class FCSiamEncoder(nn.Cell):
    def __init__(self, in_channels=3, base_channel=16):
        super(FCSiamEncoder, self).__init__()
        filters = [base_channel, base_channel * 2, base_channel * 4, 
                   base_channel * 8, base_channel * 16]

        self.conv11 = nn.Conv2d(in_channels, filters[0], kernel_size=3,padding=1,pad_mode='pad')
        self.bn11 = nn.BatchNorm2d(filters[0])
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(filters[0], filters[0], kernel_size=3,padding=1,pad_mode='pad')
        self.bn12 = nn.BatchNorm2d(filters[0])
        self.do12 = nn.Dropout2d(p=0.2)

        self.conv21 = nn.Conv2d(filters[0], filters[1], kernel_size=3,padding=1,pad_mode='pad')
        self.bn21 = nn.BatchNorm2d(filters[1])
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(filters[1], filters[1], kernel_size=3,padding=1,pad_mode='pad')
        self.bn22 = nn.BatchNorm2d(filters[1])
        self.do22 = nn.Dropout2d(p=0.2)

        self.conv31 = nn.Conv2d(filters[1], filters[2], kernel_size=3,padding=1,pad_mode='pad')
        self.bn31 = nn.BatchNorm2d(filters[2])
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(filters[2], filters[2], kernel_size=3,padding=1,pad_mode='pad')
        self.bn32 = nn.BatchNorm2d(filters[2])
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(filters[2], filters[2], kernel_size=3,padding=1,pad_mode='pad')
        self.bn33 = nn.BatchNorm2d(filters[2])
        self.do33 = nn.Dropout2d(p=0.2)

        self.conv41 = nn.Conv2d(filters[2], filters[3], kernel_size=3,padding=1,pad_mode='pad')
        self.bn41 = nn.BatchNorm2d(filters[3])
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(filters[3], filters[3], kernel_size=3,padding=1,pad_mode='pad')
        self.bn42 = nn.BatchNorm2d(filters[3])
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(filters[3], filters[3], kernel_size=3,padding=1,pad_mode='pad')
        self.bn43 = nn.BatchNorm2d(filters[3])
        self.do43 = nn.Dropout2d(p=0.2)

    def construct(self,x):
        x1 = x[:,0]
        x2 = x[:,1]
        x11 = F.relu(self.bn11(self.conv11(x1)))
        x11 = self.do11(x11)
        x12_1 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_1, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_1 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_1, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_1 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_1, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_1 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p1 = F.max_pool2d(x43_1, kernel_size=2, stride=2)

        ####################################################
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x2))))
        x12_2 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_2, kernel_size=2, stride=2)


        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_2 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_2, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_2 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_2, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_2 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p2 = F.max_pool2d(x43_2, kernel_size=2, stride=2)

        return (x12_1,x22_1,x33_1,x43_1,x4p1,),(x12_2,x22_2,x33_2,x43_2,x4p2,)

class FCEFEncoder(nn.Cell):
    def __init__(self, in_channels=6, base_channel=16):
        super(FCEFEncoder, self).__init__()
        filters = [base_channel, base_channel * 2, base_channel * 4, 
                   base_channel * 8, base_channel * 16]

        self.conv11 = nn.Conv2d(in_channels, filters[0], kernel_size=3,padding=1,pad_mode='pad')
        self.bn11 = nn.BatchNorm2d(filters[0])
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(filters[0], filters[0], kernel_size=3,padding=1,pad_mode='pad')
        self.bn12 = nn.BatchNorm2d(filters[0])
        self.do12 = nn.Dropout2d(p=0.2)

        self.conv21 = nn.Conv2d(filters[0], filters[1], kernel_size=3,padding=1,pad_mode='pad')
        self.bn21 = nn.BatchNorm2d(filters[1])
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(filters[1], filters[1], kernel_size=3,padding=1,pad_mode='pad')
        self.bn22 = nn.BatchNorm2d(filters[1])
        self.do22 = nn.Dropout2d(p=0.2)

        self.conv31 = nn.Conv2d(filters[1], filters[2], kernel_size=3,padding=1,pad_mode='pad')
        self.bn31 = nn.BatchNorm2d(filters[2])
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(filters[2], filters[2], kernel_size=3,padding=1,pad_mode='pad')
        self.bn32 = nn.BatchNorm2d(filters[2])
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(filters[2], filters[2], kernel_size=3,padding=1,pad_mode='pad')
        self.bn33 = nn.BatchNorm2d(filters[2])
        self.do33 = nn.Dropout2d(p=0.2)

        self.conv41 = nn.Conv2d(filters[2], filters[3], kernel_size=3,padding=1,pad_mode='pad')
        self.bn41 = nn.BatchNorm2d(filters[3])
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(filters[3], filters[3], kernel_size=3,padding=1,pad_mode='pad')
        self.bn42 = nn.BatchNorm2d(filters[3])
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(filters[3], filters[3], kernel_size=3,padding=1,pad_mode='pad')
        self.bn43 = nn.BatchNorm2d(filters[3])
        self.do43 = nn.Dropout2d(p=0.2)

    def construct(self,x):
        input1 = x[:,0]
        input2 = x[:,1]
        x1 = F.concat([input1, input2], axis=1)
        x11 = self.do11(F.relu(self.bn11(self.conv11(x1))))
        x12_1 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_1, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_1 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_1, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_1 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_1, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_1 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p1 = F.max_pool2d(x43_1, kernel_size=2, stride=2)
        
        return (x12_1,x22_1,x33_1,x43_1,x4p1,)
    