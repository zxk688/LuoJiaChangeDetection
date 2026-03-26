from luojianet import nn
import luojianet.ops as F

class convx3(nn.Module):
    def __init__(self, *ch):
        super(convx3, self).__init__()
        self.conv_number = len(ch) - 1

        self.model = nn.SequentialCell(
            nn.Conv2d(ch[0], ch[0 + 1], 11, 1, 'pad', 5),
            nn.BatchNorm2d(ch[0+1]),
            nn.ReLU(),
            nn.Conv2d(ch[1], ch[1 + 1], 11, 1, 'pad', 5),
            nn.BatchNorm2d(ch[1+1]),
            nn.ReLU(),
            nn.Conv2d(ch[2], ch[2 + 1], 11, 1, 'pad', 5),
            nn.BatchNorm2d(ch[2+1]),
            nn.ReLU(),
        )
        # for i in range(self.conv_number):
        #     self.model.insert_child_to_cell('conv{0}'.format(i), nn.Conv2d(ch[i], ch[i + 1], 11, 1,'pad', 5))
        #     self.model.insert_child_to_cell('bn{0}'.format(i),nn.BatchNorm2d(ch[i+1]))
        #     self.model.insert_child_to_cell('relu{0}'.format(i),nn.ReLU())

    def forward(self, x):
        y = self.model(x)
        return y


class convx2(nn.Module):
    def __init__(self, *ch):
        super(convx2, self).__init__()
        self.conv_number = len(ch) - 1

        self.model = nn.SequentialCell(
            nn.Conv2d(ch[0], ch[0 + 1], 11, 1, 'pad', 5),
            nn.BatchNorm2d(ch[0+1]),
            nn.ReLU(),
            nn.Conv2d(ch[1], ch[1 + 1], 11, 1, 'pad', 5),
            nn.BatchNorm2d(ch[1+1]),
            nn.ReLU(),
            
        )
        # for i in range(self.conv_number):
        #     self.model.insert_child_to_cell('conv{0}'.format(i), nn.Conv2d(ch[i], ch[i + 1], 11, 1,'pad', 5))
        #     self.model.insert_child_to_cell('bn{0}'.format(i),nn.BatchNorm2d(ch[i+1]))
        #     self.model.insert_child_to_cell('relu{0}'.format(i),nn.ReLU())

    def forward(self, x):
        y = self.model(x)
        return y


class SUNetDecoder(nn.Module):
    def __init__(self):
        super(SUNetDecoder, self).__init__()
        self.deconv1 = nn.Conv2dTranspose(128, 128, kernel_size=2, stride=2)
        self.conv5 = convx3(*[256, 128, 128, 64])
        self.deconv2 = nn.Conv2dTranspose(64, 64, kernel_size=2, stride=2)
        self.conv6 = convx3(*[128, 64, 64, 32])
        self.deconv3 = nn.Conv2dTranspose(32, 32, kernel_size=2, stride=2)
        self.conv7 = convx2(*[64, 32, 16])
        self.deconv4 = nn.Conv2dTranspose(16, 16, kernel_size=2, stride=2)
        self.conv8 = convx2(*[32, 16, 1])
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1):
        h = self.deconv1(x1[4])
        h = self.conv5(F.cat((h, x1[3]), 1))
        h = self.deconv2(h)
        h = self.conv6(F.cat((h, x1[2]), 1))
        h = self.deconv3(h)
        h = self.conv7(F.cat((h, x1[1]), 1))
        h = self.deconv4(h)
        h = self.conv8(F.cat((h, x1[0]), 1))
        out = self.sigmoid(h)
        return out