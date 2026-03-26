import torch
from torch import nn
import torch.nn.functional as F

class convx2(nn.Module):
    def __init__(self, *ch):
        super(convx2, self).__init__()
        self.conv_number = len(ch) - 1
        print(len(ch))
        self.model = nn.Sequential()
        for i in range(self.conv_number):
            self.model.add_module('conv{0}'.format(i), nn.Conv2d(ch[i], ch[i + 1], 11, 1, 5))
            self.model.add_module('bn{0}'.format(i),nn.BatchNorm2d(ch[i+1]))
            self.model.add_module('relu{0}'.format(i),nn.ReLU())

    def forward(self, x):
        y = self.model(x)
        return y


class SUNetDecoder(nn.Module):
    def __init__(self):
        super(SUNetDecoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv5 = convx2(*[256, 128, 128, 64])
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv6 = convx2(*[128, 64, 64, 32])
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv7 = convx2(*[64, 32, 16])
        self.deconv4 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.conv8 = convx2(*[32, 16, 1])
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1):
        h = self.deconv1(x1[4])
        h = self.conv5(torch.cat((h, x1[3]), 1))
        h = self.deconv2(h)
        h = self.conv6(torch.cat((h, x1[2]), 1))
        h = self.deconv3(h)
        h = self.conv7(torch.cat((h, x1[1]), 1))
        h = self.deconv4(h)
        h = self.conv8(torch.cat((h, x1[0]), 1))
        out = self.sigmoid(h)
        return out