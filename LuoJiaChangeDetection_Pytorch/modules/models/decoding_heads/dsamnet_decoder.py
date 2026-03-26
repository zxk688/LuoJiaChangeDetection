import torch
import torch.nn as nn
import torch.nn.functional as F


class DR(nn.Module):
    def __init__(self, in_d, out_d):
        super(DR, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2d(self.in_d, self.out_d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio, kernel_size):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x
    
class Decoder(nn.Module):
    def __init__(self, fc, BatchNorm):
        super(Decoder, self).__init__()
        self.fc = fc
        self.dr2 = DR(64, 96)
        self.dr3 = DR(128, 96)
        self.dr4 = DR(256, 96)
        self.dr5 = DR(64, 96)
        self.last_conv = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, self.fc, kernel_size=1, stride=1, padding=0, bias=False),
                                       BatchNorm(self.fc),
                                       nn.ReLU(),
                                       )

    def forward(self, x, low_level_feat2, low_level_feat3, low_level_feat4):

        x2 = self.dr2(low_level_feat2)
        
        x3 = self.dr3(low_level_feat3)
        
        x4 = self.dr4(low_level_feat4)
       
        x = self.dr5(x)


        x2 = F.interpolate(x, size=x.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x.size()[2:], mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=x.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x, x2, x3, x4), dim=1)

        x = self.last_conv(x)
        
        return x

def build_decoder(fc, BatchNorm):
    return Decoder(fc, BatchNorm)


class DSAMNetDecoder(nn.Module):
    def __init__(self,  ratio = 8, kernel = 7,  f_c=64):
        super(DSAMNetDecoder, self).__init__()
        self.decoder = build_decoder(f_c, nn.BatchNorm2d)
        self.cbam0 = CBAM(f_c, ratio, kernel)
        self.cbam1 = CBAM(f_c, ratio, kernel)
        
    def forward(self, inputs):
        input1, input2 = inputs
        
        x1 = self.decoder(input1[0], input1[1], input1[2], input1[3])

        x2 = self.decoder(input2[0], input2[1], input2[2], input2[3])

        x1 = self.cbam0(x1)
        x2 = self.cbam1(x2)
        return x1, x2

