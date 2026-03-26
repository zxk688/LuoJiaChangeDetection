import torch.nn as nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BiSRNetDecoder(nn.Module):
    def __init__(self,num_classes=7):
        super(BiSRNetDecoder, self).__init__()        
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)
        self.classifier1 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, kernel_size=1))
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )
        
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))   
        return nn.Sequential(*layers)

    def forward(self,inputs,fi_out): 
        x1,x2 = inputs
        x_size = x1[0].size() 
        x = self.resCD(fi_out[1])
        change = self.classifierCD(x)
        out1 = self.classifier1(x1[1])
        out2 = self.classifier2(x2[1])
        out1 = F.interpolate(out1, x_size[2:], mode='bilinear')
        out2 = F.interpolate(out2, x_size[2:], mode='bilinear')
        change = F.interpolate(change, x_size[2:], mode='bilinear')
        #return change
        return change,out1,out2 