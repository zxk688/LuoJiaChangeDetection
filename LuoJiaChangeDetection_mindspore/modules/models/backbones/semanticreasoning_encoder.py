
import mindspore
import mindspore.nn as nn
import mindspore.ops as F
from .resnet import resnet34

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SR(nn.Cell):
    '''Spatial reasoning module'''
    #codes from DANet 'Dual attention network for scene segmentation'
    def __init__(self, in_dim):
        super(SR, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = mindspore.Parameter(F.zeros(1))

        self.softmax = nn.Softmax(axis=-1)
        
    def construct(self, x):
        ''' inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW) '''
        m_batchsize, C, height, width = x.shape
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = F.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = F.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = x+self.gamma*out        

        return out


class SemanticReasoningEncoder(nn.Cell):
    def __init__(self, in_channels=3, num_classes=7):
        super(SemanticReasoningEncoder, self).__init__()        
        self.resnet = resnet34()
        self.SiamSR = SR(128)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, pad_mode='pad', padding=3, has_bias=False)
        newconv1.weight.data[:, 0:3, :, :] = self.resnet.conv1.weight.data[:, 0:3, :, :].copy()
        # if in_channels>3:
        #   newconv1.weight.data[:, 3:in_channels, :, :].copy(self.resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
          
        self.layer0 = nn.SequentialCell(newconv1, self.resnet.bn1, self.resnet.relu)
        for n, m in self.resnet.layer3.cells_and_names():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.resnet.layer4.cells_and_names():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.head = nn.SequentialCell(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, has_bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())
                                  
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.SequentialCell(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.SequentialCell(*layers)
        
    def base_forward(self, x):
        x = self.layer0(x) #size:1/4
        x = self.resnet.maxpool(x) #size:1/4
        x = self.resnet.layer1(x) #size:1/4
        x = self.resnet.layer2(x) #size:1/8
        x = self.resnet.layer3(x) #size:1/16
        x = self.resnet.layer4(x)
        x = self.head(x)
        x = self.SiamSR(x)
        return x
    def construct(self, inputs):

        x1, x2 = inputs[:,0,:,:,:], inputs[:,1,:,:,:]
        # split_tensors = F.Split(1,2)(inputs)
        # x1 = split_tensors[0]  
        # x2 = split_tensors[1] 
  
        x1_0 = x1
        x2_0 = x2
        x1 = self.base_forward(x1)
        x2 = self.base_forward(x2)
        return (x1_0,x1,),(x2_0,x2,)
