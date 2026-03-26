import luojianet.nn as nn
import luojianet.ops as F
from .pvtv2 import pvt_v2_b1
from .resnet import resnet18

class DualCNNandTransEncoder(nn.Module):
    def __init__(self):
        super(DualCNNandTransEncoder, self).__init__()
        self.resnet = resnet18()
        self.backbone = pvt_v2_b1() 
    def forward(self, x):
        imgs1 = x[:,0]  # 10x3x256x256
        imgs2 = x[:,1] 
        pvt = self.backbone(imgs1)
        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        pvt_img2 = self.backbone(imgs2)
        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)
        return (c0,c1,c2,c3)  , (c0_img2,c1_img2,c2_img2,c3_img2) , pvt, pvt_img2
