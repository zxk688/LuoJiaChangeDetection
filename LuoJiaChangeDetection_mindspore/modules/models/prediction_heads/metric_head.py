import mindspore.nn as nn
import mindspore.ops as F

class DistanceHead(nn.Cell):
    def __init__(self, n_class=1):
        super(DistanceHead, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def construct(self, inputs):
        x1, x2 = inputs
        
        # 计算特征距离，得到原始的 logits
        dist = (x1 - x2).pow(2).sum(axis=1, keepdims=True).sqrt()
        
        # 上采样到目标尺寸
        dist = F.interpolate(dist, size=256, mode='bilinear', align_corners=True)
        output = self.sigmoid(dist)
        
        return output