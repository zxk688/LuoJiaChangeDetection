
import luojianet.nn as nn
import luojianet.ops as F


def normlize(input):
    output = (input-F.min(input)[0]) /(F.max(input)[0]-F.min(input)[0])
    return output

class DistanceHead(nn.Module):
    def __init__(self, n_class=1):
        super(DistanceHead, self).__init__()
    def forward(self, inputs):
        x1, x2 = inputs
        dist = (x1 - x2).pow(2).sqrt()
        
        dist = F.interpolate(dist, size=256, mode='bilinear', align_corners=True)

        dist = normlize(F.sum(dist,1).unsqueeze(1))
        return dist