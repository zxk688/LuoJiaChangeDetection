import torch.nn.functional as F
import torch.nn as nn
import torch

def normlize(input):
    output = (input-torch.min(input)) /(torch.max(input)-torch.min(input))
    return output

class DistanceHead(nn.Module):
    def __init__(self, n_class=1):
        super(DistanceHead, self).__init__()

    def forward(self, inputs):
        x1, x2 = inputs
        dist = F.pairwise_distance(x1, x2, keepdim=True) # channel = 1
        dist = F.interpolate(dist, size=256, mode='bilinear', align_corners=True)

        dist = normlize(torch.sum(dist,1).unsqueeze(1))
        return dist