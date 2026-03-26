import torch
import torch.nn as nn
from torchvision.models import vgg16

class vgg16_base(nn.Module):
    def __init__(self):
        super(vgg16_base, self).__init__()
        features = list(vgg16(pretrained=True).features)[:30]
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15, 22, 29}:
                results.append(x)
        return results

class VggEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vggnet = vgg16_base()
        self.t1_base = vggnet
        self.t2_base = vggnet
        
    def forward(self, t1_input, t2_input):
        t1_list = self.t1_base(t1_input)
        t2_list = self.t2_base(t2_input)

        return t1_list, t2_list