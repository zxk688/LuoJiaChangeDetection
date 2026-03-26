import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )
class FDCNNDecoder(nn.Module):
    def __init__(self, filters=[64,128,256,512]):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up = double_conv(sum(filters), 64)

    def forward(self, feas_list):
        d1 = self.upsample(feas_list[1])
        d2 = self.upsample(self.upsample(feas_list[2]))
        d3 = self.upsample(self.upsample(self.upsample(feas_list[3])))
        
        x = torch.cat([feas_list[0], d1,d2,d3], dim=1)
        x= self.dconv_up(x)
        # print(x.shape)
        return x
