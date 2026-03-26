import mindspore
import mindspore.nn as nn
import mindspore.ops as F


def double_conv(in_channels, out_channels):
    return nn.SequentialCell(
        nn.Conv2d(in_channels, out_channels, 3, padding=1,pad_mode='pad'),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1,pad_mode='pad'),
        nn.ReLU()
    )
class FDCNNDecoder(nn.Cell):
    def __init__(self,filters,n_class=1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True,recompute_scale_factor=True)
        self.dconv_up = double_conv(sum(filters), 64)

    def construct(self, feas_list):
        d1 = self.upsample(feas_list[1])
        d2 = self.upsample(self.upsample(feas_list[2]))
        d3 = self.upsample(self.upsample(self.upsample(feas_list[3])))
        
        x = F.cat([feas_list[0], d1,d2,d3], axis=1)
        x= self.dconv_up(x)
        # print(x.shape)
        return x
