import luojianet
import luojianet.nn as nn
import luojianet.ops as F
from luojianet.nn import ReplicationPad2d

class UpMask(nn.Module):
    def __init__(
        self,
        scale_factor: float,
        nin: int,
        nout: int,
    ):
        super().__init__()
        self._upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True,recompute_scale_factor = True)
        self._convolution = nn.SequentialCell(
            nn.Conv2d(nin, nin, 3, 1, group=nin, padding=1,pad_mode='pad'),
            nn.PReLU(),
            nn.InstanceNorm2d(nin),
            nn.Conv2d(nin, nout, kernel_size=1, stride=1),
            nn.PReLU(),
            nn.InstanceNorm2d(nout),
        )

    def forward(self, x, y):
        x = self._upsample(x)
        if y is not None:
            x = x * y
        return self._convolution(x)


class TinyCDDecoder(nn.Module): 
    def __init__(self):
        super().__init__()
        self._up =nn.CellList(
            [
                UpMask(2.0, 56, 64),
                UpMask(2.0, 64, 64),
                UpMask(2.0, 64, 32),
            ]
        )
    def forward(self, features):
        upping = features[-1]
        for i, j in enumerate(range(-2, -5, -1)):
            upping = self._up[i](upping, features[j])
            
        return upping
        #return (upping,)
