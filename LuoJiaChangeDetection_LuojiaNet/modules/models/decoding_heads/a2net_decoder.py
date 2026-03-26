import luojianet.nn as nn
import luojianet.ops as F

class SupervisedAttentionModule(nn.Module):
    def __init__(self, mid_d):
        super(SupervisedAttentionModule, self).__init__()
        self.mid_d = mid_d
        # fusion
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)
        self.conv_context = nn.SequentialCell(
            nn.Conv2d(2, self.mid_d, kernel_size=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU()
        )
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1,pad_mode='pad', padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU()
        )

    def forward(self, x):
        mask = self.cls(x)
        mask_f = F.Sigmoid()(mask)
        mask_b = 1 - mask_f
        context = F.cat([mask_f, mask_b], axis=1)
        context = self.conv_context(context)
        x = x.mul(context)
        x_out = self.conv2(x)

        return x_out, mask
    
class Decoder(nn.Module):
    def __init__(self, mid_d=320):
        super(Decoder, self).__init__()
        self.mid_d = mid_d
        # fusion
        self.sam_p5 = SupervisedAttentionModule(self.mid_d)
        self.sam_p4 = SupervisedAttentionModule(self.mid_d)
        self.sam_p3 = SupervisedAttentionModule(self.mid_d)
        self.conv_p4 = nn.SequentialCell(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, pad_mode='pad',padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU()
        )
        self.conv_p3 = nn.SequentialCell(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, pad_mode='pad',padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU()
        )
        self.conv_p2 = nn.SequentialCell(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, pad_mode='pad',padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU()
        )
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)

    def forward(self, d2, d3, d4, d5):
        # high-level
        p5, mask_p5 = self.sam_p5(d5)
        p4 = self.conv_p4(d4 + F.interpolate(p5, size=(p5.shape[-1]*2,p5.shape[-1]*2), mode='bilinear'))

        p4, mask_p4 = self.sam_p4(p4)
        p3 = self.conv_p3(d3 + F.interpolate(p4, size=(p4.shape[-1]*2,p4.shape[-1]*2), mode='bilinear'))

        p3, mask_p3 = self.sam_p3(p3)
        p2 = self.conv_p2(d2 + F.interpolate(p3, size=(p3.shape[-1]*2,p3.shape[-1]*2), mode='bilinear'))
        mask_p2 = self.cls(p2)

        return p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5


class A2NetDecoder(nn.Module):
    def __init__(self,en_d):
        super(A2NetDecoder, self).__init__()
        self.en_d = en_d
        self.decoder = Decoder(self.en_d * 2)
    def forward(self, feas): 
        p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5 = self.decoder(feas[0], feas[1], feas[2], feas[3])
        # change map
        mask_p2 = F.interpolate(mask_p2, size=(mask_p2.shape[-1]*4,mask_p2.shape[-1]*4), mode='bilinear')
        mask_p2 = F.Sigmoid()(mask_p2)
        mask_p3 = F.interpolate(mask_p3, size=(mask_p3.shape[-1]*8,mask_p3.shape[-1]*8), mode='bilinear')
        mask_p3 = F.Sigmoid()(mask_p3)
        mask_p4 = F.interpolate(mask_p4, size=(mask_p4.shape[-1]*16,mask_p4.shape[-1]*16), mode='bilinear')
        mask_p4 = F.Sigmoid()(mask_p4)
        mask_p5 = F.interpolate(mask_p5, size=(mask_p5.shape[-1]*32,mask_p5.shape[-1]*32), mode='bilinear')
        mask_p5 = F.Sigmoid()(mask_p5)
        return mask_p5