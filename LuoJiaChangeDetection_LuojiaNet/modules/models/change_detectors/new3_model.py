import luojianet
import luojianet.nn as nn
import luojianet.ops as F
from ..backbones.mobilenet import mobilenet_v2

class NeighborFeatureAggregation(nn.Module):
    def __init__(self, in_d=None, out_d=64):
        super(NeighborFeatureAggregation, self).__init__()
        if in_d is None:
            in_d = [16, 24, 32, 96, 320]
        self.in_d = in_d
        self.mid_d = out_d // 2
        self.out_d = out_d
        # scale 2
        self.conv_scale2_c2 = nn.SequentialCell(
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU()
        )
        self.conv_scale2_c3 = nn.SequentialCell(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU()
        )
        self.conv_aggregation_s2 = FeatureFusionModule(self.mid_d * 2, self.in_d[1], self.out_d)
        # scale 3
        self.conv_scale3_c2 = nn.SequentialCell(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU()
        )
        self.conv_scale3_c3 = nn.SequentialCell(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU()
        )
        self.conv_scale3_c4 = nn.SequentialCell(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU()
        )
        self.conv_aggregation_s3 = FeatureFusionModule(self.mid_d * 3, self.in_d[2], self.out_d)
        # scale 4
        self.conv_scale4_c3 = nn.SequentialCell(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU()
        )
        self.conv_scale4_c4 = nn.SequentialCell(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU()
        )
        self.conv_scale4_c5 = nn.SequentialCell(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU()
        )
        self.conv_aggregation_s4 = FeatureFusionModule(self.mid_d * 3, self.in_d[3], self.out_d)
        # scale 5
        self.conv_scale5_c4 = nn.SequentialCell(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU()
        )
        self.conv_scale5_c5 = nn.SequentialCell(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU()
        )
        self.conv_aggregation_s5 = FeatureFusionModule(self.mid_d * 2, self.in_d[4], self.out_d)

    def forward(self, c2, c3, c4, c5):
        # scale 2
        c2_s2 = self.conv_scale2_c2(c2)

        c3_s2 = self.conv_scale2_c3(c3)

        c3_s2 = F.interpolate(c3_s2, size=(c3_s2.shape[-1]*2,c3_s2.shape[-1]*2),mode='bilinear')
        s2 = self.conv_aggregation_s2(F.cat([c2_s2, c3_s2], axis=1), c2)
        # scale 3
        c2_s3 = self.conv_scale3_c2(c2)

        c3_s3 = self.conv_scale3_c3(c3)

        c4_s3 = self.conv_scale3_c4(c4)
        c4_s3 = F.interpolate(c4_s3, size=(c4_s3.shape[-1]*2,c4_s3.shape[-1]*2), mode='bilinear')

        s3 = self.conv_aggregation_s3(F.cat([c2_s3, c3_s3, c4_s3], axis=1), c3)
        # scale 4
        c3_s4 = self.conv_scale4_c3(c3)

        c4_s4 = self.conv_scale4_c4(c4)

        c5_s4 = self.conv_scale4_c5(c5)
        c5_s4 = F.interpolate(c5_s4, size=(c5_s4.shape[-1]*2,c5_s4.shape[-1]*2), mode='bilinear')

        s4 = self.conv_aggregation_s4(F.cat([c3_s4, c4_s4, c5_s4], axis=1), c4)
        # scale 5
        c4_s5 = self.conv_scale5_c4(c4)

        c5_s5 = self.conv_scale5_c5(c5)

        s5 = self.conv_aggregation_s5(F.cat([c4_s5, c5_s5], axis=1), c5)

        return s2, s3, s4, s5



class FeatureFusionModule(nn.Module):
    def __init__(self, fuse_d, id_d, out_d):
        super(FeatureFusionModule, self).__init__()
        self.fuse_d = fuse_d
        self.id_d = id_d
        self.out_d = out_d
        self.conv_fuse = nn.SequentialCell(
            nn.Conv2d(self.fuse_d, self.out_d, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(),
            nn.Conv2d(self.out_d, self.out_d, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(self.out_d)
        )
        self.conv_identity = nn.Conv2d(self.id_d, self.out_d, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, c_fuse, c):
        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse + self.conv_identity(c))

        return c_out


class TemporalFeatureFusionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(TemporalFeatureFusionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.relu = nn.ReLU()
        # branch 1
        self.conv_branch1 = nn.SequentialCell(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, pad_mode='pad', padding=7, dilation=7),
            nn.BatchNorm2d(self.in_d)
        )
        # branch 2
        self.conv_branch2 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch2_f = nn.SequentialCell(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, pad_mode='pad', padding=5, dilation=5),
            nn.BatchNorm2d(self.in_d)
        )
        # branch 3
        self.conv_branch3 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch3_f = nn.SequentialCell(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, pad_mode='pad', padding=3, dilation=3),
            nn.BatchNorm2d(self.in_d)
        )
        # branch 4
        self.conv_branch4 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch4_f = nn.SequentialCell(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, pad_mode='pad', padding=1, dilation=1),
            nn.BatchNorm2d(self.out_d)
        )
        self.conv_branch5 = nn.Conv2d(self.in_d, self.out_d, kernel_size=1)

    def forward(self, x1,x2):

        # temporal fusion
        x = F.abs(x1 - x2)
        # branch 1
        x_branch1 = self.conv_branch1(x)
        # branch 2
        x_branch2 = self.relu(self.conv_branch2(x) + x_branch1)
        x_branch2 = self.conv_branch2_f(x_branch2)
        # branch 3
        x_branch3 = self.relu(self.conv_branch3(x) + x_branch2)
        x_branch3 = self.conv_branch3_f(x_branch3)
        # branch 4
        x_branch4 = self.relu(self.conv_branch4(x) + x_branch3)
        x_branch4 = self.conv_branch4_f(x_branch4)
        x_out = self.relu(self.conv_branch5(x) + x_branch4)

        return x_out


class TemporalFusionModule(nn.Module):
    def __init__(self, in_d=32, out_d=32):
        super(TemporalFusionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        # fusion
        self.tffm_x2 = TemporalFeatureFusionModule(self.in_d, self.out_d)
        self.tffm_x3 = TemporalFeatureFusionModule(self.in_d, self.out_d)
        self.tffm_x4 = TemporalFeatureFusionModule(self.in_d, self.out_d)
        self.tffm_x5 = TemporalFeatureFusionModule(self.in_d, self.out_d)

    def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):
        # temporal fusion
        c2 = self.tffm_x2(x1_2, x2_2)
        c3 = self.tffm_x3(x1_3, x2_3)
        c4 = self.tffm_x4(x1_4, x2_4)
        c5 = self.tffm_x5(x1_5, x2_5)

        return c2, c3, c4, c5


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


class new3net(nn.Module):
    def __init__(self, input_nc=3, output_nc=1,changedetector_cfg: dict = None):
        super(new3net, self).__init__()
        self.backbone = mobilenet_v2(pretrained=False)
        channles = [16, 24, 32, 96, 320]
        self.en_d = 32
        self.mid_d = self.en_d * 2
        self.swa = NeighborFeatureAggregation(channles, self.mid_d)
        self.tfm = TemporalFusionModule(self.mid_d, self.en_d * 2)
        self.decoder = Decoder(self.en_d * 2)

    def forward(self, x):
        x1 = x[:,0]
        x2 = x[:,1]
        # forward backbone resnet
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)
        # aggregation
        x1_2, x1_3, x1_4, x1_5 = self.swa(x1_2, x1_3, x1_4, x1_5)
        x2_2, x2_3, x2_4, x2_5 = self.swa(x2_2, x2_3, x2_4, x2_5)
        # temporal fusion
        c2, c3, c4, c5 = self.tfm(x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5)
        # fpn
        p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5 = self.decoder(c2, c3, c4, c5)

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