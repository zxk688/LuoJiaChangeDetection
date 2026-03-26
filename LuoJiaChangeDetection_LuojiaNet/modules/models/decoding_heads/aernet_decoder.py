import luojianet.nn as nn
import luojianet.ops as F
from luojianet import Parameter,Tensor

class h_sigmoid(nn.Module):
    def __init__(self):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = F.cat([x_h, x_w], axis=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = F.split(y, [h, w], axis=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h


        return out
class DWConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            pad_mode='pad',
            padding=1,
        )
        self.point_conv = nn.SequentialCell(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU())


    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class BAM(nn.Module):
    """ Basic self-attention module
    """

    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(BAM, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in //8
        self.activation = activation
        self.ds = ds  #
        self.pool = nn.AvgPool2d(self.ds)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(F.zeros(1))

        self.softmax = nn.Softmax(axis=-1)  #

    def forward(self, input):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        x = self.pool(input)
        m_batchsize, C, width, height = x.shape
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        energy = F.bmm(proj_query, proj_key)  # transpose check
        energy = (self.key_channel**-.5) * energy

        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = F.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = F.interpolate(out, [width*self.ds,height*self.ds])
        out = out + input

        return out

class decoder_block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(decoder_block, self).__init__()

        self.de_block1 = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        self.de_block2 = DWConv(out_channels, out_channels)

        self.att = CoordAtt(out_channels,out_channels)

        self.de_block3 = DWConv(out_channels, out_channels)

        self.de_block4 = nn.Conv2d(out_channels, 1, 1)

        self.de_block5 = nn.Conv2dTranspose(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, input1, input, input2):

        x0 = F.cat((input1, input, input2), axis=1)
        x0 = self.de_block1(x0)
        x = self.de_block2(x0)
        x = self.att(x)
        x = self.de_block3(x)
        x = x + x0
        al = self.de_block4(x)
        result = self.de_block5(x)

        return al, result

class ref_seg(nn.Module):
    def __init__(self):
        super(ref_seg, self).__init__()
        self.dir_head = nn.SequentialCell(nn.Conv2d(32, 32, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 8, 1, 1))
        self.conv0=nn.Conv2d(1,8,3,1,'pad',1,has_bias=False)
        self.conv0.weight = Parameter(Tensor([[[[0,0, 0], [1, 0, 0], [0, 0, 0]]],
                                                       [[[1,0, 0], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0,1, 0], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0,0, 1], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0,0, 0], [0, 0, 1], [0, 0, 0]]],
                                                       [[[0,0, 0], [0, 0, 0], [0, 0, 1]]],
                                                       [[[0,0, 0], [0, 0, 0], [0, 1, 0]]],
                                                       [[[0,0, 0], [0, 0, 0], [1, 0, 0]]]]).float())
    def forward(self,x,masks_pred,edge_pred):
        direc_pred = self.dir_head(x)
        direc_pred=F.softmax(direc_pred,1)
        edge_mask=1*(F.sigmoid(edge_pred)>0.5)
        refined_mask_pred=(self.conv0(masks_pred)*direc_pred).sum(1).unsqueeze(1)*edge_mask+masks_pred*(1-edge_mask)
        return refined_mask_pred
    
class AERNetDecoder(nn.Module):
    def __init__(self):
        super(AERNetDecoder, self).__init__()

        self.bam = BAM(1024)
        self.db1 = nn.SequentialCell(
            nn.Conv2d(1024, 512, 1), nn.BatchNorm2d(512), nn.ReLU(),
            DWConv(512, 512),
            nn.Conv2dTranspose(512, 512, kernel_size=2, stride=2)
        )

        self.db2 = decoder_block(1024, 256)
        self.db3 = decoder_block(512, 128)
        self.db4 = decoder_block(256, 64)
        self.db5 = decoder_block(192, 32)

        self.classifier1 = nn.SequentialCell(
            nn.Conv2d(32, 32, 3, 1,'pad', 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, 1))

        self.classifier2 = nn.SequentialCell(
            nn.Conv2d(32+1, 32, 3, 1,'pad', 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, 1))
        # self.interpo = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)
        self.refine = ref_seg()

    def forward(self, inputs, fi_out):
        input1,input2 = inputs
        x=fi_out[4]
        x = self.bam(x)
        x = self.db1(x)
        #512*16*16
        al1,x = self.db2(input1[3], x, input2[3])   #256*32*32
        al2,x = self.db3(input1[2], x, input2[2])   #128*64*64
        al3,x = self.db4(input1[1], x, input2[1])   #64*128*128
        al4,x = self.db5(input1[0], x, input2[0])   #32*256*256

        edge = self.classifier1(x)
        al4 = F.interpolate(al4,size=(al4.shape[-1]*2,al4.shape[-1]*2),mode='bilinear')
        seg = self.classifier2(F.cat((x, al4), 1))
        result = self.refine(x, seg, edge)

        return result
