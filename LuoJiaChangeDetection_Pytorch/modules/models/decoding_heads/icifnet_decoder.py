import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        # print("++",x.size()[1],self.inp_dim,x.size()[1],self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class decode(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, out_channel,norm_layer=nn.BatchNorm2d):
        super(decode, self).__init__()
        self.conv_d1 = nn.Conv2d(in_channel_down, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(in_channel_left, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channel*2, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(out_channel)

    def forward(self, left, down):
        down_mask = self.conv_d1(down)
        left_mask = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(left_mask * down_, inplace=True)
        else:
            z1 = F.relu(left_mask * down, inplace=True)

        if down_mask.size()[2:] != left.size()[2:]:
            down_mask = F.interpolate(down_mask, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_mask * left, inplace=True)

        out = torch.cat((z1, z2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1+in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2):

        x1 = self.up(x1)
        # input is CHW
        
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        if self.attn_block is not None:
            x2 = self.attn_block(x1, x2)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
                )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x)+self.identity(x))

class ICIFDecoder(nn.Module):
    def __init__(self, num_classes=1):
        super(ICIFDecoder, self).__init__()  
        self.final_x = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )

        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        self.up2 = Up(256, 128, 128, attn=True)
        self.up3 = Up(128, 64, 64, attn=True)

        self.up2_img2 = Up(256, 128, 128, attn=True)
        self.up3_img2 = Up(128, 64, 64, attn=True)
        
        # low-level & high-level
        self.Translayer2_g = BasicConv2d(320,128, 1)
        self.fam43_1 = decode(128,128,128)
        self.Translayer3_g = BasicConv2d(128,64, 1)
        self.fam32_1 = decode(64,64,64)

        self.Translayer2_l = BasicConv2d(320,128, 1)
        self.fam43_2 = decode(128,128,128)
        self.Translayer3_l = BasicConv2d(128,64, 1)
        self.fam32_2 = decode(64,64,64)

        self.Translayer2_g_img2 = BasicConv2d(320,128, 1)
        self.fam43_1_img2 = decode(128,128,128)
        self.Translayer3_g_img2 = BasicConv2d(128,64, 1)
        self.fam32_1_img2 = decode(64,64,64)

        self.Translayer2_l_img2 = BasicConv2d(320,128, 1)
        self.fam43_2_img2 = decode(128,128,128)
        self.Translayer3_l_img2 = BasicConv2d(128,64, 1)
        self.fam32_2_img2 = decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.sigmoid = nn.Sigmoid()              
    def forward(self, corss_out_list):

        x_up_2 = self.up2(corss_out_list[0][0], corss_out_list[1][0])
        x_up_3 = self.up3(x_up_2, corss_out_list[2][0])

        out3_g = self.fam43_1(corss_out_list[1][1], self.Translayer2_g(corss_out_list[0][1]))
        out2_g = self.fam32_1(corss_out_list[2][1], self.Translayer3_g(out3_g))

        out3_l = self.fam43_2(corss_out_list[1][2], self.Translayer2_l(corss_out_list[0][2]))
        out2_l = self.fam32_2(corss_out_list[2][2], self.Translayer3_l(out3_l))

        x_up_2_img2 = self.up2_img2(corss_out_list[3][0],corss_out_list[4][0])
        x_up_3_img2 = self.up3_img2(x_up_2_img2,corss_out_list[5][0])                                 #decoder rdio the most

        out3_g_img2 = self.fam43_1_img2(corss_out_list[4][1], self.Translayer2_g_img2(corss_out_list[3][1]))
        out2_g_img2 = self.fam32_1_img2(corss_out_list[5][1], self.Translayer3_g_img2(out3_g_img2))

        out3_l_img2 = self.fam43_2_img2(corss_out_list[4][2], self.Translayer2_l_img2(corss_out_list[3][2]))
        out2_l_img2 = self.fam32_2_img2(corss_out_list[5][2], self.Translayer3_l_img2(out3_l_img2))

        final2 = self.upsamplex4(torch.abs(out2_g-out2_g_img2))
        final1 = self.upsamplex4(torch.abs(out2_l-out2_l_img2))
        finalx = self.upsamplex4(torch.abs(x_up_3-x_up_3_img2))

        map_x = self.final_2(final2)
        map_1 = self.final_1(final1)
        map_2 = self.final_x(finalx)

        return (self.sigmoid(map_x)+self.sigmoid(map_1)+self.sigmoid(map_2))/3
 