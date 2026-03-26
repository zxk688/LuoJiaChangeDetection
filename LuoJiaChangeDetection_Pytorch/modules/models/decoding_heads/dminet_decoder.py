import torch
import torch.nn as nn
import torch.nn.functional as F

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

class CrossAtt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels

        self.query1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key1   = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.value1 = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)

        self.query2 = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key2   = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.value2 = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)

        self.gamma = nn.Parameter(torch.zeros(1)) 
        self.softmax = nn.Softmax(dim = -1)

        self.conv_cat = nn.Sequential(nn.Conv2d(in_channels*2, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU()) # conv_f

    def forward(self, input1, input2):
        batch_size, channels, height, width = input1.shape
        q1 = self.query1(input1)
        k1 = self.key1(input1).view(batch_size, -1, height * width)
        v1 = self.value1(input1).view(batch_size, -1, height * width)

        q2 = self.query2(input2) 
        k2 = self.key2(input2).view(batch_size, -1, height * width)
        v2 = self.value2(input2).view(batch_size, -1, height * width)

        q = torch.cat([q1,q2],1).view(batch_size, -1, height * width).permute(0, 2, 1)
        attn_matrix1 = torch.bmm(q, k1)  
        attn_matrix1 = self.softmax(attn_matrix1)
        out1 = torch.bmm(v1, attn_matrix1.permute(0, 2, 1)) 
        out1 = out1.view(*input1.shape)
        out1 = self.gamma * out1 + input1

        attn_matrix2 = torch.bmm(q, k2) 
        attn_matrix2 = self.softmax(attn_matrix2)
        out2 = torch.bmm(v2, attn_matrix2.permute(0, 2, 1))  
        out2 = out2.view(*input2.shape)
        out2 = self.gamma * out2 + input2

        feat_sum = self.conv_cat(torch.cat([out1,out2],1))
        return feat_sum, out1, out2

class DMINetDecoder(nn.Module):
    def __init__(self):
        super(DMINetDecoder, self).__init__()
        self.cross2 = CrossAtt(256, 256) 
        self.cross3 = CrossAtt(128, 128) 
        self.cross4 = CrossAtt(64, 64) 

        self.Translayer2_1 = BasicConv2d(256,128,1)
        self.fam32_1 = decode(128,128,128) # AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(128,64,1)
        self.fam43_1 = decode(64,64,64) # AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(256,128,1)
        self.fam32_2 = decode(128,128,128)
        self.Translayer3_2 = BasicConv2d(128,64,1)
        self.fam43_2 = decode(64,64,64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
    def forward(self, inputs):
        feas1, feas2 = inputs
        cross_result2, cur1_2, cur2_2 = self.cross2(feas1[-1], feas2[-1])
        cross_result3, cur1_3, cur2_3 = self.cross3(feas1[-2], feas2[-2])
        cross_result4, cur1_4, cur2_4 = self.cross4(feas1[-3], feas2[-3]) 

        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))

        out4_up = self.upsamplex4(out4)
        #return (out4_up,)
        return out4_up