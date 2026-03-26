import luojianet
import luojianet.nn as nn
import luojianet.ops as F
from luojianet import Parameter

class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(F.zeros(1))
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.shape
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = F.bmm(proj_query, proj_key)
        energy_new = F.max(energy, -1, keepdims=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = F.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class Conv_CAM_Layer(nn.Module):

    def __init__(self, in_ch, out_in, use_pam=False):
        super(Conv_CAM_Layer, self).__init__()
        self.attn = nn.SequentialCell(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            CAM_Module(),
            nn.Conv2d(32, out_in, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(out_in),
            nn.PReLU()
        )

    def forward(self, x):
        return self.attn(x)

class RowAttention(nn.Module):

    def __init__(self, in_dim, q_k_dim, use_pam=False):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        '''
        super(RowAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = nn.Softmax(axis=2)
        self.gamma = Parameter(F.zeros(1))

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        '''
        b, _, h, w = x.shape

        Q = self.query_conv(x)  # size = (b,c2, h,w)
        K = self.key_conv(x)  # size = (b, c2, h, w)
        V = self.value_conv(x)  # size = (b, c1,h,w)

        Q = Q.permute(0, 2, 1, 3).view(b * h, -1, w).permute(0, 2, 1)  # size = (b*h,w,c2)
        K = K.permute(0, 2, 1, 3).view(b * h, -1, w)  # size = (b*h,c2,w)
        V = V.permute(0, 2, 1, 3).view(b * h, -1, w)  # size = (b*h, c1,w)

        row_attn = F.bmm(Q, K)
        row_attn = self.softmax(row_attn)
        out = F.bmm(V, row_attn.permute(0, 2, 1))
        out = out.view(b, h, -1, w).permute(0, 2, 1, 3)
        out = self.gamma * out + x
        return out


class ColAttention(nn.Module):

    def __init__(self, in_dim, q_k_dim, use_pam=False):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        '''
        super(ColAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = nn.Softmax(axis=2)
        self.gamma = Parameter(F.zeros(1))

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        '''

        b, _, h, w = x.shape

        Q = self.query_conv(x)  # size = (b,c2, h,w)
        K = self.key_conv(x)  # size = (b, c2, h, w)
        V = self.value_conv(x)  # size = (b, c1,h,w)

        Q = Q.permute(0, 3, 1, 2).view(b * w, -1, h).permute(0, 2, 1)  # size = (b*w,h,c2)
        K = K.permute(0, 3, 1, 2).view(b * w, -1, h)  # size = (b*w,c2,h)
        V = V.permute(0, 3, 1, 2).view(b * w, -1, h)  # size = (b*w,c1,h)

        # size = (b*w,h,h) [:,i,j]
        col_attn = F.bmm(Q, K)
        col_attn = self.softmax(col_attn)
        out = F.bmm(V, col_attn.permute(0, 2, 1))
        # size = (b,c1,h,w)
        out = out.view(b, w, -1, h).permute(0, 2, 3, 1)
        out = self.gamma * out + x

        return out

class HANetDecoder(nn.Module):
    def __init__(self, base_channel):
        super(HANetDecoder,self).__init__()
        filters = [base_channel, base_channel * 2, base_channel * 4, base_channel * 8]
        self.conv6 = nn.Conv2d(sum(filters), filters[1], kernel_size=1, stride=1)

        self.conv6_1_1 = nn.Conv2d(filters[0] * 2, filters[0], padding=1, kernel_size=3, group=filters[0] // 2,dilation=1, pad_mode='pad')
        self.conv6_1_2 = nn.Conv2d(filters[0] * 2, filters[0], padding=2, kernel_size=3, group=filters[0] // 2,dilation=2, pad_mode='pad')
        self.conv6_1_3 = nn.Conv2d(filters[0] * 2, filters[0], padding=3, kernel_size=3, group=filters[0] // 2,dilation=3, pad_mode='pad')
        self.conv6_1_4 = nn.Conv2d(filters[0] * 2, filters[0], padding=4, kernel_size=3, group=filters[0] // 2,dilation=4, pad_mode='pad')
        self.conv1_1 = nn.Conv2d(filters[0] * 4, filters[0], kernel_size=1, stride=1)

        self.conv6_2_1 = nn.Conv2d(filters[1] * 2, filters[1], padding=1, kernel_size=3, group=filters[1] // 2, dilation=1, pad_mode='pad')
        self.conv6_2_2 = nn.Conv2d(filters[1] * 2, filters[1], padding=2, kernel_size=3, group=filters[1] // 2, dilation=2, pad_mode='pad')
        self.conv6_2_3 = nn.Conv2d(filters[1] * 2, filters[1], padding=3, kernel_size=3, group=filters[1] // 2, dilation=3, pad_mode='pad')
        self.conv6_2_4 = nn.Conv2d(filters[1] * 2, filters[1], padding=4, kernel_size=3, group=filters[1] // 2, dilation=4, pad_mode='pad')
        self.conv2_1 = nn.Conv2d(filters[1] * 4, filters[1], kernel_size=1, stride=1)

        self.conv6_3_1 = nn.Conv2d(filters[2] * 2, filters[2], padding=1, kernel_size=3, group=filters[2] // 2, dilation=1, pad_mode='pad')
        self.conv6_3_2 = nn.Conv2d(filters[2] * 2, filters[2], padding=2, kernel_size=3, group=filters[2] // 2, dilation=2, pad_mode='pad')
        self.conv6_3_3 = nn.Conv2d(filters[2] * 2, filters[2], padding=3, kernel_size=3, group=filters[2] // 2, dilation=3, pad_mode='pad')
        self.conv6_3_4 = nn.Conv2d(filters[2] * 2, filters[2], padding=4, kernel_size=3, group=filters[2] // 2, dilation=4, pad_mode='pad')
        self.conv3_1 = nn.Conv2d(filters[2] * 4, filters[2], kernel_size=1, stride=1)

        self.conv6_4_1 = nn.Conv2d(filters[3]*2, filters[3], padding=1, kernel_size=3, group=filters[3]//2, dilation=1, pad_mode='pad')
        self.conv6_4_2 = nn.Conv2d(filters[3]*2, filters[3], padding=2, kernel_size=3, group=filters[3]//2, dilation=2, pad_mode='pad')
        self.conv6_4_3 = nn.Conv2d(filters[3]*2, filters[3], padding=3, kernel_size=3, group=filters[3]//2, dilation=3, pad_mode='pad')
        self.conv6_4_4 = nn.Conv2d(filters[3]*2, filters[3], padding=4, kernel_size=3, group=filters[3]//2, dilation=4, pad_mode='pad')
        self.conv4_1 = nn.Conv2d(filters[3]*4, filters[3], kernel_size=1, stride=1)

        # SA
        self.cam_attention_1 = Conv_CAM_Layer(filters[0], filters[0], False)  #SA4
        self.cam_attention_2 = Conv_CAM_Layer(filters[1], filters[1], False)  #SA3
        self.cam_attention_3 = Conv_CAM_Layer(filters[2], filters[2], False)  #SA2
        self.cam_attention_4 = Conv_CAM_Layer(filters[3], filters[3], False)  #SA1

        #Row Attention
        self.row_attention_1 = RowAttention(filters[0], filters[0], False)  # SA4
        self.row_attention_2 = RowAttention(filters[1], filters[1], False)  # SA3
        self.row_attention_3 = RowAttention(filters[2], filters[2], False)  # SA2
        self.row_attention_4 = RowAttention(filters[3], filters[3], False)  # SA1

        # Col Attention
        self.col_attention_1 = ColAttention(filters[0], filters[0], False)  # SA4
        self.col_attention_2 = ColAttention(filters[1], filters[1], False)  # SA3
        self.col_attention_3 = ColAttention(filters[2], filters[2], False)  # SA2
        self.col_attention_4 = ColAttention(filters[3], filters[3], False)  # SA1

        self.c4_conv = nn.Conv2d(filters[3], filters[1], kernel_size=3, padding=1, pad_mode='pad')
        self.c3_conv = nn.Conv2d(filters[2], filters[1], kernel_size=3, padding=1, pad_mode='pad')
        self.c2_conv = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1, pad_mode='pad')
        self.c1_conv = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1, pad_mode='pad')

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    def forward(self, inputs):
        feas1, feas2 = inputs
        c4_1 = self.conv4_1(
            F.cat([self.conv6_4_1(F.cat([feas1[0], feas2[0]], 1)), self.conv6_4_2(F.cat([feas1[0], feas2[0]], 1)),
                       self.conv6_4_3(F.cat([feas1[0], feas2[0]], 1)), self.conv6_4_4(F.cat([feas1[0], feas2[0]], 1))], 1))
        c4 = self.cam_attention_4(c4_1) + self.row_attention_4(self.col_attention_4(c4_1))

        c3_1 = (self.conv3_1(F.cat(
            [self.conv6_3_1(F.cat([feas1[1], feas2[1]], 1)), self.conv6_3_2(F.cat([feas1[1], feas2[1]], 1)),
             self.conv6_3_3(F.cat([feas1[1], feas2[1]], 1)), self.conv6_3_4(F.cat([feas1[1], feas2[1]], 1))], 1)))
        c3 = F.cat([(self.cam_attention_3(c3_1)+self.row_attention_3(self.col_attention_3(c3_1))), F.upsample(c4,scale_factor=2.0, mode='bilinear', align_corners=True, recompute_scale_factor=True)], 1)
        
        c2_1 = (self.conv2_1(F.cat(
            [self.conv6_2_1(F.cat([feas1[2], feas2[2]], 1)), self.conv6_2_2(F.cat([feas1[2], feas2[2]], 1)),
             self.conv6_2_3(F.cat([feas1[2], feas2[2]], 1)), self.conv6_2_4(F.cat([feas1[2], feas2[2]], 1))], 1)))
        c2 = F.cat([(self.cam_attention_2(c2_1)+self.row_attention_2(self.col_attention_2(c2_1))), F.upsample(c3,scale_factor=2.0, mode='bilinear', align_corners=True, recompute_scale_factor=True)], 1)
        
        c1_1 = (self.conv1_1(F.cat(
            [self.conv6_1_1(F.cat([feas1[3], feas2[3]], 1)), self.conv6_1_2(F.cat([feas1[3], feas2[3]], 1)),
             self.conv6_1_3(F.cat([feas1[3], feas2[3]], 1)), self.conv6_1_4(F.cat([feas1[3], feas2[3]], 1))], 1)))
        c1 = F.cat([(self.cam_attention_1(c1_1)+self.row_attention_1(self.col_attention_1(c1_1))), F.upsample(c2,scale_factor=2.0, mode='bilinear', align_corners=True, recompute_scale_factor=True)], 1)
        out1 = self.conv6(c1)
        return out1