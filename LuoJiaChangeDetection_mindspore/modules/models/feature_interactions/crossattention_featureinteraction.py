import numpy as np
import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as F
from mindspore import Parameter,Tensor

class Conv(nn.Cell):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, has_bias=bias, pad_mode='pad')
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def construct(self, x):
        assert x.shape[1] == self.inp_dim, "{} {}".format(x.shape[1], self.inp_dim)
        # print("++",x.size()[1],self.inp_dim,x.size()[1],self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Residual(nn.Cell):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def construct(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 
    
class ConvRelPosEnc(nn.Cell):
    """ Convolutional relative position encoding. """
    def __init__(self, Ch, h, window):
        """
        Initialization.
            Ch: Channels per head.
            h: Number of heads.
            window: Window size(s) in convolutional relative positional encoding. It can have two forms:
                    1. An integer of window size, which assigns all attention heads with the same window size in ConvRelPosEnc.
                    2. A dict mapping window size to #attention head splits (e.g. {window size 1: #attention head split 1, window size 2: #attention head split 2})
                       It will apply different window size to the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            window = {window: h}                                                         # Set the same window size for all attention heads.
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()            
        
        self.conv_list = nn.CellList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1                                                                 # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2         # Determine padding size. Ref: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
            cur_conv = nn.Conv2d(cur_head_split*Ch, cur_head_split*Ch,
                kernel_size=(cur_window, cur_window), 
                padding=padding_size,
                dilation=(dilation, dilation),                          
                group=cur_head_split*Ch,pad_mode='pad'
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x*Ch for x in self.head_splits]

    def construct(self, q, v, size):
        B, h, N, Ch = q.shape
        H, W = size
        assert N == H * W
        # print(q.shape,v.shape)
        # Convolutional relative position encoding.
        # q_img = q                                                             # Shape: [B, h, H*W, Ch].
        # v_img = v                                                             # Shape: [B, h, H*W, Ch].
        # print(q.shape,v.shape)
        # v_img = rearrange(v, 'B h (H W) Ch -> B (h Ch) H W', H=H, W=W)               # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        transpose = F.Transpose()
        input_perm1 = (0, 2, 1, 3)
        v = transpose(v, input_perm1)
        v = F.reshape(v,(v.shape[0],v.shape[1],-1))
        input_perm2 = (0, 2, 1)
        v = transpose(v, input_perm2)
        v_img = F.reshape(v,(v.shape[0],v.shape[1],H,W))

        v_img_list = F.split(v_img, self.channel_splits, axis=1)                      # Split according to channels.
        conv_v_img_list = [conv(x) for conv, x in zip(self.conv_list, v_img_list)]
        conv_v_img = F.cat(conv_v_img_list, axis=1)
        m=conv_v_img
        m = F.reshape(m,(m.shape[0],m.shape[1],-1))
        input_perm3 = (0, 2, 1)
        m = transpose(m, input_perm3)
        m = F.reshape(m,(m.shape[0],m.shape[1],h,Ch))
        input_perm4 = (0, 2, 1, 3)
        conv_v_img = transpose(m, input_perm4)
        # conv_v_img = rearrange(conv_v_img, 'B (h Ch) H W -> B h (H W) Ch', h=h)          # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].

        EV_hat_img = q* conv_v_img
        # print(EV_hat_img.shape)
        # zero = F.zeros((B, h, 0, Ch), dtype=q.dtype)
        # EV_hat = F.cat((zero, EV_hat_img), axis=2)                                # Shape: [B, h, N, Ch].
        # print(EV_hat.shape)
        return EV_hat_img
    
class MultiHeadAttention(nn.Cell):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def positional_encoding_2d(self, d_model, height, width):
        """
        reference: wzlxjtu/PositionalEncoding2D

        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = F.zeros(d_model, height, width)
        try:
            pe = pe.to(F.device("GPU"))
        except RuntimeError:
            pass
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = F.exp(
            F.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = F.arange(0., width).unsqueeze(1)
        pos_h = F.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = F.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = F.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = F.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = F.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe

    def construct(self, x):
        raise NotImplementedError()

class MultiHeadDense(nn.Cell):
    def __init__(self, d, bias=False):
        super(MultiHeadDense, self).__init__()
        # self.weight = nn.Parameter(torch.Tensor(d, d))
        self.weight = Parameter(Tensor((d,d),dtype=mindspore.float32))

    #     if bias:
    #         raise NotImplementedError()
    #     else:
    #         self.register_parameter('bias', None)
    #     self.reset_parameters()

    # def reset_parameters(self) -> None:
    #     mindspore.common.initializer(self.weight, a=math.sqrt(5))
    #     if self.bias is not None:
    #         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
    #         bound = 1 / math.sqrt(fan_in)
    #         nn.init.uniform_(self.bias, -bound, bound)

    def construct(self, x):
        # x:[b, h*w, d]
        b, wh, d = x.shape
        x = F.bmm(x, self.weight.repeat(b, 1, 1))
        # x = F.linear(x, self.weight, self.bias)
        return x
    
class PositionalEncodingPermute2D(nn.Cell):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)        
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)
        

    def construct(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)
    
class PositionalEncoding2D(nn.Cell):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000
                         **(F.arange(0, channels, 2).float() / channels))

        self.inv_freq = Parameter(inv_freq, name='inv_freq')

    def construct(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = F.arange(x,dtype=mindspore.float32)
        pos_y = F.arange(y,dtype=mindspore.float32)
        sin_inp_x = F.einsum("i,j->ij", pos_x,self.inv_freq)
        sin_inp_y = F.einsum("i,j->ij", pos_y,self.inv_freq)
        
        emb_x = F.cat((sin_inp_x.sin(), sin_inp_x.cos()), axis=-1).unsqueeze(1)
        emb_y = F.cat((sin_inp_y.sin(), sin_inp_y.cos()), axis=-1)
        emb = F.zeros((x, y, self.channels * 2),dtype=mindspore.float32)
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y
        tiled_tensor = F.tile(emb[None, :, :, :orig_ch], (batch_size, 1, 1, 1))
        return tiled_tensor
class FactorAtt_ConvRelPosEnc(nn.Cell):
    """ Factorized attention with convolutional relative position encoding class. """
    def __init__(self, dim, num_heads=8, qkv_bias=False,  proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3)                                       # Note: attn_drop is actually not used.
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = ConvRelPosEnc(Ch=dim // num_heads, h=num_heads, window={3:2, 5:3, 7:3})

    def construct(self, q,k,v, size):
        B, N, C = size[0],size[1],size[2]

        # # Generate Q, K, V.
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # Shape: [3, B, h, N, Ch].
        # q, k, v = qkv[0], qkv[1], qkv[2]                                                 # Shape: [B, h, N, Ch].

        # Factorized attention.
        k_softmax = F.softmax(k,axis=2)                                                   # Softmax on dim N.
        k_softmax_T_dot_v = F.einsum('b h n k, b h n v -> b h k v', k_softmax, v)          # Shape: [B, h, Ch, Ch].
        factor_att        = F.einsum('b h n k, b h k v -> b h n v', q, k_softmax_T_dot_v)  # Shape: [B, h, N, Ch].

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=[size[3],size[4]])                                                # Shape: [B, h, N, Ch].
        x = self.scale * factor_att + crpe
        # Merge and reshape.
        transpose = F.Transpose()
        input_perm5 = (0, 2, 1, 3)
        x = transpose(x, input_perm5)                                   # Shape: [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C].
        x = F.reshape(x,(x.shape[0],x.shape[1],-1))
        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x   

class MultiHeadCrossAttention(MultiHeadAttention):
    def __init__(self, channelY, channelS, ch_out, drop_rate=0.2,qkv_bias=False):
        super(MultiHeadCrossAttention, self).__init__()
        self.Sconv = nn.SequentialCell(
            nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU())
        self.Yconv = nn.SequentialCell(
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU())
            
        self.query = MultiHeadDense(channelS, bias=False)
        self.key = MultiHeadDense(channelS, bias=False)
        self.value = MultiHeadDense(channelS, bias=False)

        self.softmax = nn.Softmax(axis=1)
        self.Spe = PositionalEncodingPermute2D(channelS)
        self.Ype = PositionalEncodingPermute2D(channelY)

        self.qkv = nn.Dense(channelS, channelS * 3)
        self.num_heads = 8
        head_dim = channelS// 8
        self.scale = head_dim ** -0.5
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(channelS,self.num_heads,qkv_bias=qkv_bias,  proj_drop=drop_rate)
        self.residual = Residual(channelS*2, ch_out)
        self.dropout = nn.Dropout2d(p = drop_rate)
        self.drop_rate = drop_rate

    def construct(self, Y, S):
        Sb, Sc, Sh, Sw = S.shape
        Yb, Yc, Yh, Yw = Y.shape
        
        Spe = self.Spe(S)
        S = S + Spe
        S1 = self.Sconv(S)
        S1=S1.reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)

        Ype = self.Ype(Y)
        Y = Y + Ype
        Y1 = self.Yconv(Y).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)

        B, N, C = Y1.shape
        size=[B, N, C,Sh, Sw ]

        qkv_l= self.qkv(Y1)
        qkv_l=qkv_l.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # Shape: [3, B, h, N, Ch].
        q_l, k_l, v_l = qkv_l[0], qkv_l[1], qkv_l[2] 

        qkv_g = self.qkv(S1)
        qkv_g=qkv_g.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # Shape: [3, B, h, N, Ch].
        q_g, k_g, v_g = qkv_g[0], qkv_g[1], qkv_g[2] 

        cur1 = self.factoratt_crpe(q_g, k_l, v_l, size).permute(0, 2, 1).reshape(Yb, Sc, Yh, Yw) 
        cur2 = self.factoratt_crpe(q_l, k_g, v_g, size).permute(0, 2, 1).reshape(Yb, Sc, Yh, Yw)

        fuse = self.residual(F.cat([cur1,cur2], 1))
        if self.drop_rate > 0:
            return self.dropout(fuse),cur1,cur2
        else:
            return fuse,cur1,cur2

class CrossAttentionInteraction(nn.Cell):
    def __init__(self,num_classes=1, drop_rate=0.2):
        super(CrossAttentionInteraction, self).__init__()  

        self.cross2 = MultiHeadCrossAttention(256, 320,ch_out=256, drop_rate=drop_rate/2,qkv_bias=True)
        self.cross3 = MultiHeadCrossAttention(128, 128,ch_out=128, drop_rate=drop_rate/2,qkv_bias=True)
        self.cross4 = MultiHeadCrossAttention(64, 64,ch_out=64, drop_rate=drop_rate/2,qkv_bias=True)

        self.cross2_img2 = MultiHeadCrossAttention(256, 320,ch_out=256, drop_rate=drop_rate/2,qkv_bias=True)
        self.cross3_img2 = MultiHeadCrossAttention(128, 128,ch_out=128, drop_rate=drop_rate/2,qkv_bias=True)
        self.cross4_img2 = MultiHeadCrossAttention(64, 64,ch_out=64, drop_rate=drop_rate/2,qkv_bias=True)
          
    def construct(self, inputs):
        cnnfeas1, cnnfeas2, transfeas1,transfeas2 = inputs
        cross_2, curg_2, curl_2 = self.cross2(cnnfeas1[3], transfeas1[2]) # 128 320 320
        corss2_list = (cross_2, curg_2, curl_2,)

        cross_3, curg_3, curl_3 = self.cross3(cnnfeas1[2], transfeas1[1]) # 64 128 128
        corss3_list = (cross_3, curg_3, curl_3,)

        cross_4, curg_4, curl_4 = self.cross4(cnnfeas1[1], transfeas1[0]) # 32 64 64
        corss4_list = (cross_4, curg_4, curl_4,)

        cross_2_img2,curg_2_img2,curl_2_img2=self.cross2_img2(cnnfeas2[3],transfeas2[2])
        corss2_img2_list = (cross_2_img2,curg_2_img2,curl_2_img2,)

        cross_3_img2,curg_3_img2,curl_3_img2=self.cross3_img2(cnnfeas2[2],transfeas2[1])
        corss3_img2_list = (cross_3_img2,curg_3_img2,curl_3_img2,)

        cross_4_img2,curg_4_img2,curl_4_img2=self.cross4_img2(cnnfeas2[1],transfeas2[0])
        corss4_img2_list = (cross_4_img2,curg_4_img2,curl_4_img2,)
        
        corss_out_list = (corss2_list, corss3_list, corss4_list, corss2_img2_list, corss3_img2_list, corss4_img2_list,)
        return corss_out_list