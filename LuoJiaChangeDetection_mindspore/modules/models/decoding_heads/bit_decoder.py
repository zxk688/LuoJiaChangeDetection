
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as F
from mindspore import Tensor
from einops import rearrange
from ..backbones.resnet import ResNetEncoder
from mindspore import dtype as mstype



###############################################################################
# Helper Functions
###############################################################################
class TwoLayerConv2d(nn.SequentialCell):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, has_bias=False,pad_mode='pad'),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1,pad_mode='pad')
                         )


class Residual(nn.Cell):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def construct(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Cell):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def construct(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Cell):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm((dim,))
        self.fn = fn
    def construct(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Cell):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm((dim,))
        self.fn = fn
    def construct(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class FeedForward(nn.Cell):
    def __init__(self, dim, hidden_dim, dropout = 0.0):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Dense(hidden_dim, dim),
            nn.Dropout(p=dropout)
        )
    def construct(self, x):
        return self.net(x)


class Cross_Attention(nn.Cell):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.0, softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Dense(dim, inner_dim, has_bias=False)
        self.to_k = nn.Dense(dim, inner_dim, has_bias=False)
        self.to_v = nn.Dense(dim, inner_dim, has_bias=False)

        self.to_out = nn.SequentialCell(
            nn.Dense(inner_dim, dim),
            nn.Dropout(p=dropout)
        )

        # 在 Cross_Attention 类中
    def construct(self, x, m, mask = None):
        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)
        transpose = F.Transpose()
        
        q, k, v = map(lambda t: transpose((F.reshape(t,(t.shape[0],t.shape[1],h,-1))),(0, 2, 1, 3)), [q,k,v])

        dots = F.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            # MindSpore 的 masking 实现
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            attention_mask = mask[:, None, :] * mask[:, :, None]
            # 创建一个布尔掩码，在需要屏蔽的位置为 True
            bool_attention_mask = (attention_mask == 0)
            # 获取一个与 dots 类型匹配的极大负数
            mask_value = F.cast(F.scalar_to_tensor(-1e9), dots.dtype)
            # 使用 select 进行屏蔽，在 True 的位置填充 mask_value
            dots = F.select(bool_attention_mask, F.full_like(dots, mask_value), dots)
            del mask

        if self.softmax:
            attn = F.softmax(dots,axis=-1)
        else:
            attn = dots

        out = F.einsum('bhij,bhjd->bhid', attn, v)
        
        transpose = F.Transpose()
        out = transpose(out, (0, 2, 1, 3))
        out = F.reshape(out,(out.shape[0],out.shape[1],-1))
        out = self.to_out(out)

        return out


class Attention(nn.Cell):
    def __init__(self, dim, heads = 8, dim_head = 64, p=0.0):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Dense(dim, inner_dim * 3, has_bias = False)
        self.to_out = nn.SequentialCell(
            nn.Dense(inner_dim, dim),
            nn.Dropout(p=p)
        )

        # 在 Attention 类中
    def construct(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, axis = -1)
        q, k, v = map(lambda t: F.Transpose()(F.reshape(t, (b, n, h, -1)), (0, 2, 1, 3)), qkv)


        dots = F.einsum('bhid,bhjd->bhij',q, k)* self.scale

        if mask is not None:
            # MindSpore 的 masking 实现
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            attention_mask = mask[:, None, :] * mask[:, :, None]
            # 创建一个布尔掩码，在需要屏蔽的位置为 True
            bool_attention_mask = (attention_mask == 0)
            # 获取一个与 dots 类型匹配的极大负数
            mask_value = F.cast(F.scalar_to_tensor(-1e9), dots.dtype)
            # 使用 select 进行屏蔽，在 True 的位置填充 mask_value
            dots = F.select(bool_attention_mask, F.full_like(dots, mask_value), dots)
            del mask

        attn = F.softmax(dots,axis=-1)

        out = F.einsum('bhij,bhjd->bhid', attn, v)
        
        transpose = F.Transpose()
        out = transpose(out, (0, 2, 1, 3))
        out = F.reshape(out,(out.shape[0],out.shape[1],-1))
        out = self.to_out(out)
        return out


class Transformer(nn.Cell):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.CellList([])
        for _ in range(depth):
            self.layers.append(nn.CellList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, p = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def construct(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x


class TransformerDecoder(nn.Cell):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.CellList([])
        for _ in range(depth):
            self.layers.append(nn.CellList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def construct(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)
        return x


###############################################################################
# main Functions
###############################################################################


class BITTrans(ResNetEncoder):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, input_nc=3, output_nc=1, with_pos='learned', resnet_stages_num=4,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(BITTrans, self).__init__()
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, has_bias=False)
        self.tokenizer = tokenizer
        self.if_upsample_2x = True
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2*dim
        self.sigmoid = nn.Sigmoid()
        self.with_pos = with_pos
        if with_pos == 'learned':
            self.pos_embedding = Tensor(F.randn(1, self.token_len*2, 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =Tensor(F.randn(1, 32,
                                                        decoder_pos_size,
                                                        decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0.0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0.0,
                                                      softmax=decoder_softmax)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        # spatial_attention = spatial_attention.view([b, self.token_len, -1])
        spatial_attention = spatial_attention.reshape(b, self.token_len,-1)

        spatial_attention = F.softmax(spatial_attention, axis=-1)
        x = x.reshape(b, c, -1)
        equation = "bln,bcn->blc"
        tokens = F.einsum(equation, spatial_attention, x)

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        transpose = F.Transpose()
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder

        # x = rearrange(x, 'b c h w -> b (h w) c')
        
        x = F.reshape(x,(x.shape[0],x.shape[1],-1))
        x = transpose(x, (0, 2, 1))

        x = self.transformer_decoder(x, m)

        # x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        x = transpose(x, (0, 2, 1))
        x = F.reshape(x,(x.shape[0],x.shape[1],h,w))

        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def construct(self, inputs):
        # forward backbone resnet
        # x1 = self.forward_single(x1)
        # x2 = self.forward_single(x2)
        x1, x2 = inputs
        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = F.cat([token1, token2], axis=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, axis=1)
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        # feature differencing
        x = F.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)

        
        return x
