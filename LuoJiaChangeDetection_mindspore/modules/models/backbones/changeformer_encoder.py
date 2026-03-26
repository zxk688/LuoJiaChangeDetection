import math
from functools import partial

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common import initializer as weight_init
from .misc.drop_path import DropPath 
from .misc.identity import Identity
from .misc.compatibility import Dropout

#Transormer Ecoder with x2, x4, x8, x16 scales
class EncoderTransformer_v3(nn.Cell):
    def __init__(self, img_size=256, patch_size=3, in_chans=3, num_classes=2, embed_dims=[32, 64, 128, 256],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 3, 6, 18], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = 4
        linear=False
        start = Tensor(0, mindspore.float32)
        stop = Tensor(drop_path_rate, mindspore.float32)
        dpr = [float(x) for x in ops.linspace(start, stop, sum(depths))]  # stochastic depth decay rule
        cur = 0

        patch_embed_list = []
        block_list = []
        norm_list = []

        for i in range(self.num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.CellList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear, block_id=j)
                for j in range(depths[i])])

            norm = norm_layer([embed_dims[i]])

            cur += depths[i]

            patch_embed_list.append(patch_embed)
            block_list.append(block)
            norm_list.append(norm)
        self.patch_embed_list = nn.CellList(patch_embed_list)
        self.block_list = nn.CellList(block_list)
        self.norm_list = nn.CellList(norm_list)
        # classification head
        self.head = nn.Dense(embed_dims[3], num_classes) if num_classes > 0 else Identity()
        self._initialize_weights()

    def freeze_patch_emb(self):
        self.patch_embed_list[0].requires_grad = False

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape, cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(), cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                fan_out = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                fan_out //= cell.group
                cell.weight.set_data(weight_init.initializer(weight_init.Normal(sigma=math.sqrt(2.0 / fan_out)),
                                                             cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(), cell.bias.shape, cell.bias.dtype))

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Dense(self.embed_dim, num_classes) if num_classes > 0 else Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = self.patch_embed_list[i]
            block = self.block_list[i]
            norm = self.norm_list[i]
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            outs.append(x)

        return outs

    # def forward_head(self, x: Tensor) -> Tensor:
    #     return self.head(x)

    def construct(self, x):
        x = self.forward_features(x)
        # x = self.forward_head(x)

        return x

class OverlapPatchEmbed(nn.Cell):
    """Overlapping Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, has_bias=True)
        self.norm = nn.LayerNorm([embed_dim])

    def construct(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = ops.transpose(ops.reshape(x, (B, C, H * W)), (0, 2, 1))
        x = self.norm(x)

        return x, H, W

class Block(nn.Cell):
    """Block with Linear Spatial Reduction Attention and Convolutional Feed-Forward"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False, block_id=0):
        super().__init__()
        self.norm1 = norm_layer([dim])

        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()

        self.norm2 = norm_layer([dim])

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

    def construct(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

class Attention(nn.Cell):
    """Linear Spatial Reduction Attention"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.kv = nn.Dense(dim, dim * 2, has_bias=qkv_bias)
        self.attn_drop = Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = Dropout(p=proj_drop)
        self.qk_batmatmul = ops.BatchMatMul(transpose_b=True)
        self.batmatmul = ops.BatchMatMul()
        self.softmax = nn.Softmax(axis=-1)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, has_bias=True)
                self.norm = nn.LayerNorm([dim])

        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, has_bias=True)
            self.norm = nn.LayerNorm([dim])
            self.act = nn.GELU()

    def construct(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)
        q = ops.reshape(q, (B, N, self.num_heads, C // self.num_heads))
        q = ops.transpose(q, (0, 2, 1, 3))

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = ops.reshape(ops.transpose(x, (0, 2, 1)), (B, C, H, W))

                x_ = self.sr(x_)
                x_ = ops.transpose(ops.reshape(x_, (B, C, -1)), (0, 2, 1))
                x_ = self.norm(x_)

                kv = self.kv(x_)
                kv = ops.transpose(ops.reshape(kv, (B, -1, 2, self.num_heads, C // self.num_heads)), (2, 0, 3, 1, 4))
            else:
                kv = self.kv(x)
                kv = ops.transpose(ops.reshape(kv, (B, -1, 2, self.num_heads, C // self.num_heads)), (2, 0, 3, 1, 4))

        else:
            x_ = ops.reshape(ops.transpose(x, (0, 2, 1)), (B, C, H, W))
            x_ = self.sr(self.pool(x_))
            x_ = ops.reshape(ops.transpose(x_, (0, 2, 1)), (B, C, -1))
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = ops.transpose(ops.reshape(self.kv(x_), (B, -1, 2, self.num_heads, C // self.num_heads)),
                               (2, 0, 3, 1, 4))
        k, v = kv[0], kv[1]

        attn = self.qk_batmatmul(q, k) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = self.batmatmul(attn, v)
        x = ops.reshape(ops.transpose(x, (0, 2, 1, 3)), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Mlp(nn.Cell):
    """MLP with depthwise separable convolution"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = Dropout(p=drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU()

    def construct(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Cell):
    """Depthwise separable convolution"""

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, has_bias=True, group=dim)

    def construct(self, x, H, W):
        B, N, C = x.shape
        x = ops.transpose(x, (0, 2, 1)).view((B, C, H, W))
        x = self.dwconv(x)
        x = ops.transpose(x.view((B, C, H * W)), (0, 2, 1))

        return x
class TransEncoder(nn.Cell):
    def __init__(self,input_nc, output_nc,embed_dims,depths,embedding_dim,decoder_softmax=False):
        super().__init__()
        self.embed_dims = embed_dims
        self.depths = depths
        self.embedding_dim = embedding_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1 
        self.Tenc_x2 = EncoderTransformer_v3(img_size=256, patch_size = 7, in_chans=input_nc, num_classes=output_nc, embed_dims=self.embed_dims,
                 num_heads = [1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=self.drop_rate,
                 attn_drop_rate = self.attn_drop, drop_path_rate=self.drop_path_rate, norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
                 depths=self.depths, sr_ratios=[8, 4, 2, 1])
        
    def construct(self, x):
        x1 = x[:,0]  # 10x3x256x256
        x2 = x[:,1] 
        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]
        #fx1_list(4, 64, 64, 64) (4, 128, 32, 32) (4, 320, 16, 16) (4, 512, 8, 8)
        return fx1, fx2
    