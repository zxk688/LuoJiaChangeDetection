import math
from functools import partial
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common import initializer as weight_init

class ResidualBlock(nn.Cell):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def construct(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = ops.add(out, residual)
        return out
    
class ConvLayer(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
#         reflection_padding = kernel_size // 2
#         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1,pad_mode='pad')

    def construct(self, x):
#         out = self.reflection_pad(x)
        out = self.conv2d(x)
        return out
    
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):

    return ops.interpolate(input, size, scale_factor, mode, align_corners)


class DecoderTransformer_v3(nn.Cell):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True, 
                    in_channels = [32, 64, 128, 256], embedding_dim= 64, output_nc=2, 
                    decoder_softmax = False, feature_strides=[2, 4, 8, 16]):
        super(DecoderTransformer_v3, self).__init__()
        #assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        
        #settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index        = in_index
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim
        self.output_nc       = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        #MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        #convolutional Difference Modules
        self.diff_c4   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)

        #taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        #Final linear fusion layer
        self.linear_fuse = nn.SequentialCell(
            nn.Conv2d(in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim,
                                        kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.SequentialCell( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.SequentialCell( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid() 

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = ops.cat(upsampled_inputs, axis=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def construct(self, inputs1, inputs2):
        #Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        #img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4   = self.diff_c4(ops.cat((_c4_1, _c4_2), axis=1))
        p_c4  = self.make_pred_c4(_c4)
        outputs.append(p_c4)
        _c4_up= resize(_c4, size=c1_2.shape[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3   = self.diff_c3(ops.cat((_c3_1, _c3_2), axis=1)) + ops.interpolate(_c4, scale_factor=2.0, mode="bilinear",recompute_scale_factor=True)
        p_c3  = self.make_pred_c3(_c3)
        outputs.append(p_c3)
        _c3_up= resize(_c3, size=c1_2.shape[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2   = self.diff_c2(ops.cat((_c2_1, _c2_2), axis=1)) + ops.interpolate(_c3, scale_factor=2.0, mode="bilinear",recompute_scale_factor=True)
        p_c2  = self.make_pred_c2(_c2)
        outputs.append(p_c2)
        _c2_up= resize(_c2, size=c1_2.shape[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1   = self.diff_c1(ops.cat((_c1_1, _c1_2), axis=1)) + ops.interpolate(_c2, scale_factor=2.0, mode="bilinear",recompute_scale_factor=True)
        p_c1  = self.make_pred_c1(_c1)
        outputs.append(p_c1)

        #Linear Fusion of difference image from all scales
        _c = self.linear_fuse(ops.cat((_c4_up, _c3_up, _c2_up, _c1), axis=1))

        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None

        #Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        #Residual block
        x = self.dense_2x(x)
        #Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        #Residual block
        x = self.dense_1x(x)

        #Final prediction
        cp = self.change_probability(x)
        
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs
#Difference module
def conv_diff(in_channels, out_channels):
    return nn.SequentialCell(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,pad_mode='pad'),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,pad_mode='pad'),
        nn.ReLU()
    )

#Intermediate prediction module
def make_prediction(in_channels, out_channels):
    return nn.SequentialCell(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,pad_mode='pad'),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,pad_mode='pad')
    )

class MLP(nn.Cell):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Dense(input_dim, embed_dim)

    def construct(self, x):

        x = ops.reshape(x,(x.shape[0],x.shape[1],-1))
        input_perm = (0, 2, 1)
        transpose = ops.Transpose()
        output = transpose(x, input_perm)
        output = self.proj(output)
        return output
    

class UpsampleConvLayer(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.Conv2dTranspose(in_channels, out_channels, kernel_size, stride=stride, padding=1,pad_mode='pad')

    def construct(self, x):
        out = self.conv2d(x)
        return out

class TransDecoder(nn.Cell):
    def __init__(self,output_nc,embed_dims,depths,embedding_dim,decoder_softmax=False):
        super().__init__()
        self.embed_dims = embed_dims
        self.depths = depths
        self.embedding_dim = embedding_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1 
        #Transformer Decoder
        self.TDec_x2  = DecoderTransformer_v3(input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=False, 
                    in_channels = self.embed_dims, embedding_dim= self.embedding_dim, output_nc=output_nc, 
                    decoder_softmax = decoder_softmax, feature_strides=[2, 4, 8, 16])
        self.sigmoid = nn.Sigmoid()
    def construct(self, inputs):
        fx1, fx2 = inputs
        cp = self.TDec_x2(fx1, fx2)

        return self.sigmoid(cp[4])

