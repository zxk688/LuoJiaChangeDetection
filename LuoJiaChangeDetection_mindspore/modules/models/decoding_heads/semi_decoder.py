import mindspore 
import mindspore.nn as nn 
import mindspore.ops as F 
import random
import numpy as np
import cv2
from mindspore import ParameterTuple
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.nn.probability.distribution.uniform import Uniform

def icnr(x, scale=2):
    """
    Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/abs/1707.02937
    """

    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    t = F.zeros([ni2,nf,h,w])
    k = initializer(init='HeNormal',shape=t)
    k = F.transpose(k,(0, 1))
    k = F.reshape(k,(ni2, nf, -1))
    k = k.repeat(1, 1, scale**2)
    k = F.reshape(k,(nf,ni,h,w)).transpose(0, 1)
    x.data.copy_(k)


class PixelShuffle(nn.Cell):
    """
    Real-Time Single Image and Video Super-Resolution
    https://arxiv.org/abs/1609.05158
    """
    def __init__(self, n_channels, scale):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels*(scale**2), kernel_size=1)
        icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        self.relu = nn.ReLU()

    def construct(self,x):
        x = self.shuf(self.relu(self.conv(x)))
        return x
def upsample(in_channels, out_channels, upscale, kernel_size=3):
    layers = []

    mid_channels = 32

    diff_conv1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1, has_bias=False, pad_mode='pad')
    initializer(init='HeNormal',shape=diff_conv1x1.weight.shape)
    layers.append(diff_conv1x1)

    #ReLU
    diff_relu = nn.ReLU()
    layers.append(diff_relu)

    #Upsampling to original size
    up = nn.Upsample(scale_factor=upscale, mode='bilinear',recompute_scale_factor=True)
    layers.append(up)

    #Classification layer
    conv1x1 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, has_bias=False)
    # nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
    initializer('HeNormal',conv1x1.weight.shape)
    layers.append(conv1x1)

    return nn.SequentialCell(*layers)


class MainDecoder(nn.Cell):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(MainDecoder, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def construct(self, x):
        x = self.upsample(x)
        return x


class DropOutDecoder(nn.Cell):
    def __init__(self, upscale, conv_in_ch, num_classes, drop_rate=0.3, spatial_dropout=True):
        super(DropOutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def construct(self, x, _, pertub=True):
        if pertub:
            x = self.upsample(self.dropout(x))
        else:
            x = self.upsample(x)
        return x


class FeatureDropDecoder(nn.Cell):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(FeatureDropDecoder, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def feature_dropout(self, x):
        attention = F.ReduceMean(keep_dims=True)(x,1)
        max_val = F.ReduceMax(keep_dims=True)(attention.view(x.shape[0], -1),1)
        
        threshold = max_val * Tensor(np.random.uniform(0.7, 0.9), dtype=mindspore.float32)
        threshold = threshold.view(x.shape[0], 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        drop_mask = drop_mask.astype(mindspore.float32)

        return x * drop_mask

    def construct(self, x, _, pertub=True):
        if pertub:
            x = self.feature_dropout(x)
            x = self.upsample(x)
        else:
            x = self.upsample(x)
        return x


class FeatureNoiseDecoder(nn.Cell):
    def __init__(self, upscale, conv_in_ch, num_classes, uniform_range=0.3):
        super(FeatureNoiseDecoder, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist._sample(x.shape[1:]).unsqueeze(0)
        x_noise = F.mul(x,noise_vector) + x
        return x_noise

    def construct(self, x, pertub=True):
        pertub=True
        if pertub:
            x = self.feature_based_noise(x)
            x = self.upsample(x)
        else:
            x = self.upsample(x)
        return x

class GradWrap(nn.Cell):
    """求函数输入梯度"""
    def __init__(self, network):
        super(GradWrap, self).__init__(auto_prefix=False)
        self.network = network
        # 用Tuple的形式包装weight
        self.weights = ParameterTuple(filter(lambda x: x.requires_grad, network.get_parameters()))

    def construct(self, x, label):
        weights = self.weights
        # 返回值为梯度
        return F.GradOperation(get_by_list=True)(self.network, weights)(x, label)

def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = F.reshape(d,(d.shape[0], -1, *(1 for _ in range(d.dim() - 2))))
    d /= F.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

def get_r_adv(x, decoder, it=1, xi=1e-1, eps=10.0):
    x_detached = x
    pred = F.Softmax(axis=1)(decoder(x_detached))

    d = F.rand(x.shape).sub(0.5)
    d = _l2_normalize(d)

    # 调用GradWrap
    
    criterion = F.KLDivLoss(reduction='mean')
    net_with_criterion = nn.WithLossCell(decoder, criterion)
    train_network = GradWrap(net_with_criterion)

    optimizer = nn.Momentum(filter(lambda x: x.requires_grad, decoder.get_parameters()), 0.1, 0.9)
    for _ in range(it):
        d.requires_grad = True
        pred_hat = decoder(x_detached + xi * d)
        logp_hat = F.LogSoftmax(axis=1)(pred_hat)
        loss_output = criterion(logp_hat, pred)
        grads = train_network(x_detached + xi * d, pred)
        #adv_distance.backward()
        optimizer(grads)
    r_adv = d * eps
    return r_adv

class VATDecoder(nn.Cell):
    def __init__(self, upscale, conv_in_ch, num_classes, xi=1e-1, eps=10.0, iterations=1):
        super(VATDecoder, self).__init__()
        self.xi = xi
        self.eps = eps
        self.it = iterations
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def construct(self, x, _, pertub=True):
        if pertub:
            r_adv = get_r_adv(x, self.upsample, self.it, self.xi, self.eps)
            x = self.upsample(x + r_adv)
        else:
            x = self.upsample(x)
        return x



def guided_cutout(output, upscale, resize, erase=0.4, use_dropout=False):
    if len(output.shape) == 3:
        masks = (output > 0).float()
    else:
        masks = (output.argmax(1) > 0).float()

    if use_dropout:
        p_drop = random.randint(3, 6)/10
        maskdroped = (F.dropout(masks, p_drop) > 0).float()
        maskdroped = maskdroped + (1 - masks)
        maskdroped.unsqueeze_(0)
        maskdroped = F.interpolate(maskdroped, size=resize, mode='nearest')

    masks_np = []
    for mask in masks:
        mask_np = np.uint8(mask.numpy())
        mask_ones = np.ones_like(mask_np)
        try: # Version 3.x
            _, contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except: # Version 4.x
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polys = [c.reshape(c.shape[0], c.shape[-1]) for c in contours if c.shape[0] > 50]
        for poly in polys:
            min_w, max_w = poly[:, 0].min(), poly[:, 0].max()
            min_h, max_h = poly[:, 1].min(), poly[:, 1].max()
            bb_w, bb_h = max_w-min_w, max_h-min_h
            rnd_start_w = random.randint(0, int(bb_w*(1-erase)))
            rnd_start_h = random.randint(0, int(bb_h*(1-erase)))
            h_start, h_end = min_h+rnd_start_h, min_h+rnd_start_h+int(bb_h*erase)
            w_start, w_end = min_w+rnd_start_w, min_w+rnd_start_w+int(bb_w*erase)
            mask_ones[h_start:h_end, w_start:w_end] = 0
        masks_np.append(mask_ones)
    masks_np = np.stack(masks_np)

    # maskcut = Tensor.from_numpy(masks_np).float().unsqueeze_(1)
    maskcut = Tensor(masks_np, dtype=mindspore.float32).unsqueeze(1)
    maskcut = F.interpolate(maskcut, size=resize, mode='nearest')

    if use_dropout:
        return maskcut, maskdroped
    return maskcut

class CutOutDecoder(nn.Cell):
    def __init__(self, upscale, conv_in_ch, num_classes, drop_rate=0.3, spatial_dropout=True, erase=0.4):
        super(CutOutDecoder, self).__init__()
        self.erase = erase
        self.upscale = upscale 
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def construct(self, x, pred=None, pertub=True):
        if pertub:
            maskcut = guided_cutout(pred, upscale=self.upscale, erase=self.erase, resize=(x.shape[2], x.shape[3]))
            x = x * maskcut
            x = self.upsample(x)
        else:
            x = self.upsample(x)
        return x


def guided_masking(x, output, upscale, resize, return_msk_context=True):
    if len(output.shape) == 3:
        masks_context = (output > 0).float().unsqueeze(1)
    else:
        masks_context = (output.argmax(1) > 0).float().unsqueeze(1)
    
    masks_context = F.interpolate(masks_context, size=resize, mode='nearest')

    x_masked_context = masks_context * x
    if return_msk_context:
        return x_masked_context

    masks_objects = (1 - masks_context)
    x_masked_objects = masks_objects * x
    return x_masked_objects


class ContextMaskingDecoder(nn.Cell):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(ContextMaskingDecoder, self).__init__()
        self.upscale = upscale
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def construct(self, x, pred=None, pertub=True):
        if pertub:
            x_masked_context = guided_masking(x, pred, resize=(x.shape[2], x.shape[3]),
                                          upscale=self.upscale, return_msk_context=True)
            x_masked_context = self.upsample(x_masked_context)
        else:
            x_masked_context = self.upsample(x)
        return x_masked_context


class ObjectMaskingDecoder(nn.Cell):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(ObjectMaskingDecoder, self).__init__()
        self.upscale = upscale
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def construct(self, x, pred=None, pertub=True):
        if pertub:
            x_masked_obj = guided_masking(x, pred, resize=(x.shape[2], x.shape[3]),
                                      upscale=self.upscale, return_msk_context=False)
            x_masked_obj = self.upsample(x_masked_obj)
        else:
            x_masked_obj = self.upsample(x)
        return x_masked_obj

