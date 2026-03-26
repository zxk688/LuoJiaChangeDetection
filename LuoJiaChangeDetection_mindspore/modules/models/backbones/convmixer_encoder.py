import mindspore.nn as nn
import mindspore.ops as F
import mindspore.numpy as np
from mindspore import Tensor,Parameter
class SwitchNorm2d(nn.Cell):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.9,
        using_moving_average=True,
        using_bn=False,
        last_gamma=False,
    ):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = Parameter(Tensor(np.ones((1, num_features, 1, 1), dtype=np.float32)), name="weight")
        self.bias = Parameter(Tensor(np.zeros((1, num_features, 1, 1), dtype=np.float32)), name="bias")

        if self.using_bn:
            self.mean_weight = Parameter(Tensor(np.ones(3)))
            self.var_weight = Parameter(Tensor(np.ones(3)))
        else:
            self.mean_weight = Parameter(Tensor(np.ones(2)))
            self.var_weight = Parameter(Tensor(np.ones(2)))

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def construct(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.shape
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keep_dims=True)
        var_in = x.var(-1, keepdims=True)

        mean_ln = mean_in.mean(1, keep_dims=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keep_dims=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keep_dims=True)
                var_bn = temp.mean(0, keep_dims=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = F.autograd.Variable(self.running_mean)
                var_bn = F.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = (
                mean_weight[0] * mean_in
                + mean_weight[1] * mean_ln
                + mean_weight[2] * mean_bn
            )
            var = (
                var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
            )
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

class mixer(nn.Cell):
    def __init__(self, dim):
        super(mixer, self).__init__()

        self.depthconv = nn.Conv2d(dim, dim, kernel_size=9, padding=4, group=dim,pad_mode='pad')
        self.gn1 = SwitchNorm2d(dim)

        self.pointconv = nn.Conv2d(dim, dim, kernel_size=1)
        self.gn2 = SwitchNorm2d(dim)

        self.gelu = nn.GELU()

    def construct(self, x):
        shortcut = x

        x = self.depthconv(x)
        x = self.gn1(x)
        x = self.gelu(x)

        x = x + shortcut
        x = self.pointconv(x)
        x = self.gn2(x)
        x = self.gelu(x)

        return x
    
class ConvMixerEncoder(nn.Cell):
    def __init__(self, in_ch,hid_ch):
        super(ConvMixerEncoder, self).__init__()

        self.patchEmb = nn.Conv2d(in_ch * 2, hid_ch, kernel_size=8, stride=8)
        self.gn1 = SwitchNorm2d(hid_ch)

        self.mixer1 = mixer(hid_ch)
        self.mixer2 = mixer(hid_ch)
        self.mixer3 = mixer(hid_ch)
        self.mixer4 = mixer(hid_ch)
        self.mixer5 = mixer(hid_ch)
        self.mixer6 = mixer(hid_ch)
        self.gelu = nn.GELU()
    def construct(self, x):
        feas = []
        x1 = x[:,0]
        x2 = x[:,0]
        x = F.concat([x1,x2],axis=1)
        x = self.patchEmb(x)
        x = self.gn1(x)
        x = self.gelu(x)

        x = self.mixer1(x)
        feas.append(x)
        x = self.mixer2(x)
        feas.append(x)
        x = self.mixer3(x)
        feas.append(x)
        x = self.mixer4(x)
        feas.append(x)
        x = self.mixer5(x)
        feas.append(x)
        x = self.mixer6(x)
        feas.append(x)
        return feas
