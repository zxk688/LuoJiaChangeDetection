import torch.nn as nn
import torch


class SwitchNorm1d(nn.Module):
    def __init__(
        self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True
    ):
        super(SwitchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))
        self.register_buffer("running_mean", torch.zeros(1, num_features))
        self.register_buffer("running_var", torch.zeros(1, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError("expected 2D input (got {}D input)".format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        mean_ln = x.mean(1, keepdim=True)
        var_ln = x.var(1, keepdim=True)

        if self.training:
            mean_bn = x.mean(0, keepdim=True)
            var_bn = x.var(0, keepdim=True)
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        mean = mean_weight[0] * mean_ln + mean_weight[1] * mean_bn
        var = var_weight[0] * var_ln + var_weight[1] * var_bn

        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias


class SwitchNorm2d(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.9,
        using_moving_average=True,
        using_bn=True,
        last_gamma=False,
    ):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer("running_mean", torch.zeros(1, num_features, 1))
            self.register_buffer("running_var", torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

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


class SwitchNorm3d(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.997,
        using_moving_average=True,
        using_bn=True,
        last_gamma=False,
    ):
        super(SwitchNorm3d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer("running_mean", torch.zeros(1, num_features, 1))
            self.register_buffer("running_var", torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, D, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

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
        x = x.view(N, C, D, H, W)
        return x * self.weight + self.bias



class mixer(nn.Module):
    def __init__(self, dim):
        super(mixer, self).__init__()

        self.depthconv = nn.Conv2d(dim, dim, kernel_size=9, padding=4, groups=dim)
        self.gn1 = SwitchNorm2d(dim)

        self.pointconv = nn.Conv2d(dim, dim, kernel_size=1)
        self.gn2 = SwitchNorm2d(dim)

        self.gelu = nn.GELU()

    def forward(self, x):
        shortcut = x

        x = self.depthconv(x)
        x = self.gn1(x)
        x = self.gelu(x)

        x = x + shortcut
        x = self.pointconv(x)
        x = self.gn2(x)
        x = self.gelu(x)

        return x


class up_sampling(nn.Module):
    def __init__(self, in_ch, out_ch, stride=8):
        super(up_sampling, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            SwitchNorm2d(out_ch),
            nn.GELU(),
        )

        dim = out_ch
        self.patchup = nn.ConvTranspose2d(dim, dim, kernel_size=stride, stride=stride)
        self.bn2 = SwitchNorm2d(dim)

        self.gelu = nn.GELU()

    def forward(self, x):

        x = self.layer1(x)

        x = self.patchup(x)
        x = self.bn2(x)
        output = self.gelu(x)

        return output

    
class RDPNetDecoder(nn.Module):
    def __init__(self, out_ch=1, hid_ch=384, depth=32):
        super(RDPNetDecoder, self).__init__()
        
        
        self.mixer1 = mixer(hid_ch)
        self.ch1 = up_sampling(hid_ch, depth)
        self.mixer2 = mixer(hid_ch)
        self.ch2 = up_sampling(hid_ch, depth)
        self.mixer3 = mixer(hid_ch)
        self.ch3 = up_sampling(hid_ch, depth)
        self.mixer4 = mixer(hid_ch)
        self.ch4 = up_sampling(hid_ch, depth)
        self.mixer5 = mixer(hid_ch)
        self.ch5 = up_sampling(hid_ch, depth)
        self.mixer6 = mixer(hid_ch)
        self.ch6 = up_sampling(hid_ch, depth)

        self.weight = nn.Parameter(torch.randn(1, depth * 6, 1, 1))

        self.final = nn.Conv2d(depth * 6, out_ch, kernel_size=1)

        self.gelu = nn.GELU()
        
        self.sigmoid = nn.Sigmoid()
    def forward(self, feas):
        ch1 = self.ch1(feas[0])
        ch2 = self.ch2(feas[1])
        ch3 = self.ch3(feas[2])
        ch4 = self.ch4(feas[3])
        ch5 = self.ch5(feas[4])
        ch6 = self.ch6(feas[5])
            
        out = torch.cat([ch1, ch2, ch3, ch4, ch5, ch6], 1)
        out = out * self.weight
        out = self.final(out)

        return self.sigmoid(out)
