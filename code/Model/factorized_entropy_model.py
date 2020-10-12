import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.uniform import Uniform
from torch.nn.parameter import Parameter


class Low_bound(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-6)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        grad1[x < 1e-6] = 0
        pass_through_if = np.logical_or(
            x.cpu().numpy() >= 1e-6, g.cpu().numpy() < 0.0)
        t = torch.Tensor(pass_through_if+0.0).cuda()
        return grad1*t


class Entropy_bottleneck(nn.Module):
    def __init__(self, channel, init_scale=10, filters=(3, 3, 3), likelihood_bound=1e-6,
                 tail_mass=1e-9, optimize_integer_offset=True):
        super(Entropy_bottleneck, self).__init__()

        self.filters = tuple(int(t) for t in filters)
        self.init_scale = float(init_scale)
        self.likelihood_bound = float(likelihood_bound)
        self.tail_mass = float(tail_mass)

        self.optimize_integer_offset = bool(optimize_integer_offset)

        if not 0 < self.tail_mass < 1:
            raise ValueError(
                "`tail_mass` must be between 0 and 1")
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1.0 / (len(self.filters) + 1))
        self._matrices = nn.ParameterList([])
        self._bias = nn.ParameterList([])
        self._factor = nn.ParameterList([])
        # print ('scale:',scale)
        for i in range(len(self.filters) + 1):

            init = np.log(np.expm1(1.0 / scale / filters[i + 1]))

            self.matrix = Parameter(torch.FloatTensor(
                channel, filters[i + 1], filters[i]))

            self.matrix.data.fill_(init)

            self._matrices.append(self.matrix)

            self.bias = Parameter(
                torch.FloatTensor(channel, filters[i + 1], 1))

            noise = np.random.uniform(-0.5, 0.5, self.bias.size())
            noise = torch.FloatTensor(noise)
            self.bias.data.copy_(noise)
            self._bias.append(self.bias)

            if i < len(self.filters):
                self.factor = Parameter(
                    torch.FloatTensor(channel, filters[i + 1], 1))

                self.factor.data.fill_(0.0)

                self._factor.append(self.factor)

    def _logits_cumulative(self, logits, stop_gradient):

        for i in range(len(self.filters) + 1):

            matrix = f.softplus(self._matrices[i])
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(matrix, logits)

            bias = self._bias[i]
            if stop_gradient:
                bias = bias.detach()
            logits += bias

            if i < len(self._factor):
                factor = f.tanh(self._factor[i])
                if stop_gradient:
                    factor = factor.detach()
                logits += factor * f.tanh(logits)
        return logits

    def add_noise(self, x):
        noise = np.random.uniform(-0.5, 0.5, x.size())
        noise = torch.Tensor(noise).cuda()
        return x + noise

    def forward(self, x, training):
        x = x.permute(1, 0, 2, 3).contiguous()
        shape = x.size()
        x = x.view(shape[0], 1, -1)
        if training == 0:
            x = self.add_noise(x)
        elif training == 1:
            x = UniverseQuant.apply(x)
        else:
            x = torch.round(x)
        lower = self._logits_cumulative(x - 0.5, stop_gradient=False)
        upper = self._logits_cumulative(x + 0.5, stop_gradient=False)

        sign = -torch.sign(torch.add(lower, upper))
        sign = sign.detach()
        likelihood = torch.abs(
            f.sigmoid(sign * upper) - f.sigmoid(sign * lower))

        if self.likelihood_bound > 0:
            likelihood = Low_bound.apply(likelihood)

        likelihood = likelihood.view(shape)
        likelihood = likelihood.permute(1, 0, 2, 3)
        x = x.view(shape)
        x = x.permute(1, 0, 2, 3)
        return x, likelihood


class UniverseQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        #b = np.random.uniform(-1,1)
        b = 0
        uniform_distribution = Uniform(-0.5*torch.ones(x.size())
                                       * (2**b), 0.5*torch.ones(x.size())*(2**b)).sample().cuda()
        return torch.round(x+uniform_distribution)-uniform_distribution

    @staticmethod
    def backward(ctx, g):

        return g
