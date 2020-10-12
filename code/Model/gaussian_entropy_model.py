import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


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

        return grad1 * t


class Distribution_for_entropy(nn.Module):
    def __init__(self):
        super(Distribution_for_entropy, self).__init__()

    def forward(self, x, p_dec):
        
        c = p_dec.size()[1]
        mean = p_dec[:, :c//2, :, :]
        scale = p_dec[:, c//2:, :, :]

    # to make the scale always positive
        scale[scale == 0] = 1e-9
    #scale1 = torch.clamp(scale1,min = 1e-6)
        m1 = torch.distributions.normal.Normal(mean, scale)

        lower = m1.cdf(x - 0.5)
        upper = m1.cdf(x + 0.5)

        #sign = -torch.sign(torch.add(lower, upper))
        #sign = sign.detach()
        #likelihood = torch.abs(f.sigmoid(sign * upper) - f.sigmoid(sign * lower))
        likelihood = torch.abs(upper - lower)

        likelihood = Low_bound.apply(likelihood)
        return likelihood


class Distribution_for_entropy3(nn.Module):
    def __init__(self):
        super(Distribution_for_entropy3, self).__init__()

    def forward(self, x, p_dec):

        mean = p_dec[:, 0, :, :, :]
        scale = p_dec[:, 1, :, :, :]

    # to make the scale always positive
        scale[scale == 0] = 1e-9
    #scale1 = torch.clamp(scale1,min = 1e-6)
        m1 = torch.distributions.normal.Normal(mean, scale)

        lower = m1.cdf(x - 0.5)
        upper = m1.cdf(x + 0.5)

        #sign = -torch.sign(torch.add(lower, upper))
        #sign = sign.detach()
        #likelihood = torch.abs(f.sigmoid(sign * upper) - f.sigmoid(sign * lower))
        likelihood = torch.abs(upper - lower)

        likelihood = Low_bound.apply(likelihood)
        return likelihood


class Distribution_for_entropy2(nn.Module):
    def __init__(self):
        super(Distribution_for_entropy2, self).__init__()

    def forward(self, x, p_dec):
        # you can use use 3 gaussian
        prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = [
            torch.chunk(p_dec, 9, dim=1)[i].squeeze(1) for i in range(9)]
        # keep the weight  summation of prob == 1
        probs = torch.stack([prob0, prob1, prob2], dim=-1)
        probs = f.softmax(probs, dim=-1)
        # process the scale value to non-zero
        scale0[scale0 == 0] = 1e-6
        scale1[scale1 == 0] = 1e-6
        scale2[scale2 == 0] = 1e-6
        # 3 gaussian distribution
        m0 = torch.distributions.normal.Normal(mean0, scale0)
        m1 = torch.distributions.normal.Normal(mean1, scale1)
        m2 = torch.distributions.normal.Normal(mean2, scale2)

        likelihood0 = torch.abs(m0.cdf(x + 0.5)-m0.cdf(x-0.5))
        likelihood1 = torch.abs(m1.cdf(x + 0.5)-m1.cdf(x-0.5))
        likelihood2 = torch.abs(m2.cdf(x + 0.5)-m2.cdf(x-0.5))

        likelihoods = Low_bound.apply(
            probs[:, :, :, :, 0]*likelihood0+probs[:, :, :, :, 1]*likelihood1+probs[:, :, :, :, 2]*likelihood2)

        return likelihoods


class Laplace_for_entropy(nn.Module):
    def __init__(self):
        super(Laplace_for_entropy, self).__init__()

    def forward(self, x, p_dec):
        mean = p_dec[:, 0, :, :, :]
        scale = p_dec[:, 1, :, :, :]

    # to make the scale always positive
        scale[scale == 0] = 1e-9
    #scale1 = torch.clamp(scale1,min = 1e-6)
        m1 = torch.distributions.laplace.Laplace(mean, scale)

        lower = m1.cdf(x - 0.5)
        upper = m1.cdf(x + 0.5)
        likelihood = torch.abs(upper - lower)

        likelihood = Low_bound.apply(likelihood)
        return likelihood
