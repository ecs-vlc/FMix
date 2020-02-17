import math

import torch
import torch.nn as nn
from torch import distributions
from torch.distributions import constraints, register_kl

import torchbearer
from torchbearer import state_key

LATENT = state_key('latent')


class LogitNormal(distributions.Normal):
    arg_constraints = {'loc': constraints.real, 'log_scale': constraints.real}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, log_scale, validate_args=None):
        self.log_scale = log_scale
        scale = distributions.transform_to(distributions.Normal.arg_constraints['scale'])(log_scale)
        super().__init__(loc, scale, validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = self.log_scale
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))


@register_kl(LogitNormal, distributions.Normal)
def kl_logitnormal_normal(p, q):
    log_var_ratio = 2 * (p.log_scale - q.scale.log())
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    return 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)


@register_kl(LogitNormal, LogitNormal)
def kl_logitnormal_logitnormal(p, q):
    log_var_ratio = 2 * (p.log_scale - q.log_scale)
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    return 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class View(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view(x.size(0), *self.args)


class SimpleEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            Flatten()
        )

        self.output_size = 32 * 4 * 4


class SimpleDecoder(nn.Sequential):
    def __init__(self, z_dims):
        super().__init__(
            nn.Linear(z_dims, 32 * 4 * 4),
            View(32, 4, 4),
            nn.ConvTranspose2d(32, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 3, (4, 4), stride=2, padding=1)
        )


class DCGANDecoder(nn.Sequential):
    def __init__(self, z_dims, dim=128):
        super().__init__(
            nn.Linear(z_dims, dim * 4 * 4 * 4),
            nn.ReLU(True),
            nn.BatchNorm1d(dim * 4 * 4 * 4),
            View(dim * 4, 4, 4),
            nn.ConvTranspose2d(dim * 4, dim * 2, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(dim * 2),
            nn.ConvTranspose2d(dim * 2, dim, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, 3, 5, stride=2, padding=2, output_padding=1)
        )


class DCGANEncoder(nn.Sequential):
    def __init__(self, dim=128):
        super().__init__(
            nn.Conv2d(3, dim, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim * 2, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(dim * 2),
            nn.Conv2d(dim * 2, dim * 4, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(dim * 4),
            Flatten()
        )

        self.output_size = dim * 4 * 4 * 4


class BetaVAEDecoder(nn.Sequential):
    def __init__(self, z_dims):
        super().__init__(
            nn.Linear(z_dims, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64 * 2 * 2),
            nn.ReLU(True),
            nn.BatchNorm1d(64 * 2 * 2),
            View(64, 2, 2),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32,  3, 4, stride=2, padding=1)
        )


class BetaVAEDecoder2(nn.Sequential):
    def __init__(self, z_dims):
        super().__init__(
            nn.Linear(z_dims, 256),
            nn.ReLU(True),
            # nn.BatchNorm1d(256),
            nn.Linear(256, 64 * 2 * 2),
            nn.ReLU(True),
            # nn.BatchNorm1d(64 * 2 * 2),
            View(64, 2, 2),
            nn.Upsample(size=5, mode='bilinear'),
            nn.Conv2d(64, 64, 4, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Upsample(size=9, mode='bilinear'),
            nn.Conv2d(64, 32, 4, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Upsample(size=17, mode='bilinear'),
            nn.Conv2d(32, 32, 4, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Upsample(size=33, mode='bilinear'),
            nn.Conv2d(32, 3, 4, padding=1)
        )


class BetaVAEEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d( 3, 32, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            Flatten(),
            nn.Linear(64 * 2 * 2, 256),
            nn.ReLU(True),
            # nn.BatchNorm1d(256)
        )

        self.output_size = 256


class VAE(nn.Module):
    def __init__(self, z_dims=64, encoder=BetaVAEEncoder, decoder=BetaVAEDecoder2, var=0.1):
        super(VAE, self).__init__()

        self.var = var

        self.encoder = encoder()
        self.decoder = decoder(z_dims)

        self.loc = nn.Linear(self.encoder.output_size, z_dims)
        self.scale = nn.Linear(self.encoder.output_size, z_dims)
        self.loc.weight.data.zero_()
        self.loc.bias.data.zero_()
        self.scale.weight.data.zero_()
        self.scale.bias.data.zero_()

    def encode(self, x):
        x = self.encoder(x)
        loc = self.loc(x)
        scale = self.scale(x)
        return LogitNormal(loc, scale)

    def forward(self, x, state=None):
        if state is not None:
            state[torchbearer.TARGET] = x.detach()

        latent = self.encode(x)

        if state is not None:
            state[LATENT] = latent

        x = self.decoder(latent.rsample())
        return LogitNormal(x, (torch.ones_like(x) * self.var).log())


class PredictionNetwork(nn.Module):
    def __init__(self, encoder_a, encoder_b, z_dims=32):
        super().__init__()
        self.z_dims = z_dims

        self.encoder_a = encoder_a
        for param in self.encoder_a.parameters():
            param.requires_grad = False

        self.encoder_b = encoder_b
        for param in self.encoder_b.parameters():
            param.requires_grad = False

        self.net = nn.Sequential(
            nn.Linear(z_dims, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, z_dims * 2)
        )

        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x, state):
        self.encoder_a.eval()
        self.encoder_b.eval()

        a = self.encoder_a.encode(x).rsample().detach()
        b = self.encoder_b.encode(x)
        b.loc = b.loc.detach()
        b.scale = b.scale.detach()

        state[torchbearer.TARGET] = b

        x = self.net(a)
        loc = x[:, :self.z_dims]
        scale = x[:, self.z_dims:]
        return LogitNormal(loc, scale)


class MINetwork(nn.Module):
    def __init__(self, encoder_a, encoder_b, upper=False):
        super().__init__()
        self.upper = upper

        self.encoder_a = encoder_a
        for param in self.encoder_a.parameters():
            param.requires_grad = False

        self.encoder_b = encoder_b
        for param in self.encoder_b.parameters():
            param.requires_grad = False

    def forward(self, x, state):
        self.encoder_a.eval()
        self.encoder_b.eval()

        if not self.upper:
            x = self.encoder_a(x).rsample()
        return self.encoder_b.encode(x)
