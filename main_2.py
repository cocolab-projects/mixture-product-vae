from typing import Union

import math
import pathlib
import pyro
import torch
import torchvision

import numpy as np

from dataclasses import dataclass
from ignite.contrib.handles import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer, EarlySpotting
from ignite.metrics import RunningAverage
from tqdm import tqdm


def get_data_loaders(dataset_name, train_batch_size, val_batch_size):

    dataset_class = getattr(torchvision.datasets, dataset_name)

    train_loader = torch.utils.data.DataLoader(
        dataset_class(
            download=True,
            root="./datasets/",
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(32),
                torchvision.transforms.ToTensor(),
            ]),
            train=True
        ),
        batch_size=train_batch_size,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset_class(
            download=False,
            root="./datasets/",
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(32),
                torchvision.transforms.ToTensor(),
            ]),
            train=False,
        ),
        batch_size=val_batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


def get_conv_output_dim(I, K, P, S):
    # I = input height / lenght
    # K = filter size
    # P = padding
    # S = stride
    # O = output height / length
    O = (I - K + 2 * P) / float(S) + 1
    return int(O)


def gen_32_conv_output_dim(s):
    s = get_conv_output_dim(s, 2, 0, 2)
    s = get_conv_output_dim(s, 2, 0, 2)
    s = get_conv_output_dim(s, 2, 0, 2)
    return s


def bernoulli_log_pdf(x, mu):
    mu = torch.clamp(mu, 1e-7, 1.-1e-7)
    return torch.sum(x * torch.log(mu) + (1.  -x) * torch.log(1. - mu), dim=1)


class ImagineEncoder(torch.nn.Module):

    @dataclass
    class Config:
        input_channels: int
        image_size: int
        z_dim: int
        n_filter: int
        mixtures: int

    def __init__(self, config: self.Config):
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(config.input_channels),
        )


class ImageDecoder(torch.nn.Module):

    @dataclass
    class Config:
        output_channels: int
        image_size: int
        z_dim: int
        n_filter: int

    def __init__(self, config: self.Config):
        self.output_channels = config.output_channels
        self.image_size = config.image_size
        self.z_dim = config.z_dim
        self.n_filter = config.n_filter

        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(n_filters * 4, n_filters * 4, 2, 2,
                                     padding=0),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 2, 2,
                                     padding=0),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(n_filters * 2, n_filters, 2, 2, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(n_filters, output_channels, 1, 1, padding=0),
        )

        cout = gen_32_conv_output_dim(image_size)


class MultimodelVAE(torch.nn.Module):

    @dataclass
    class Config:
        z_dim: int
        input_channels: int
        image_size: int
        hidden_channels: int
        mixtures: int
        loss: ELBO

    def __init__(self, config: self.Config):
        pass

        self.mixture_shape = (-1, self.mixtures, self.z_dim)
        if mixtures > 1:
            self.input_to_logits = torch.nn.Linear(
                self.input_channels * self.image_size * self.image_size, mixtures)

    def get_logits(self, image):
        return self.input_to_logits(
            image.view(
                -1, self.input_channels * self.image_size * self.image_size))

    def reparam(self, mu, logvar, logits=None, image=None):
        std = (0.5 * logvar).exp()
        if self.mixtures == 1:
            return torch.distributions.normal.Normal(
                loc=mu, scale=std).rsample()

        return pyro.distributions.MixtureOfDiagNormals(
            locs=mu.view(self.mixture_shape),
            coord_scale=std.view(self.mixture_shape)<
            component_logits=self.logits
        ).rsample()

    def forward(self, image):
        self.logits = self.get_logits(images) if mixtures > 1 else None
        mu, logvar = self.encode(image)
        z = self.reparam(mu, logvar)
        return recon, z, mu, logvar, self.logits


def get_num_interval(k):
    n = 0
    while math.factorial(n) < k:
        n += 1
    return n


def get_fixed_init(n, a, b):
    interval = np.linspace(a, b, n + 1)
    inits = []
    for i in range(0, len(interval) - 1):
        for j in range(i + 1, len(interval)):
            inits.append([interval[i], interval[j]])
    return np.array(inits)


def get_fixed_init_mixtures(k, a, b):
    n = get_num_interval(k)
    return np.random.choice(get_fixed_init(n, a, b), k)


class ELBO(torch.nn.Module):

    @dataclass
    class Config:
        prior_parameters_are: str
        mixtures: int
        z_dim: int

    def __init__(self, config: self.Config):
        '''
        '''
        self.prior_parameters_are = config.prior_parameters_are
        self.mixtures = config.mixtures
        self.z_dim = config.z_dim

        self.parameters_are = parameters_are


    def _init_prior(self):
        if self.prior_is == 'normal':
            self._init_normal_prior()
            self.prior_func = self._normal_prior
        elif self.prior_is == 'mixtured_learned':
            self._init_learned_prior_params()
            self.prior_func = self._learned_prior
        elif self.prior == 'mixture_fixed':
            self._fixed_prior_params()
            self.prior_func = self._fixed_prior

    @property
    def mixtures_reshape_shape(self):
        return (-1, self.mixtures, self.z_dim)

    def _init_learned_prior(self):
        parameters_size = (self.mixtures, self.z_dim)
        self.mu_parameters = torch.nn.Parameters(
            torch.randn(parameters_size))
        self.std_parameters = torch.ones(parameters_size)
        self.logits = torch.nn.Parameters(torch.randn(self.mixtures))

    def _fixed_prior(self):
        pass
        q_prob(self, z, mu=None, log)
    def posterior_prob(self, z, mu=None, logvar=None, logits=None):
        self.posterior = - pro.distributions.MixtureOfDiagNormals(

        )

    def prior_prob(self, z, mu=None, logvar=None, logits=None):

        if self.parameters_are == 'normal':
            self.prior_prob = 
        elif self.parameters_are == 'fixed':
            self.prior_prob = 
        elif self.parameters_are == 'learned':
            self.prior_prob = - pyro.distributions.MixtureOfDiagNormals(
                locs=self.mu_parameters,
                coord_scale=self.std_parameters,
                component_logits=self.logits,
            ).log_prob(z)
        elif self.parameters_are == 'dynamic':
            std = (0.5 * logvar).exp()
            self.prior_prob = - pyro.distributions.MixtureOfDiagNormals(
                mu=mu,
                locs=std,
                component_logits=self.logits,
            ).log_prob(z)

    def kl_diverg(self):
        if self.mixtures < 1:
            return -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        posterior_prob = self.

    def forward(self, orig, z, recon, mu, logvar, logits, kl_weight: int = 1):
        pass
