from typing import Callable

import torch
import pyro
import argparse
import torchvision
from custom_dist import MixtureOfDiagNormals 

from tqdm import tqdm


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Encoder(torch.nn.Module):

    def __init__(self, input_size, hidden_size, latent_size,
                 activation: Callable = torch.nn.functional.relu):
        super().__init__()
        self.input_to_hidden = torch.nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = torch.nn.Linear(hidden_size, hidden_size)
        self.hidden_to_mu = torch.nn.Linear(hidden_size, latent_size)
        self.hidden_to_logvar = torch.nn.Linear(hidden_size, latent_size)
        self.activation = activation

    def forward(self, image):
        hidden = self.activation(self.input_to_hidden(image))
        hidden = self.activation(self.hidden_to_hidden(hidden))
        mu = self.hidden_to_mu(hidden)
        logvar = self.hidden_to_logvar(hidden)
        return mu, logvar


class Decoder(torch.nn.Module):

    def __init__(self, latent_size, hidden_size, recon_size,
                 activation: Callable = torch.nn.functional.relu):
        super().__init__()
        self.latent_to_hidden = torch.nn.Linear(latent_size, hidden_size)
        self.hidden_to_hidden_1 = torch.nn.Linear(hidden_size, hidden_size)
        self.hidden_to_hidden_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.hidden_to_recon = torch.nn.Linear(hidden_size, recon_size)
        self.activation = activation

    def forward(self, z):
        hidden = self.activation(self.latent_to_hidden(z))
        hidden = self.activation(self.hidden_to_hidden_1(hidden))
        hidden = self.activation(self.hidden_to_hidden_2(hidden))
        recon = self.hidden_to_recon(hidden)
        return recon


class MultimodelVAE(torch.nn.Module):

    def __init__(self, input_size, hidden_size, latent_size,
                 reparametrize_with: str = 'normal', mixture_size: int = 10):
        super().__init__()

        self.input_size = input_size
        self.latent_size = latent_size
        self.mixture_size = mixture_size
        self.reparametrize_with = reparametrize_with
        self.encode = Encoder(input_size, hidden_size, latent_size * mixture_size)
        self.decode = Decoder(latent_size, hidden_size, input_size)

        if reparametrize_with == 'mixture-of-normal':
            self.input_to_logits = torch.nn.Linear(input_size, mixture_size)

    def reparametrize(self, mu, logvar, logits=None, image=None):
        std = (0.5 * logvar).exp()
        logits = None
        if self.reparametrize_with == 'mixture-of-normal':
            temp = self.input_to_logits(image)
            logits = torch.nn.functional.softmax(temp, dim=1) + 1e-5

        if self.reparametrize_with == 'normal':
            return torch.distributions.normal.Normal(loc=mu, scale=std).rsample(), logits
        if self.reparametrize_with == 'mixture-of-normal':
            return MixtureOfDiagNormals(
                locs=mu.view(-1, self.mixture_size, 
                    self.latent_size),
                coord_scale=(std.view(-1, self.mixture_size,
                    self.latent_size)),
                component_logits=logits,
            ).rsample(), logits

    def encoder(self, image):
        mu, logvar = self.encode(image)
        return mu, logvar

    def decoder(self, z):
        recon = self.decode(z)
        return recon

    def forward(self, image):
        mu, logvar = self.encoder(image)

        z, logits = self.reparametrize(mu, logvar, logits=None, image=image)
        recon = self.decoder(z)
        return recon, z, mu, logvar, logits

    def eblo_loss(self, orig, z, recon, mu, logvar, logits, kl_weight=1):
        bce_loss = torch.sum(
            torch.nn.functional.binary_cross_entropy(
                torch.sigmoid(recon),
                orig,
                reduction='none'), dim=1)
    
        normal_prob = - torch.sum(torch.distributions.normal.Normal(
            loc=0, scale=1).log_prob(z), dim=1)

        std = (0.5 * logvar).exp()
        mixture_prob = - MixtureOfDiagNormals(
            locs=mu.view(-1, self.mixture_size, self.latent_size),
            coord_scale=(std.view(-1, self.mixture_size, self.latent_size)),
            component_logits=logits).log_prob(z)
        return torch.mean(bce_loss + normal_prob - mixture_prob, dim=0)

# kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
# return torch.mean(bce_loss + (kl_loss * kl_weight), dim=0)

import pdb
import traceback


class DetectAnomaly(torch.autograd.detect_anomaly):

    def __init__(self):
        super().__init__()

    def __enter__(self):
        super().__enter__()
        return self

    def __exit__(self, type, value, trace):
        super().__exit__()
        if isinstance(value, RuntimeError):
            traceback.print_tb(trace)
            pdb.set_trace()


def train(epochs, model, optimizer, dataset_loader):
    for epoch in range(epochs):
        total_loss = 0
        running_average = AverageMeter()

        print(f'Starting epoch {epoch + 1} of {epochs}...')

        for images, _labels in tqdm(dataset_loader):
            with torch.autograd.detect_anomaly():
                recon, z, mu, logvar, logits = model(images)
                loss = model.eblo_loss(images, z, recon, mu, logvar, logits)
                total_loss += loss.item()
                running_average.update(loss.item())
                loss.backward()
                optimizer.step()

        print(f'\t> ELBO: {loss}')
        print(f'\t> Running Average ELBO: {running_average.avg}')


def main():
    model = MultimodelVAE(784, 512, 32, reparametrize_with='mixture-of-normal',
                          mixture_size=10)
    epochs = 10
    batch_size = 64
    optimizer = torch.optim.Adam(model.parameters())
    train_loader   = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(
                lambda x: torch.squeeze(x.view(-1, 784))
            )
        ])),
    batch_size=batch_size, shuffle=True)
    train(epochs, model, optimizer, dataset_loader=train_loader)


if __name__ == '__main__':
    main()
