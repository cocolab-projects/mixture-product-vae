from typing import Callable, Tuple

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils import data


class SimpleMultimodelDataset(data.TensorDataset):

    def __init__(self, number_of_examples) -> None:
        '''
        Args:
        Returns:
        '''
        self.number_of_examples = number_of_examples
        self.dataset = torch.zeros(number_of_examples, 1)
        mask = self.dataset.bernoulli(p=0.5).byte()
        self.dataset[mask] = self.dataset[mask].normal_(mean=-1, std=0.1)
        self.dataset[~mask] = self.dataset[~mask].normal_(mean=1, std=0.1)

    def __len__(self) -> int:
        return self.number_of_examples

    def __getitem__(self, index) -> torch.Tensor:
        return self.dataset[index]


class RunningAverage:

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.count = 0

    def update(self, new_value):
        self.value += new_value
        self.count += 1

    @property
    def running_average(self):
        return self.value / self.count


def reparameterization_trick(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    batch_size, latent_length = mean.shape
    epsilon = torch.randn(batch_size, latent_length)
    std = torch.exp(0.5 * logvar)
    return std * epsilon + mean


def log_pdf(x, reconstructed_mean, reconstructed_logvar) -> float:
    reconstructed_std = torch.exp(0.5 * reconstructed_logvar)
    return torch.sum(
        torch.distributions.normal.Normal(
            loc=reconstructed_mean,
            scale=reconstructed_std
        ).log_prob(x)
    )


def train(epochs: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
          dataset_loader: data.TensorDataset):

    ELBO_running_average = RunningAverage()

    for current_epoch in range(epochs):
        print(f'> Starting epoch {current_epoch} of {epochs + 1}...')
        for batch in tqdm(dataset_loader):
            reconstructed_mean, reconstructed_logvar, latent, latent_mean, latent_logvar = model(batch)
            loss = ELBO(batch, reconstructed_mean, reconstructed_logvar, latent_mean, latent_logvar)
            loss.backward()
            ELBO_running_average.update(loss.item())
            optimizer.step()
        current_running_average = ELBO_running_average.running_average
        print(f'\t| ELBO Running Average: {current_running_average}')

    _inputs = dataset_loader.dataset.dataset[:3000]
    inputs = torch.flatten(_inputs)
    results, _ = model.encode(_inputs)
    results = results.flatten().detach()

    plt.hist(inputs, bins=100, density=False)
    plt.title('Dataset')
    plt.savefig('graphs/dataset.png', bbox_inches='tight')

    plt.clf()

    plt.hist(results, bins=100, density=False)
    plt.title('Results')
    plt.savefig('graphs/results.png', bbox_inches='tight')


def gaussian_kl_divergence(mean: torch.Tensor, logvar: torch.Tensor, weight: float = 1):
    kl_div = 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return weight * kl_div


def ELBO(x: torch.Tensor, reconstructed_mean, reconstructed_logvar,
         latent_mean, latent_logvar) -> float:
    pdf_loss = log_pdf(x, reconstructed_mean, reconstructed_logvar)
    kl_loss = gaussian_kl_divergence(latent_mean, latent_logvar, weight=100)
    return torch.mean(-pdf_loss - kl_loss)


class BaselineVAE(torch.nn.Module):

    def __init__(self, input_size, hidden_size, latent_size,
                 activation: Callable = torch.tanh) -> None:
        super().__init__()
        self.activation = activation
        self.encoder_input_to_hidden = torch.nn.Linear(input_size, hidden_size)
        self.encoder_hidden_to_mean = torch.nn.Linear(hidden_size, latent_size)
        self.encoder_hidden_to_logvar = torch.nn.Linear(hidden_size, latent_size)

    def encode(self, x: torch.Tensor):
        hidden = self.encoder_input_to_hidden(x)
        latent_mean = self.encoder_hidden_to_mean(hidden)
        latent_logvar = self.encoder_hidden_to_logvar(hidden)
        return latent_mean, latent_logvar

    def reparameterize(self, latent_mean: torch.Tensor,
                       latent_logvar: torch.Tensor) -> torch.Tensor:
        return reparameterization_trick(latent_mean, latent_logvar)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        reconstructed_mean = latent
        reconstructed_sigma = torch.ones_like(reconstructed_mean) * 0.1
        reconstructed_logvar = 2.0 * torch.log(reconstructed_sigma)
        reconstructed_logvar = reconstructed_logvar.to(latent.device)
        return reconstructed_mean, reconstructed_logvar

    def forward(self, x: torch.Tensor):
        latent_mean, latent_logvar = self.encode(x)
        latent = self.reparameterize(latent_mean, latent_logvar)
        reconstructed_mean, reconstructed_logvar = self.decode(latent)
        return reconstructed_mean, reconstructed_logvar, latent, latent_mean, latent_logvar


def add_hyperparameters(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--dataset-size', default=1000, type=int)
    parser.add_argument('--hidden-size', default=8, type=int)
    parser.add_argument('--latent-size', default=1)
    return parser


def main():
    parser = argparse.ArgumentParser('Baseline Multimodel VAE')
    parser = add_hyperparameters(parser)
    args = parser.parse_args()

    model = BaselineVAE(input_size=1, hidden_size=args.hidden_size, latent_size=args.latent_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = SimpleMultimodelDataset(args.dataset_size)
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size)
    train(args.epochs, model, optimizer, dataset_loader)


if __name__ == '__main__':
    main()
