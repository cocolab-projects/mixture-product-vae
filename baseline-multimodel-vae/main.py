from typing import NamedTuple, Callable, Tuple

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils import data


class RunningAverage:

    def __init__(self, value):
        self._value = value
        self.count = 1

    def update(self, new_value):
        self._value += new_value
        self.count += 1

    @property
    def value(self):
        return self._value / self.count


class Latent(NamedTuple):
    mu: torch.Tensor
    sigma: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.mu.shape[0]


def reparameterize_trick(latent: Latent) -> torch.Tensor:
    '''
    '''
    epsilon = torch.randn(latent.batch_size, 1)
    return latent.sigma * epsilon + latent.mu


class BaselineVAE(torch.nn.Module):
    '''
    '''

    def __init__(self, hidden_size: int, input_size: int = 1,
                 latent_size: int = 1,
                 activation: Callable = torch.nn.functional.relu) -> None:
        '''
        Args:
            input_size:
            hidden_size:
            latent_size:
            activation:
        '''
        super().__init__()
        self.encoder_input_to_hidden = torch.nn.Linear(input_size, hidden_size)
        self.encoder_hidden_to_mu = torch.nn.Linear(hidden_size, latent_size)
        self.encoder_hidden_to_sigma = torch.nn.Linear(hidden_size, latent_size)
        self.activation = activation

    def encode(self, x: torch.Tensor) -> NamedTuple:
        '''
        Args:
            x:
        Returns:
            A namedtuple that contains mu and sigma properties with a dimension
            of the latent_size.
        '''
        hidden = self.activation(self.encoder_input_to_hidden(x))
        mu = self.encoder_hidden_to_mu(hidden)
        sigma = self.encoder_hidden_to_sigma(hidden)
        return Latent(mu, sigma)

    def decode(self, latent: Latent) -> torch.Tensor:
        '''
        '''
        z = reparameterize_trick(latent)
        x_mu = z
        return x_mu, 0.1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Latent]:
        latent = self.encode(x)
        x_mu, x_sigma = self.decode(latent)
        return x, x_mu, x_sigma, latent


class SimpleMultimodelDataset(data.TensorDataset):

    def __init__(self, number_of_examples):
        self.number_of_examples = number_of_examples
        self.dataset = torch.zeros(number_of_examples, 1)
        mask = self.dataset.bernoulli(p=0.5).byte()
        self.dataset[mask] = self.dataset[mask].normal_(mean=-10, std=1)
        self.dataset[~mask] = self.dataset[~mask].normal_(mean=10, std=1)

    def __len__(self) -> int:
        return self.number_of_examples

    def __getitem__(self, index) -> torch.Tensor:
        return self.dataset[index]


def construct_dataset(number_of_examples) -> torch.utils.data.Dataset:
    dataset = SimpleMultimodelDataset(number_of_examples)
    return dataset


def pdf(x, x_mu, x_sigma) -> torch.Tensor:
    return -torch.sum(
        torch.distributions.Normal(loc=x_mu, scale=x_sigma).log_prob(x), dim=1)


def gaussian_kl_divergence(latent: Latent, weight = 1: float) -> float:
    return weight * 0.5 * torch.sum(
        1 + latent.sigma - latent.mu.pow(2) - latent.sigma.exp(), dim=1)


def ELBO(x: torch.Tensor, x_mu: torch.Tensor, x_sigma, latent: Latent) -> float:
    pdf_loss
    kl_loss
    return pdf_loss - kl_loss
    return torch.mean(pdf(x, x_mu, x_sigma) - guassian_kl_divergence(latent, weight=0.01), dim=0)


def train(epochs: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
          dataset: torch.utils.data.TensorDataset):
    loss_running_average = RunningAverage(0)
    for current_epoch in range(epochs):
        print(f'Starting epoch {current_epoch} of {epochs}...')
        for index, batch in enumerate(tqdm(dataset)):
            x, x_mu, x_sigma, latent = model(batch)
            loss = ELBO(x, x_mu, x_sigma, latent)
            loss_running_average.update(loss.item())
            optimizer.step()
        print(ra.value)

    samples = Latent(mu=torch.randn(1, 3000), sigma=torch.randn(1, 3000))

    x_mu, x_sigma = model.decode(samples)

    prior = torch.mean(torch.distributions.Normal(samples.mu, samples.sigma).sample(), dim=0)
    results = torch.mean(torch.distributions.Normal(x_mu, x_sigma).sample(), dim=0)

    plt.hist(torch.flatten(dataset.dataset.dataset[:3000]), bins=100, density=False)
    plt.title('Dataset')
    plt.savefig('graphs/dataset.png', bbox_inches='tight')

    plt.clf()

    plt.hist(prior, bins=100, density=False)
    plt.title('Prior')
    plt.savefig('graphs/prior.png', bbox_inches='tight')

    plt.clf()

    plt.hist(results, bins=100, density=False)
    plt.title('Results')
    plt.savefig('graphs/results.png', bbox_inches='tight')


def add_hyperparameters(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch-size', default=8)
    parser.add_argument('--dataset-size', default=10000)
    parser.add_argument('--hidden-size', default=16)
    parser.add_argument('--latent-size', default=1)
    return parser


def main():
    parser = argparse.ArgumentParser(prog='Baseline Multimodel VAE')
    parser = add_hyperparameters(parser)
    args = parser.parse_args()

    model = BaselineVAE(args.hidden_size, latent_size=args.latent_size)
    optimizer = torch.optim.Adam(model.parameters())
    dataset = torch.utils.data.DataLoader(construct_dataset(args.dataset_size),
                                          batch_size=args.batch_size)
    print('Starting training with')
    tensorboard = SummaryWriter()
    train(args.epochs, model, optimizer, dataset, tensorboard)


if __name__ == '__main__':
    main()
