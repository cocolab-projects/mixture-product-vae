import os
import math
import copy
import torch
import torchvision
import numpy as np
from tqdm import tqdm

from models import MixtureVAE
from utils import AverageMeter, save_checkpoint


def get_data_loaders(dataset_name, train_batch_size, val_batch_size):

    dataset_class = getattr(torchvision.datasets, dataset_name)

    train_loader = torch.utils.data.DataLoader(
        dataset_class(
            download=True,
            root="./datasets/",
            transform=torchvision.transforms.Compose([
                # for simplicity, we reshape MNIST to 32 bc
                # its so much easier to work with images that
                # are a constant of 32 sized
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


def run(
        checkpoint_directory: str,
        batch_size: int,
        val_batch_size: int,
        dataset: str,
        epochs: int,
        lr: float,
        input_channels: int,
        latent_size: int,
        n_mixtures: int,
    ):

    os.makedirs(checkpoint_directory, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader = get_data_loaders(
        dataset, batch_size, val_batch_size)

    # HACK for now, we should use train/val/test
    test_loader = copy.deepcopy(val_loader)

    model = MixtureVAE(
        input_channels,
        32,
        latent_size,
        n_mixtures,
        n_filters = 32,
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    def train_one_epoch(epoch):
        model.train()
        loss_meter = AverageMeter()

        print('Train Epoch ({}/{})'.format(epoch + 1, epochs))

        with tqdm(total=len(train_loader)) as pbar:
            for x, _ in train_loader:
                batch_size = x.size(0)
                x = x.to(device)
                
                optimizer.zero_grad()
                x_mu, z, z_mu, z_logvar, logits = model(x)
                loss = model.elbo(x, x_mu, z, z_mu, z_logvar, logits)
                loss.backward()
                optimizer.step()

                loss_meter.update(loss.item(), batch_size)
                pbar.update()

                pbar.set_postfix({'train elbo': loss_meter.avg})
        
        return loss_meter.avg

    
    def validate(epoch):
        model.eval()
        loss_meter = AverageMeter()

        print('Val Epoch ({}/{})'.format(epoch + 1, epochs))

        pbar = tqdm(total=len(val_loader))
        with torch.no_grad():
            for x, _ in val_loader:
                batch_size = x.size(0)
                x = x.to(device)

                x_mu, z, z_mu, z_logvar, logits = model(x)
                loss = model.elbo(x, x_mu, z, z_mu, z_logvar, logits)
                loss_meter.update(loss.item(), batch_size)
                pbar.update()

        print('Val ELBO: {}'.format(loss_meter.avg))
        return loss_meter.avg


    def test_log_density(epoch):
        model.eval()
        loss_meter = AverageMeter()

        pbar = tqdm(total=len(test_loader))
        with torch.no_grad():
            for x, _ in test_loader:
                batch_size = x.size(0)
                x = x.to(device)

                log_density = model.log_likelihood(x, n_samples=100)
                loss_meter.update(log_density.item(), batch_size)
                pbar.update()
        
        print('Test Log Density: {}'.format(loss_meter.avg))
        return loss_meter.avg


    best_loss = np.inf
    is_best = False

    for epoch in range(epochs):
        train_elbo = train_one_epoch(epoch)
        val_elbo = validate(epoch)
        
        if val_elbo < best_loss:
            is_best = True
            best_loss = val_elbo

        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch,
            'train_elbo': train_elbo,
            'val_elbo': val_elbo,
        }, is_best=is_best, folder=checkpoint_directory)

    # at this point, load the best model
    model_best_path = os.path.join( checkpoint_directory, 
                                    'model_best.pth.tar')
    checkpoint = torch.load(model_best_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.eval()

    test_loglike = test_log_density(epoch)

    # save this back into the checkpoint
    checkpoint['test_loglike'] = test_loglike
    torch.save(checkpoint, model_best_path)

    return test_loglike


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-directory', type=str, default='checkpoints')
    parser.add_argument('--batch-size', type=int, default=64, help='default: 64')
    parser.add_argument('--val_batch_size', type=int, default=200, help='default: 100')
    parser.add_argument('--dataset', type=str, default='MNIST', help='default: MNIST')
    parser.add_argument('--lr', type=float, default=3e-4, help='default: 3e-4')
    parser.add_argument('--epochs', type=int, default=200, help='default: 200')
    parser.add_argument('--latent-size', type=int, default=2, help='default: 2')
    parser.add_argument('--n-mixtures', type=int, default=1, help='default: 1')
    args = parser.parse_args()

    dset2channel = {'MNIST': 1, 'FashionMNIST': 1, 'CIFAR10': 3}

    test_loglike = run(
        checkpoint_directory=args.checkpoint_directory,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        dataset=args.dataset,
        epochs=args.epochs,
        lr=args.lr,
        input_channels=dset2channel[args.dataset],
        latent_size=args.latent_size,
        n_mixtures=args.n_mixtures,
    )

    print('Test Log-Likelihood: {}'.format(test_loglike))
