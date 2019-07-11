import argparse
import math
import os
import json
import torch
import torchvision

from models import MixtureVAE


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-directory', type=str, default='checkpoints')
parser.add_argument('--sample-filename', type=int, default='samples_default')
parser.add_argument('--n-samples', type=int, default=100)
args = parser.parse_args()

model_best_path = os.path.join(args.checkpoint_directory,
                                'model_best.pth.tar')
model_args_path = os.path.join(args.checkpoint_directory,
                                'args.json')

with open(model_args_path, encoding='utf-8') as data_file:
    model_args = json.loads(data_file.read())

# HACK
input_channels = 1

model = MixtureVAE(
    input_channels,
    32,
    model_args['latent_size'],
    model_args['n_mixtures'],
    n_filters = 64,
    prior = model_args['prior'],
)

checkpoint = torch.load(model_best_path)
model.load_state_dict(checkpoint['state_dict'])
model = model.eval()



sample_batch = torch.distributions.normal.Normal(
    torch.zeros((args.n_samples, model_args['latent_size'])),
    torch.ones((args.n_samples, model_args['latent_size']))).sample()

samples = model.decoder(sample_batch)
breakpoint()

torchvision.utils.save_image(
    samples.cpu(), f'{args.sample_filename}.png', nrow=int(math.sqrt(args.n_samples)))