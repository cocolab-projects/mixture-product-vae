import math
import pathlib
import pyro
import torch
import torchvision
import numpy as np

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, Timer, EarlyStopping
from ignite.metrics import RunningAverage
from tqdm import tqdm

from models import bernoulli_log_pdf, MultimodelVAE


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


def compute_validation_loss(latent_size, model, val_loader, device,
                            number_of_mixtures=1, n=1):

    val_loss = 0
    prior_distribution = torch.distributions.Normal(0, 1)
    total_examples = 0

    with torch.no_grad():
        for images, _labels in tqdm(val_loader):
            images = images.to(device)

            for image in images:
                total_examples += 1
                image = torch.unsqueeze(image, dim=0)
                mu, logvar = model.encode(image)
                std = (0.5 * logvar).exp()
                encoder_distribution = None
                if model.reparametrize_with == 'normal':
                    encoder_distribution = torch.distributions.Normal(mu, std)
                else:
                    # breaks for 3 channels images
                    encoder_distribution = pyro.distributions.MixtureOfDiagNormals(
                        locs=mu.view(-1, number_of_mixtures, latent_size),
                        coord_scale=std.view(-1, number_of_mixtures, latent_size),
                        component_logits=torch.nn.functional.softmax(
                            model.input_to_logits(image.view(-1, 32 * 32)), dim=1),
                    )
                log_weights = torch.empty(1, n)

                for current_n in range(n):
                    z = encoder_distribution.sample()
                    recon = model.decode(z.to(device))
                    encoder_p = encoder_distribution.log_prob(z)
                    prior_p = prior_distribution.log_prob(z).sum()
                    decoder_p = bernoulli_log_pdf(
                        image.view(1, -1), recon.view(1, -1))
                    log_weight = prior_p.cpu() + decoder_p.cpu() - encoder_p.cpu()
                    log_weights[0][current_n] = torch.sum(log_weight, dim=-1)
                log_weights = math.log(1. / n) + torch.logsumexp(log_weights, dim=1).item()
                val_loss += log_weights

    return val_loss / total_examples


def test_elbo(model, val_data_loader, latent_size, reparametrize_with):
    total_loss = 0
    with torch.no_grad():
        for image, _labels in tqdm(dataset):
            recon, z, mu, logvar, self.logits = model(image)
            loss = model.elbo(
                orig=images,
                z=z,
                recon=recon,
                mu=mu,
                logvar=logvar,
                logits=logits,
                kl_weight=1,
            )
            total_loss += loss

    return total_loss / len(val_data_loader)


def run(
        # Experiment Parameters
        current_run,
        log_directory,
        checkpoint_directory,
        log_interval,

        # Hyperparameters
        batch_size,
        val_batch_size,
        dataset,
        epochs,
        lr,
        latent_size,
        channels,
        number_of_mixtures,
    ):

    log_file = open(log_directory / 'train_log.tsv', 'w')
    log_file.write('ITERATION\tELBO_RUNNING_AVERAGE\n')

    image_size = 32
    input_channels = 3 if dataset == 'CIFAR10' else 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader = get_data_loaders(
        dataset,
        batch_size,
        val_batch_size,
    )

    reparametrize_with = 'normal'

    if number_of_mixtures > 1:
        reparametrize_with = 'mixture_of_normal'

    model = MultimodelVAE(
        input_channels=input_channels,
        image_size=image_size,
        channels=channels,
        z_dim=latent_size,
        reparametrize_with=reparametrize_with,
        mixture_size=number_of_mixtures,
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def step(engine, batch):
        images, _labels = batch
        images.to(device)
        optimizer.zero_grad()
        recon, z, mu, logvar, logits = model(images)

        loss = elbo(
            orig=images,
            z=z,
            recon=recon,
            mu=mu,
            logvar=logvar,
            logits=logits,
        )

        loss.backward()
        optimizer.step()

        return {'ELBO': loss.item()}

    trainer = Engine(step)
    checkpointer_handler = ModelCheckpoint(
        checkpoint_directory, 'mpvae', save_interval=1, n_saved=10,
        require_empty=False
    )
    timer = Time(average=True)

    RunningAverage(output_transform=lambda x: x['ELBO']).attach(trainer, 'ELBO')
    progress_bar = ProgressBar()
    progress_bar.attach(trainer, metric_names=['ELBO'])

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_iteration(engine):
        if (engine.state.iteration - 1) % log_interval:
            epoch = engine.state.epoch
            training_batchs = len(train_loader)
            current_batch_index = engine.state.iteration % training_batchs

            message = '''
                f[{epoch}/{epochs}][{current_batch_index}/{training_batches}]
            '''

            row = '{engine.state.iteration}, {engine.state.metrics["ELBO"]}'

            progress_bar.log_message(message)

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, exception):
        if isinstance(exception, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')

            checkpoint_handler(engine, {
                'model_exception': model,
            })
        else:
            raise exception

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        progress_bar.log_message(
            f'''
            Epoch {engine.state.epoch} done. Time per a batch {timer.value():.3f[s]}
            '''
        )
        time.reset()

    @trainer.on(Events.COMPLETED)
    def save_test_log_likelyhood(engine):
        test_log_likelyhoood = compute_validation_loss(
            latent_size, model, val_loader, device,
            number_of_mixtures=number_of_mixtures, n=10)
        print(f'Valuation Loss {test_log_likelyhoood}')

    @trainer.on(Events.COMPLETED)
    def close_log_file(engine):
        log_file.close()

    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED,
                              handler=checkpoint_handler, to_save={
                                  'model': model,
                              })

    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # Experiment Parameters
    parser.add_argument('--experiments-directory', type=str, default='./experiments')
    parser.add_argument('--log-directory', type=str, default='logs')
    parser.add_argument('--checkpoint-directory', type=str, default='checkpoints')
    parser.add_argument('--experiment-name', type=str, default='default')
    parser.add_argument('--log-interval', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)

    # Hyperparameters
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=200)
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--latent-size', type=int, default=2)
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--number-of-mixtures', type=int, default=1)
    parser.add_argument('--learned-mixture', type=bool, default=False)
    parser.add_argument('--fixed-mixture', type=bool, default=False)
    parser.add_argument('--dynamic-mixture', type=bool, default=True)

    args = parser.parse_args()

    experiments_directory = pathlib.Path(args.experiments_directory)

    current_experiment_directory = experiments_directory / args.experiment_name
    log_directory = current_experiment_directory / args.log_directory
    checkpoint_directory = current_experiment_directory / args.checkpoint_directory

    if (current_experiment_directory).exists():
        # is this really a ValueError?
        raise ValueError(
            f'Experiment with name {args.experiment_name} exists! Aborting.')
    current_experiment_directory.mkdir(parents=True, exists_ok=False)

    for current_run in range(args.runs):
        if args.runs == 1:
            print('Starting experiment...')
        else:
            print(f'Starting experiment run {current_run} of {args.runs}...')

        run(
            # Experiment Parameters
            current_run=current_run,
            log_directory=args.log_directory,
            checkpoint_directory=args.checkpoint_directory,
            log_interval=args.log_interval,

            # Hyperparameters
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            dataset=args.dataset,
            epochs=args.epoch,
            lr=args.lr,
            latent_size=args.latent_size,
            channels=args.channels,
            number_of_mixtures=args.number_of_mixtures,
        )
