import math
import pathlib
import pyro
import torch
import torchvision

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from tqdm import tqdm


def gen_32_conv_output_dim(s):
    s = get_conv_output_dim(s, 2, 0, 2)
    s = get_conv_output_dim(s, 2, 0, 2)
    s = get_conv_output_dim(s, 2, 0, 2)
    return s


def get_conv_output_dim(I, K, P, S):
    # I = input height/length
    # K = filter size
    # P = padding
    # S = stride
    # O = output height/length
    O = (I - K + 2 * P) / float(S) + 1
    return int(O)


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


class ImageEncoder(torch.nn.Module):

    def __init__(self, input_channels, image_size, z_dim, n_filters, number_of_mixtures):

        super().__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, n_filters, 2, 2, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(n_filters, n_filters * 2, 2, 2, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(n_filters * 2, n_filters * 4, 2, 2, padding=0),
            torch.nn.ReLU(),
        )

        cout = gen_32_conv_output_dim(image_size)

        self.fc = torch.nn.Linear(n_filters * 4 * cout**2, z_dim * 2 * number_of_mixtures)
        self.cout = cout
        self.n_filters = n_filters

    def forward(self, image):

        batch_size = image.size(0)

        out = self.conv(image)
        out = out.view(batch_size, self.n_filters * 4 * self.cout**2)

        z_params = self.fc(out)

        z_mu, z_logvar = torch.chunk(z_params, 2, dim=1)

        return z_mu, z_logvar


class ImageDecoder(torch.nn.Module):

    def __init__(self, output_channels, image_size, z_dim, n_filters):

        super().__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(n_filters * 4, n_filters * 4, 2, 2, padding=0),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 2, 2, padding=0),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(n_filters * 2, n_filters, 2, 2, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(n_filters, output_channels, 1, 1, padding=0),
        )

        cout = gen_32_conv_output_dim(image_size)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(z_dim, n_filters * 4 * cout**2),
            torch.nn.ReLU()
        )

        self.cout = cout
        self.n_filters = n_filters
        self.output_channels = output_channels
        self.image_size = image_size

    def forward(self, z):
        batch_size = z.size(0)
        out = self.fc(z)
        out = out.view(batch_size, self.n_filters * 4, self.cout, self.cout)
        out = self.conv(out)
        x_logits = out.view(batch_size, self.output_channels, self.image_size, self.image_size)
        x_mu = torch.sigmoid(x_logits)
        return x_mu


class MultimodelVAE(torch.nn.Module):

    def __init__(
        self,
        input_channels: int,
        image_size: int,
        channels: int,
        z_dim: int,
        reparametrize_with: int,
        mixture_size: int,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.input_channels = input_channels
        self.image_size = image_size
        self.hidden_channels = channels
        self.reparametrize_with = reparametrize_with
        self.mixture_size = mixture_size

        self.encoder = ImageEncoder(
            input_channels=input_channels, image_size=image_size, z_dim=z_dim,
            n_filters=channels, number_of_mixtures=mixture_size)
        self.decoder = ImageDecoder(
            output_channels=input_channels, image_size=image_size, z_dim=z_dim,
            n_filters=channels)

        if reparametrize_with == 'mixture_of_normal':
            self.input_to_logits = torch.nn.Linear(
                self.input_channels * self.image_size * self.image_size, mixture_size)
        self.logits = None

    def encode(self, image):
        mu, logvar = self.encoder(image)
        return mu, logvar

    def decode(self, z):
        recon = self.decoder(z)
        return recon

    def reparametrize(self, mu, logvar, logits=None, image=None):
        std = (0.5 * logvar).exp()
        logits = None
        if self.reparametrize_with == 'normal':
            return torch.distributions.normal.Normal(loc=mu, scale=std).rsample(), self.logits
        if self.reparametrize_with == 'mixture_of_normal':
            batch_size = mu.size(0)
            temp = self.input_to_logits(
                image.view(batch_size, self.input_channels * self.image_size * self.image_size))
            self.logits = torch.nn.functional.softmax(temp, dim=1) 
            return pyro.distributions.MixtureOfDiagNormals(
                locs=mu.view(-1, self.mixture_size, 
                    self.z_dim),
                coord_scale=(std.view(-1, self.mixture_size,
                    self.z_dim)),
                component_logits=self.logits,
            ).rsample(), self.logits

    def forward(self, image):
        if self.reparametrize_with == 'mixture_of_normal':
            self.logits = self.input_to_logits(
                image.view(-1, self.input_channels * self.image_size * self.image_size))
        else:
            self.logits = None

        mu, logvar = self.encode(image)

        z, logits = self.reparametrize(
            mu,
            logvar,
            logits=self.logits,
            image=image
        )
        recon = self.decode(z)
        return recon, z, mu, logvar, self.logits

    def elbo(self, orig, z, recon, mu, logvar, logits, number_of_mixtures,
            latent_size, kl_weight=1):

        recon_loss = bernoulli_log_pdf(orig.flatten(), mu.flatten())
        kl_diverg = kl_divergence(
            mu,
            logvar,
            z,
            logits,
            number_of_mixtures,
            latent_size,
            reparametrize_with=self.reparametrize_with,
        )
        return torch.mean(
        recon_loss + (kl_diverg * kl_weight), dim=0)


def kl_divergence_normal(mu, logvar, z):
    return -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def kl_divergence_mixture(mu, logvar, z, logits, number_of_mixtures, latent_size):
    normal_prob = - torch.sum(
        torch.distributions.normal.Normal(
            loc=0,
            scale=1).log_prob(z), dim=1)
    std = (0.5 * logvar).exp()
    mixture_prob = - pyro.distributions.MixtureOfDiagNormals(
        locs=mu.view(-1, number_of_mixtures, latent_size),
        coord_scale=std.view(-1, number_of_mixtures, latent_size),
        component_logits=torch.nn.functional.softmax(logits, dim=1)
    ).log_prob(z)
    result = normal_prob - mixture_prob
    return result


def kl_divergence(mu, logvar, z, logits, number_of_mixtures,
                  latent_size, reparametrize_with):
    if reparametrize_with == 'normal':
        return kl_divergence_normal(mu, logvar, z)
    elif reparametrize_with == 'mixture_of_normal':
        return kl_divergence_mixture(mu, logvar, z, logits,
                                     number_of_mixtures,
                                     latent_size)


def bernoulli_log_pdf(x, mu):
    mu = torch.clamp(mu, 1e-7, 1.-1e-7)
    return torch.sum(x * torch.log(mu) + (1. - x) * torch.log(1. - mu), dim=1)


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
            loss = elbo(
                orig=images,
                z=z,
                recon=recon,
                mu=mu,
                logvar=logvar,
                logits=logits,
                number_of_mixtures=number_of_mixtures,
                latent_size=latent_size,
                kl_weight=1,
                reconstruction_with=reparametrize_with,
            )
            total_loss += loss

    return total_loss / len(val_data_loader)


def run(
        batch_size: int,
        dataset: str,
        val_batch_size: int,
        epochs: int,
        lr: float,
        latent_size: int,
        channels: int,
        number_of_mixtures: int,
        log_interval: int,
    ):

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
            number_of_mixtures=number_of_mixtures,
            latent_size=latent_size,
            kl_weight=1,
            reconstruction_with=reparametrize_with,
        )

        loss.backward()
        optimizer.step()

        return {'ELBO': loss.item()}

    trainer = Engine(step)
    checkpointer_handler = ModelCheckpoint(
        checkpoint_directory, 'mpvae', save_interval=1, n_saved=10, require_empty=False)
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

            log_filename = pathlib.Path(logs_path) / logs_filename
            row = '{engine.state.iteration}, {engine}'

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

    if (pathlib.Path(args.experiments_directory) / args.experiment_name).exists():
        # is this really a ValueError?
        raise ValueError(
            f'Experiment with name {args.experiment_name} exists! Aborting.')

    for current_run in range(args.runs):
        if args.runs == 1:
            print('Starting experiment...')
        else:
            print(f'Starting experiment run {current_run} of {args.runs}...')

        run(
            current_run=current_run,
            log_directory=args.log_directory,
            checkpoint_directory=args.checkpoint_directory,
            log_interval=args.log_interval,

            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            dataset=args.dataset,
            epochs=args.epoch,
            lr=args.lr,
            latent_size=args.latent_size,
            channels=args.channels,
            number_of_mixtures=args.number_of_mixtures,
        )
