import math
import pyro
import torch
import torchvision

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
        input_channels,
        image_size,
        channels,
        z_dim,
        reparametrize_with,
        mixture_size,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.input_channels = input_channels
        self.image_size = image_size
        self.hidden_channels = channels
        self.reparametrize_with = reparametrize_with
        self.mixture_size = mixture_size

        self.encoder = ImageEncoder(
            input_channels=input_channels, image_size=image_size, z_dim=z_dim, n_filters=channels,
            number_of_mixtures=mixture_size)
        self.decoder = ImageDecoder(
            output_channels=input_channels, image_size=image_size, z_dim=z_dim, n_filters=channels)

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


def kl_divergence_normal(mu, logvar, z):
    return -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def kl_divergence_mixture(mu, logvar, z, logits,
                          number_of_mixtures,
                          latent_size):
    normal_prob = - torch.sum(
        torch.distributions.normal.Normal(
            loc=0,
            scale=1).log_prob(z), dim=1)
    std = (0.5 * logvar).exp()
    # double negative?
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


def elbo(orig, z, recon, mu, logvar, logits, number_of_mixtures,
         latent_size, kl_weight=1, reconstruction_with='normal'):

    bce_loss = torch.sum(
        torch.nn.functional.binary_cross_entropy(
            input=recon.view(-1, 32 * 32),
            target=orig.view(-1, 32 * 32),
            reduction='none'
        ),
        dim=1,
    )
    kl_diverg = kl_divergence(
        mu,
        logvar,
        z,
        logits,
        number_of_mixtures,
        latent_size,
        reparametrize_with=reconstruction_with,
    )
    return torch.mean(
       bce_loss + (kl_diverg * kl_weight), dim=0)


def bernoulli_log_pdf(x, mu):
    mu = torch.clamp(mu, 1e-7, 1.-1e-7)
    return torch.sum(x * torch.log(mu) + (1. - x) * torch.log(1. - mu), dim=1)


def compute_validation_loss(latent_size, model, val_loader, device, number_of_mixtures=1, n=1):

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
        runs: int,
    ):

    image_size = 32
    input_channels = 1
    if dataset == 'CIFAR10':
        input_channels = 3
    
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    train_loader, val_loader = get_data_loaders(
        dataset,
        batch_size,
        val_batch_size,
    )

    reparametrize_with = 'normal'

    if number_of_mixtures > 1:
        reparametrize_with = 'mixture_of_normal'

    for run in range(runs):
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

        done = False

        for epoch in range(epochs):
            last_elbo = float('inf')
            patience = 5
            average_elbo = AverageMeter()
            print(f'Epoch {epoch + 1} of {epochs}...')
            for current_batch, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                images, _labels = batch
                images = images.to(device)

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
                average_elbo.update(loss.item())

                loss.backward()
                optimizer.step()

            if average_elbo.val < last_elbo:
                patience -= 1
                if patience == 0:
                    done = True
            else:
                last_elbo = average_elbo.val 
            
            print(f'Average ELBO: {average_elbo.avg}')

            if done or (epoch == (epochs - 1)):
                val_loss = compute_validation_loss(latent_size, model, val_loader, device,
                                            number_of_mixtures=number_of_mixtures, n=10)
                print(f'Valuation Loss {val_loss}')

                with open(f'{dataset}_{number_of_mixtures}_{run}', 'w') as f:
                    f.write(str(val_loss))
                break


if __name__ == '__main__':

    run(
        batch_size=64,
        dataset='MNIST',
        val_batch_size=200,
        epochs=200,
        lr=2e-4,
        latent_size=2,
        channels=16,
        number_of_mixtures=1,
        log_interval=10,
        runs=3,
    )
