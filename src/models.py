import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from utils import get_fixed_init, get_num_interval


def gen_32_conv_output_dim(s: int):
    s = get_conv_output_dim(s, 2, 0, 2)
    s = get_conv_output_dim(s, 2, 0, 2)
    s = get_conv_output_dim(s, 2, 0, 2)
    return s


def get_conv_output_dim(I: int, K: int, P: int, S: int):
    # I = input height/length
    # K = filter size
    # P = padding
    # S = stride
    # O = output height/length
    O = (I - K + 2 * P) / float(S) + 1
    return int(O)


def kl_divergence_normal_and_spherical(mu, logvar):
    """
    Closed-form representation of KL divergence between a N(mu, sigma)
    posterior and a spherical Gaussian N(0, 1) prior.

    See https://arxiv.org/abs/1312.6114 for derivation.
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def kl_divergence_mixture_and_spherical(z_mu, z_logvar, z, component_weights):
    """
    KL Divergence between a mixture of normals posterior and a spherical 
    Gaussian N(0, 1) perior. Derivation is as follows:

    KL[q(z|x) || p(z)] = E_{q(z|x)}[log q(z|x) - log p(z)]

    Unfortunately, no analytical solution here. We must use a 
    approximate solution.
    """
    log_q_z_given_x = mixture_normal_log_pdf(z, z_mu, z_logvar, component_weights)
    log_p_z = isotropic_gaussian_log_pdf(z)
    return log_q_z_given_x - log_p_z


def kl_divergence_normal_and_mixture(z_mu, z_logvar, z,
                                     prior_mu, prior_logvar, prior_weights):
    """
    KL Divergence between a gaussian posterior and a mixture
    of normals prior. Derivation is as follows:

    KL[q(z|x) || p(z)] = E_{q(z|x)}[log q(z|x) - log p(z)]
    """
    log_q_z_given_x = gaussian_log_pdf(z, z_mu, z_logvar)
    log_p_z = mixture_normal_log_pdf(z, prior_mu, prior_logvar, prior_weights)
    return log_q_z_given_x - log_p_z

def kl_divergence_mixture_and_mixture(z_mu, z_logvar, z, logits,
                                      prior_mu, prior_logvar, prior_weights):
    """
    KL Divergence between a mixture of normals posterior and another
    mixture of normals prior. Derivation is as follows:

    KL[q(z|x) || p(z)] = E_{q(z|x)}[log q(z|x) - log p(z)]
    """
    log_q_z_given_x = mixture_normal_log_pdf(z, z_mu, z_logvar, logits)
    log_p_z = mixture_normal_log_pdf(z, prior_mu, prior_logvar, prior_weights)
    return log_q_z_given_x - log_p_z


def bernoulli_log_pdf(x: torch.Tensor, mu: torch.Tensor):
    """
    Log probability distribution function for Bernoulli distributions.

        Let pi be logits.
        pdf     = pi^x * (1 - pi)^(1-x)
        log_pdf = x * log(pi) + (1 - x) * log(1 - pi)
    
    In practice, we need to clamp pi (the logits) because if it 
    becomes 0 or 1, then log(0) will be nan.
    """
    mu = torch.clamp(mu, 1e-7, 1.-1e-7)
    return torch.sum(x * torch.log(mu) + (1. - x) * torch.log(1. - mu), dim=1)


def gaussian_log_pdf(x, mu, logvar):
    sigma = torch.exp(0.5 * logvar)
    dist = dist.normal.Normal(mu, sigma)
    # sum across all the dimensions except batch_size
    return torch.sum(dist.log_prob(x), dim=1)


def isotropic_gaussian_log_pdf(x):
    mu = torch.zeros_like(x)
    logvar = torch.zeros_like(x)
    return gaussian_log_pdf(x, mu, logvar)


def mixture_normal_log_pdf(x, mu, logvar, component_weights):
    std = torch.exp(0.5 * logvar)
    log_q_z_given_x = pyro.distributions.MixtureOfDiagNormals(
        locs=mu, coord_scale=std, component_logits=component_weights,
    ).log_prob(z)
    
    return torch.sum(log_q_z_given_x, dim=1)


def log_mean_exp(x, dim=1):
    r"""log(1/k * sum(exp(x)))"""
    m = torch.max(x, dim=dim, keepdim=True)[0]
    return m + torch.log(torch.mean(torch.exp(x - m),
                         dim=dim, keepdim=True))


class ImageEncoder(nn.Module):
    """
    Parameterizes a variational posterior distribution, q(z|x) 
    where x is an image.

    Args
    ----
    input_channels : integer
                     number of input image channels
    image_size     : integer
                     height and width of image
    z_dim          : integer
                     dimensionality of latent variable
    n_mixtures     : integer
                     number of mixture components
    n_filters      : integer
                     number of output convolutional filtures
    """

    def __init__(
        self, 
        input_channels: int, 
        image_size: int, 
        z_dim: int, 
        n_mixtures: int = 1,
        n_filters: int = 32, 
    ):

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, n_filters, 2, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters * 2, 2, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_filters * 2, n_filters * 4, 2, 2, padding=0),
        )

        cout = gen_32_conv_output_dim(image_size)

        # 2 bc we will extract mu and logvar using one network
        self.fc_latents = nn.Linear(n_filters * 4 * cout**2, z_dim * 2 * n_mixtures)

        if n_mixtures > 1:
            # we need a layer to spit out logits!
            self.fc_logits = nn.Linear(n_filters * 4 * cout**2, n_mixtures)

        self.cout = cout
        self.z_dim = z_dim
        self.n_mixtures = n_mixtures
        self.n_filters = n_filters

    def forward(self, image: torch.Tensor):
        batch_size = image.size(0)

        out = F.relu(self.conv(image))
        # flatten this object so that we can push it through the 
        # fully connected layer
        out = out.view(batch_size, self.n_filters * 4 * self.cout**2)

        z_params = self.fc_latents(out)

        if self.n_mixtures > 1:
            # important NOT to softmax here because we are expecting
            # logits (that is pre-softmax)
            logits = self.fc_logits(out)
        else:
            logits = None  # don't need this for n_mixtures = 1

        # these are the output parameters of a Gaussian distribution
        z_mu, z_logvar = torch.chunk(z_params, 2, dim=1)
        
        # reshape to (batch_size, n_mixtures, z_dim) 
        # so we explicitly store the mixture number
        # NOTE: this will be 1 if normal VAE
        z_mu = z_mu.view(batch_size, self.n_mixtures, self.z_dim)
        z_logvar = z_logvar.view(batch_size, self.n_mixtures, self.z_dim)

        return z_mu, z_logvar, logits


class ImageDecoder(nn.Module):
    """
    Parameterizes the density distribution over images, p(x|z)
    where z is a latent variable and x is an image.

    Args
    ----
    output_channels : integer
                      number of image channels in final output
    image_size      : integer
                      height/width of image
    z_dim           : integer
                      number of latent dimensions
    n_filters       : integer
                      number of filters in de-convolutional layers
    """

    def __init__(
        self, 
        output_channels: int, 
        image_size: int, 
        z_dim: int, 
        n_filters: int = 32,
    ):

        super().__init__()

        # deconvolutional layers: this will add pads to the matrix 
        # and slowly grow the image into its original size.
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(n_filters * 4, n_filters * 4, 2, 2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 2, 2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filters * 2, n_filters, 2, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_filters, output_channels, 1, 1, padding=0),
        )

        cout = gen_32_conv_output_dim(image_size)
        self.fc = nn.Linear(z_dim, n_filters * 4 * cout**2)

        self.cout = cout
        self.n_filters = n_filters
        self.output_channels = output_channels
        self.image_size = image_size

    def forward(self, z):
        batch_size = z.size(0)
        out = F.relu(self.fc(z))
        out = out.view(batch_size, self.n_filters * 4, self.cout, self.cout)
        out = self.conv(out)
        x_logits = out.view(batch_size, self.output_channels, 
                            self.image_size, self.image_size)
        x_mu = torch.sigmoid(x_logits)
        return x_mu


class MixtureVAE(nn.Module):
    """
    Variational Autoencoder with a mixture of Gaussians for the 
    posterior distribution. Optionally, one can specify a mixture
    of Gaussians for the prior as well.

    Args
    ----
    input_channels : integer
                     number of input image channels (3 for rgb)
    image_size     : integer
                     height and width of the image
    z_dim          : integer
                     number of latent dimensions
    n_mixtures     : integer
                     number of mixture components
    n_filters      : integer
                     number of filters to use in convolutional 
                     layers
    prior          : string
                     gaussian | mixture 
    """

    def __init__(
        self,
        input_channels: int,
        image_size: int,
        z_dim: int,
        n_mixtures: int,
        n_filters: int = 32,
        prior: str = 'gaussian',
    ):
        super().__init__()
        # we will assume a prior mixture of the same number of components
        assert prior in ['gaussian', 'mixture']

        self.z_dim = z_dim
        self.input_channels = input_channels
        self.image_size = image_size
        self.n_mixtures = n_mixtures
        self.n_filters = n_filters
        self.prior = prior

        self.prior_mu = None
        self.prior_logvar = None

        if self.prior == 'mixture':
            assert z_dim == 2, "we only support 2-dim latents"
            n_interval = get_num_interval(n_mixtures)
            prior_means = get_fixed_init(n_interval, 0, 1)

            self.prior_mu = torch.from_numpy(prior_inits).float()
            prior_sigma = torch.ones_like(self.prior_mu) * 0.1
            self.prior_logvar = 2 * torch.log(prior_sigma)
            self.prior_weights = torch.ones(self.n_mixtures) / self.n_mixtures

        self.encoder = ImageEncoder(
            input_channels=input_channels, image_size=image_size, z_dim=z_dim,
            n_filters=n_filters, n_mixtures=n_mixtures)
        
        self.decoder = ImageDecoder(
            output_channels=input_channels, image_size=image_size, z_dim=z_dim,
            n_filters=n_filters)

    def reparametrize(self, mu, logvar, logits):
        std = (0.5 * logvar).exp()  # shape: batch_size x n_mixtures x z_dim

        if self.n_mixtures == 1:
            # get rid of 1st dimension since n_mixtures = 1
            return dist.normal.Normal(loc=mu.squeeze(1), 
                                      scale=std.squeeze(1)).rsample()
        else:  # this means we need to reparameterize with MixtureOfDiagNormals
            batch_size = mu.size(0)
            return pyro.distributions.MixtureOfDiagNormals(
                locs=mu, coord_scale=std,
                component_logits=logits,
            ).rsample()

    def forward(self, x):
        z_mu, z_logvar, logits = self.encoder(x)
        z = self.reparametrize(z_mu, z_logvar, logits)
        x_mu = self.decoder(z)
        
        return x_mu, z, z_mu, z_logvar, logits

    def _kl_divergence(self, z, z_mu, z_logvar, logits):
        if self.n_mixtures == 1:
            if self.prior == 'gaussian':
                kl_div = kl_divergence_normal_and_spherical(z_mu, z_logvar)
            elif self.prior == 'mixture':
                prior_mu = self.prior_mu.to(x.device)
                prior_logvar = self.prior_logvar.to(x.device)
                prior_weights = self.prior_weights.to(x.device)
                kl_div = kl_divergence_normal_and_mixture(
                    z_mu, z_logvar, z, prior_mu, prior_logvar, prior_weights)
            else:
                raise Exception('prior {} not supported.'.format(self.prior))
        else:
            component_weights = F.softmax(logits, dim=1)
            if self.prior == 'gaussian':
                kl_div = kl_divergence_mixture_and_spherical(z_mu, z_logvar, z, component_weights)
            elif self.prior == 'mixture':
                prior_mu = self.prior_mu.to(x.device)
                prior_logvar = self.prior_logvar.to(x.device)
                prior_weights = self.prior_weights.to(x.device)
                kl_div = kl_divergence_mixture_and_mixture(
                    z_mu, z_logvar, z, component_weights, 
                    prior_mu, prior_logvar, prior_weights)
            else:
                raise Exception('prior {} not supported.'.format(self.prior))

        kl_div = torch.sum(kl_div, dim=1)
        return kl_div

    def elbo(self, x, x_mu, z, z_mu, z_logvar, logits):
        """
        Evidence lower bound objective on marginal log density.

        log p(x) > E_{q(z|x)}[log p(x,z) - log q(z|x)]
                 = E_{q(z|x)}[log p(x|z)] - KL(q(z|x)||p(z))
        """
        batch_size = x.size(0)
        log_p_x_given_z = bernoulli_log_pdf(x.view(batch_size, -1), 
                                            x_mu.view(batch_size, -1))
        kl_div = self._kl_divergence(z, z_mu, z_logvar, logits)
        elbo = log_p_x_given_z - kl_div
        elbo = torch.mean(elbo)
        
        # important to negate so that we have a positive loss
        return -elbo
    
    def log_likelihood(self, x, n_samples=100):
        """
        Importance weighted estimate of marginal log density.

        log p(x) ~  log { 1/K sum_{i=1}^K [exp{ log p(x,z_i) - log q(z_i|x)}] }
        
            where z_i ~ q(z|x) for i = 1 to 100
        """
        batch_size = x.size(0)
        z_mu, z_logvar, logits = self.encoder(x)

        log_w = []
        for i in range(n_samples):
            z_i = self.reparametrize(z_mu, z_logvar, logits)
            x_mu_i = self.decoder(z_i)
            log_p_x_given_z_i = bernoulli_log_pdf(x.view(batch_size, -1), 
                                                  x_mu_i.view(batch_size, -1))
            kl_div_i = self._kl_divergence(z_i, z_mu, z_logvar, logits)
            
            log_w_i = log_p_x_given_z_i - kl_div_i
            log_w_i = log_w_i.unsqueeze(1)
            log_w_i = log_w_i.cpu()
            log_w.append(log_w_i)
        log_w = torch.cat(log_w, dim=1)
        log_w = log_mean_exp(log_w, dim=1)

        return log_w
