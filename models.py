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


def bernoulli_log_pdf(x, mu):
    mu = torch.clamp(mu, 1e-7, 1.-1e-7)
    return torch.sum(x * torch.log(mu) + (1. - x) * torch.log(1. - mu), dim=1)


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
        learned_mixture_distribution=None: bool,
        prior_is_fixed_mixtured_distribution=None: bool,
        prior_is_fixed_mixtured_distribution_parameters=[0.5, 0.5],
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

            self.logits = torch.nn.functional.softmax(self.input_to_logits(
                    image.view(
                        batch_size, self.input_channels * self.image_size * self.image_size)
                    ), dim=1)

            return pyro.distributions.MixtureOfDiagNormals(
                locs=mu.view(-1, self.mixture_size, self.z_dim),
                coord_scale=(std.view(-1, self.mixture_size, self.z_dim)),
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

    @staticmethod
    def kl_divergence_normal(mu, logvar, z):
        return -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    def kl_divergence_mixture(self, mu, logvar, z, logits):
        q_prob = None
        batch_size = mu.size(0)
        if self.fixed_mixtured_distribution:
            q_prob = - pyro.distributions.MixtureOfDiagNormals(
                locs=torch.full_like(batch_size, self.number_of_mixtures, self.latent_size, )
                coord_scale=torch.full_like(batch_size, self.number_of_mixtures, self.latent_size, )
                component_logits=torch.nn.functional.softmax(
                    torch.full_like(logits, 1. / self.number_of_mixtures)),
            ).log_prob(z)
        else:
            q_prob = - torch.sum(
                torch.distributions.normal.Normal(
                    loc=0,
                scale=1).log_prob(z), dim=1)
        std = (0.5 * logvar).exp()
        p_prob = - pyro.distributions.MixtureOfDiagNormals(
            locs=mu.view(-1, self.number_of_mixtures, self.latent_size),
            coord_scale=std.view(-1, self.number_of_mixtures, self.latent_size),
            component_logits=torch.nn.functional.softmax(logits, dim=1)
        ).log_prob(z)

        result = q_prob - p_prob
        return result

    def kl_divergence(self, mu, logvar, z, logits, number_of_mixtures,
                      latent_size, reparametrize_with):
        if self.reparametrize_with == 'normal':
            return kl_divergence_normal(mu, logvar, z)
        elif self.reparametrize_with == 'mixture_of_normal':
            return kl_divergence_mixture(mu, logvar, z, logits,
                                        self.number_of_mixtures,
                                        self.latent_size)

    def elbo(self, orig, z, recon, mu, logvar, logits, kl_weight=1):

        recon_loss = bernoulli_log_pdf(orig.flatten(), mu.flatten())
        kl_diverg = kl_divergence(
            mu,
            logvar,
            z,
            logits,
            self.number_of_mixtures,
            self.latent_size,
            reparametrize_with=self.reparametrize_with,
        )
        return torch.mean(
        recon_loss + (kl_diverg * kl_weight), dim=0)

