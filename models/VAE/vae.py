"""
A categorical VAE that can train on Mario.

Based on and adapted from:
González-Duque, M., 2023. Minimal implementation of a Variational Autoencoder on Super Mario Bros (0.1).
Available from: https://github.com/miguelgondu/minimal_VAE_on_Mario [Accessed 04 April 2023]

Student-t approach inspired by:
Mathieu, E., Rainforth, T., Siddharth, N., & Teh, Y. W., 2019.
Disentangling disentanglement in variational autoencoders.
In International conference on machine learning (pp. 4402-4412). PMLR.
"""

from itertools import product
from typing import List

import numpy as np
import torch
from torch.distributions import Distribution, Normal, Categorical, kl_divergence, StudentT
import torch.nn as nn
import os

from mario_utils.plotting import get_img_from_level


def load_data(training_percentage=0.8, shuffle_seed=0, device="gpu"):
    """Returns two tensors with training and testing data"""
    # This data is structured [b, c, i, j], where c corresponds to the class.

    data_path = os.path.join(os.path.dirname(__file__), "all_levels_onehot.npz")
    data = np.load(data_path)["levels"]

    np.random.seed(shuffle_seed)
    np.random.shuffle(data)

    # Separating into training and test.
    n_data, _, _, _ = data.shape
    training_index = int(n_data * training_percentage)
    training_data = data[:training_index, :, :, :]
    testing_data = data[training_index:, :, :, :]
    training_tensors = torch.from_numpy(training_data).type(torch.float)
    test_tensors = torch.from_numpy(testing_data).type(torch.float)

    return training_tensors, test_tensors


class VAEMario(nn.Module):
    def __init__(
        self,
        w: int = 14,
        h: int = 14,
        z_dim: int = 2,
        n_sprites: int = 11,
        device: str = None,
    ):
        super(VAEMario, self).__init__()
        self.w = w
        self.h = h
        self.n_sprites = n_sprites
        self.input_dim = w * h * n_sprites  # for flattening
        self.z_dim = z_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        ).to(self.device)
        self.enc_mu = nn.Sequential(nn.Linear(128, z_dim)).to(self.device)
        self.enc_var = nn.Sequential(nn.Linear(128, z_dim)).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, self.input_dim),
        ).to(self.device)

        # The VAE prior on latent codes. Only used for the KL term in
        # the ELBO loss.
        # Normal Dist or StudenT (Prior for Axis-Aligned Disentanglement: https://arxiv.org/pdf/1812.02833.pdf

        # self.p_z = Normal(
        #     torch.zeros(self.z_dim, device=self.device),
        #     torch.ones(self.z_dim, device=self.device),
        # )

        self.p_z = StudentT(
            6,
            torch.zeros(self.z_dim, device=self.device),
            torch.ones(self.z_dim, device=self.device),
        )

        self.train_data, self.test_data = load_data(device=self.device)

        # print(self)

    def encode(self, x: torch.Tensor) -> Normal:
        # Returns q(z | x) = Normal(mu, sigma)
        x = x.view(-1, self.input_dim).to(self.device)
        result = self.encoder(x)
        mu = self.enc_mu(result)
        log_var = self.enc_var(result)

        return Normal(mu, torch.exp(0.5 * log_var))

    # def decode(self, z: t.Tensor) -> Categorical:
    #     # Returns p(x | z) = Cat(logits=what the decoder network says)
    #     logits = self.decoder(z)
    #     p_x_given_z = Categorical(
    #         logits=logits.reshape(-1, self.h, self.w, self.n_sprites)
    #     )
    #
    #     return p_x_given_z

    def decode(self, z: torch.Tensor) -> Categorical:

        z = z.to(self.device)
        self.decoder.to(z.device)

        # Returns p(x | z) = Cat(logits=what the decoder network says)
        logits = self.decoder(z)
        p_x_given_z = Categorical(
            logits=logits.reshape(-1, self.h, self.w, self.n_sprites)
        )

        return p_x_given_z

    # def decode(self, z: t.Tensor) -> Categorical:
    #     # Returns p(x | z) = Cat(logits=what the decoder network says)
    #     logits = self.decoder(z)
    #     p_x_given_z = Categorical(logits=logits)
    #     return p_x_given_z

    def forward(self, x: torch.Tensor) -> List[Distribution]:
        q_z_given_x = self.encode(x.to(self.device))

        z = q_z_given_x.rsample()

        p_x_given_z = self.decode(z.to(self.device))

        return [q_z_given_x, p_x_given_z]

    def elbo_loss_function(
        self, x: torch.Tensor, q_z_given_x: Distribution, p_x_given_z: Distribution
    ) -> torch.Tensor:
        x_ = x.to(self.device).argmax(dim=1)  # assuming x is bchw.
        rec_loss = -p_x_given_z.log_prob(x_).sum(dim=(1, 2))  # b
        if type(self.p_z) == Normal:
            kld = kl_divergence(q_z_given_x, self.p_z).sum(dim=1)
        else:
            z = q_z_given_x.rsample() # StudenT Distribution! Sample some points to calculate a probability
            kld = (q_z_given_x.log_prob(z) - self.p_z.log_prob(z)).sum(dim=1)

        return (rec_loss + kld).mean()

    def plot_grid(
        self,
        x_lims=(-5, 5),
        y_lims=(-5, 5),
        n_rows=10,
        n_cols=10,
        sample=False,
        ax=None,
    ):
        z1 = np.linspace(*x_lims, n_cols)
        z2 = np.linspace(*y_lims, n_rows)

        zs = np.array([[a, b] for a, b in product(z1, z2)])

        images_dist = self.decode(torch.from_numpy(zs).type(torch.float))
        if sample:
            images = images_dist.sample()
        else:
            images = images_dist.probs.argmax(dim=-1)

        images = np.array(
            [get_img_from_level(im) for im in images.cpu().detach().numpy()]
        )
        img_dict = {(z[0], z[1]): img for z, img in zip(zs, images)}

        positions = {
            (x, y): (i, j) for j, x in enumerate(z1) for i, y in enumerate(reversed(z2))
        }

        pixels = 16 * 14
        final_img = np.zeros((n_cols * pixels, n_rows * pixels, 3))
        for z, (i, j) in positions.items():
            final_img[
                i * pixels : (i + 1) * pixels, j * pixels : (j + 1) * pixels
            ] = img_dict[z]

        final_img = final_img.astype(int)
        print(final_img)

        if ax is not None:
            ax.imshow(final_img, extent=[*x_lims, *y_lims])

        return final_img
