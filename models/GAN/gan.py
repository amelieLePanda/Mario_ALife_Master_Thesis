'''
GAN architecture

Structure based on:
González-Duque, M., 2023. Minimal implementation of a Variational Autoencoder on Super Mario Bros (0.1).
Available from: https://github.com/miguelgondu/minimal_VAE_on_Mario [Accessed 04 April 2023]
'''
import os
import torch.nn.functional as F
from itertools import product
from typing import List

import numpy as np
import torch
from torch.distributions import Distribution, Normal, Categorical, kl_divergence
import torch.nn as nn

from mario_utils.plotting import get_img_from_level

def load_data(training_percentage=0.8, shuffle_seed=0, device="gpu"):
    """Returns two tensors with training and testing data"""
    # This data is structured [b, c, i, j], where c corresponds to the class.

    data_path = os.path.join(os.path.dirname(__file__), "all_levels_onehot.npz")
    data = np.load(data_path)["levels"]

    #data = np.load("../../data/all_levels_onehot.npz")["levels"]
    np.random.seed(shuffle_seed)
    np.random.shuffle(data)

    # Separating into training and test.
    n_data, _, _, _ = data.shape
    training_index = int(n_data * training_percentage)
    training_data = data[:training_index, :, :, :]
    testing_data = data[training_index:, :, :, :]

    # Data normalization
    training_data = training_data / 10.0  # Normalize to range [-1, 1]

    training_tensors = torch.from_numpy(training_data).type(torch.float)
    test_tensors = torch.from_numpy(testing_data).type(torch.float)

    return training_tensors, test_tensors

class GANMario(nn.Module):
    def __init__(
        self,
        width: int = 14,
        height: int = 14,
        z_dim: int = 2,
        n_sprites: int = 11,
        device: str = None,
        hidden_dim: int = 256, #128
    ):
        super(GANMario, self).__init__()
        self.width = width
        self.height = height
        self.n_sprites = n_sprites
        self.input_dim = width * height * n_sprites  # for flattening
        self.z_dim = z_dim
        self.device = device

        self.generator = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, self.input_dim),
        ).to(device)

        self.discriminator = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            #nn.Sigmoid(),  # Output a single value in the range [0, 1] -- uncomment for BCE loss
        ).to(device)

        self.p_z = Normal(
            torch.zeros(self.z_dim, device=self.device),
            torch.ones(self.z_dim, device=self.device),
        )

        self.train_data, self.test_data = load_data(device=self.device)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Returns the generated samples (fake pixel values) as a 4D tensor
        logits = self.generator(z)
        fake_samples = logits.view(-1, self.height, self.width, self.n_sprites)
        return fake_samples

    # def forward(self, x: t.Tensor) -> List[Distribution]:
    #     # input x is bchw (batch, channel, height, width)
    #     b, _, _, _ = x.shape
    #     z = t.randn(b, self.z_dim, device=self.device)
    #     p_x_given_z = self.decode(z)
    #     return [None, p_x_given_z]

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        return self.p_z.sample((batch_size, self.z_dim)).to(self.device)

    # # BCE Loss for Generator
    # def gen_loss(self, fake_samples: torch.Tensor) -> torch.Tensor:
    #     fake_scores = self.discriminator(fake_samples)
    #
    #     fake_labels = torch.zeros_like(fake_scores)
    #
    #     fake_loss = nn.functional.binary_cross_entropy_with_logits(fake_scores, fake_labels)
    #
    #     return -fake_loss

    # Wasserstein Loss for Generator
    def gen_loss(self, fake_samples: torch.Tensor) -> torch.Tensor:
        fake_scores = self.discriminator(fake_samples)
        return -torch.mean(fake_scores)

    # def gan_loss(self, real_samples: t.Tensor, fake_samples: t.Tensor) -> t.Tensor:
    #     real_scores = self.discriminator(real_samples.view(-1, self.input_dim))
    #     fake_scores = self.discriminator(fake_samples.view(-1, self.input_dim))
    #
    #     real_labels = t.ones_like(real_scores)
    #     fake_labels = t.zeros_like(fake_scores)
    #
    #     real_loss = F.binary_cross_entropy(real_scores, real_labels)
    #     fake_loss = F.binary_cross_entropy(fake_scores, fake_labels)
    #
    #     return real_loss + fake_loss

    # Binary Cross-Entropy (BCE) Loss with Logits
    # def dis_loss(self, real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
    #     real_scores = self.discriminator(real_samples.view(-1, self.input_dim))
    #     fake_scores = self.discriminator(fake_samples.view(-1, self.input_dim))
    #
    #     real_labels = torch.ones_like(real_scores)
    #     fake_labels = torch.zeros_like(fake_scores)
    #
    #     real_loss = nn.functional.binary_cross_entropy_with_logits(real_scores, real_labels)
    #     fake_loss = nn.functional.binary_cross_entropy_with_logits(fake_scores, fake_labels)
    #
    #     return real_loss + fake_loss

    # Wasserstein loss
    def dis_loss(self, real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
        real_scores = self.discriminator(real_samples.view(-1, self.input_dim))
        fake_scores = self.discriminator(fake_samples.view(-1, self.input_dim))
        dis_loss = -(torch.mean(real_scores) - torch.mean(fake_scores))
        return dis_loss

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
        images_dist = Categorical(logits=images_dist) # to be able to visualize
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
