'''
DCGAN architecture

Structure based on:
González-Duque, M., 2023. Minimal implementation of a Variational Autoencoder on Super Mario Bros (0.1).
Available from: https://github.com/miguelgondu/minimal_VAE_on_Mario [Accessed 04 April 2023]

DCGAN architecture inspired by:
Volz, V., Schrum, J., Liu, J., Lucas, S. M., Smith, A., & Risi, S.,  2018.
Evolving mario levels in the latent space of a deep convolutional generative adversarial network. In GECCO.
Aguirre, ed. Proceedings of the genetic and evolutionary computation conference, July 15 - 19, 2018, Kyoto Japan.
New York US: Association for Computing Machinery, pp. 221-228.
'''

from itertools import product

import numpy as np
import torch
from torch.distributions import Distribution, Normal, Categorical, kl_divergence
import torch.nn as nn
import os

from mario_utils.plotting import get_img_from_level

def load_data(training_percentage=0.8, shuffle_seed=0, device="cpu"):
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

    return training_tensors.to(device), test_tensors.to(device)

class Generator(nn.Module):
    def __init__(self, z_dim, n_sprites, hidden_dim, input_width, input_height):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.n_sprites = n_sprites
        self.input_width = input_width
        self.input_height = input_height

        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, hidden_dim * 4, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, n_sprites, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.Softmax(dim=1)
        )

    def forward(self, z):
        device = z.device
        z = z.view(-1, self.z_dim, 1, 1).to(device)
        fake_samples = self.net(z)
        #fake_samples = fake_samples.view(-1, self.height, self.width, self.n_sprites)
        return fake_samples


class Discriminator(nn.Module):
    def __init__(self, n_sprites, hidden_dim, input_width, input_height):
        super(Discriminator, self).__init__()
        self.n_sprites = n_sprites
        self.input_width = input_width
        self.input_height = input_height

        self.net = nn.Sequential(
            nn.Conv2d(n_sprites, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)

class DCGANMario(nn.Module):
    def __init__(
        self,
        width: int = 14,
        height: int = 14,
        z_dim: int = 2,
        n_sprites: int = 11,
        device: str = None,
        hidden_dim: int = 64,
    ):
        super(DCGANMario, self).__init__()
        self.width = width
        self.height = height
        self.n_sprites = n_sprites
        self.z_dim = z_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator(z_dim, n_sprites, hidden_dim, width, height).to(self.device)
        self.discriminator = Discriminator(n_sprites, hidden_dim, width, height).to(self.device)

        self.p_z = torch.distributions.Normal(
            torch.zeros(z_dim, device=self.device),
            torch.ones(z_dim, device=self.device),
        )

        self.train_data, self.test_data = load_data(device=self.device)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(self.device)
        fake_samples = self.generator(z)
        #fake_samples = logits.view(-1, self.height, self.width, self.n_sprites)
        return fake_samples

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        return self.p_z.sample((batch_size, self.z_dim)).to(self.device)

    def gen_loss(self, fake_samples: torch.Tensor) -> torch.Tensor:
        fake_scores = self.discriminator(fake_samples)

        fake_labels = torch.zeros_like(fake_scores)

        fake_loss = nn.functional.binary_cross_entropy_with_logits(fake_scores, fake_labels)

        return -fake_loss

    def dis_loss(self, real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
        real_scores = self.discriminator(real_samples)
        fake_scores = self.discriminator(fake_samples)

        real_labels = torch.ones_like(real_scores)
        fake_labels = torch.zeros_like(fake_scores)

        real_loss = nn.functional.binary_cross_entropy_with_logits(real_scores, real_labels)
        fake_loss = nn.functional.binary_cross_entropy_with_logits(fake_scores, fake_labels)

        return real_loss + fake_loss

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

        images_dist = self.decode(torch.from_numpy(zs).type(torch.float)).permute(0, 2, 3, 1)
        images_dist = Categorical(logits=images_dist)
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
