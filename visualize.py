"""
Loads trained models and visualizes the latent space of those.

Based on and adapted from:
González-Duque, M., 2023. Minimal implementation of a Variational Autoencoder on Super Mario Bros (0.1).
Available from: https://github.com/miguelgondu/minimal_VAE_on_Mario [Accessed 04 April 2023]
"""
import torch
import matplotlib.pyplot as plt
import os

from models.DCGAN.dcgan import DCGANMario
from models.GAN.gan import GANMario
from models.VAE.vae import load_data, VAEMario

def visualize_VAE(model_name):
    # Loading VAE and its weights
    model_file_path = os.path.join(os.path.dirname(__file__), "models", "VAE", "trained_vae")
    model = VAEMario()

    model.load_state_dict(torch.load(f"{model_file_path}/{model_name}"))

    # Plotting the encodings
    training_data, test_data = load_data()

    training_encodings = model.encode(training_data).mean.detach().cpu().numpy()
    #training_encodings_std = model.encode(training_data).stddev.detach().cpu().numpy()
    test_encodings = model.encode(test_data).mean.detach().cpu().numpy()

    # Visualizing the latent space
    _, (ax_grid, ax_latent_codes) = plt.subplots(1, 2, figsize=(7 * 2, 7))

    ax_latent_codes.scatter(training_encodings[:, 0], training_encodings[:, 1])
    ax_latent_codes.scatter(test_encodings[:, 0], test_encodings[:, 1])

    print(training_encodings[:, 0].mean(), training_encodings[:, 0].std())
    print(training_encodings[:, 1].mean(), training_encodings[:, 1].std())

    # Plotting a grid of levels by decoding a grid in latent space
    # and placing the level in the center
    model.plot_grid(ax=ax_grid)

    # Save the figure with the model_name as the filename
    output_folder = "output_images_latent_representation/vae"
    output_file_path = os.path.join(output_folder, f"{os.path.splitext(model_name)[0]}.png")
    plt.savefig(output_file_path)

    # Showing
    plt.tight_layout()
    plt.show()
    plt.close()

def visualize_GAN(model_name):
    # Loading GAN and its weights
    model_file_path = os.path.join(os.path.dirname(__file__), "models", "GAN", "trained_gan", "generator")
    model = GANMario()

    model.generator.load_state_dict(torch.load(f"{model_file_path}/{model_name}"))

    # Visualizing the latent space
    fig, ax_grid = plt.subplots(figsize=(7, 7))

    # Plotting a grid of levels by decoding a grid in latent space
    # and placing the level in the center
    model.plot_grid(ax=ax_grid)

    # Save the figure with the model_name as the filename
    output_folder = "output_images_latent_representation/dcgan"
    output_file_path = os.path.join(output_folder, f"{os.path.splitext(model_name)[0]}.png")
    plt.savefig(output_file_path)

    # Showing
    plt.tight_layout()
    plt.show()
    plt.close()

def visualize_DCGAN(model_name):
    # Loading DCGAN and its weights
    model_file_path = os.path.join(os.path.dirname(__file__), "models", "DCGAN", "trained_dcgan", "generator")
    model = DCGANMario()

    model.generator.load_state_dict(torch.load(f"{model_file_path}/{model_name}"))

    # Visualizing the latent space
    fig, ax_grid = plt.subplots(figsize=(7, 7))

    # Plotting a grid of levels by decoding a grid in latent space
    # and placing the level in the center
    model.plot_grid(ax=ax_grid)

    # Save the figure with the model_name as the filename
    output_folder = "output_images_latent_representation/dcgan"
    output_file_path = os.path.join(output_folder, f"{os.path.splitext(model_name)[0]}.png")
    plt.savefig(output_file_path)

    # Showing
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    # Visualize models
    #model = visualize_GAN('generator0.0002_20000_mariogan_epoch_8500.pt')
    #model = visualize_DCGAN('generator_0.0001_2500_5to1_mario_dcgan_epoch_1400.pt')
    model = visualize_VAE('0.0001_64_mario_vae_epoch_499_v3_StudenT.pt')
