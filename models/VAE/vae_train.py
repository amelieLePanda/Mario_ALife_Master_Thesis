"""
Training VAE with early stopping

Structure based on:
González-Duque, M., 2023. Minimal implementation of a Variational Autoencoder on Super Mario Bros (0.1).
Available from: https://github.com/miguelgondu/minimal_VAE_on_Mario [Accessed 04 April 2023]
"""

import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from vae import VAEMario, load_data

def fit(
    model: VAEMario,
    optimizer: Optimizer,
    data_loader: DataLoader,
    device: str,
):
    model.train()
    running_loss = 0.0
    for (levels,) in data_loader:
        levels = levels.to(device)
        optimizer.zero_grad()
        q_z_given_x, p_x_given_z = model.forward(levels)
        loss = model.elbo_loss_function(levels, q_z_given_x, p_x_given_z)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    return running_loss / len(data_loader)


def test(
    model: VAEMario,
    test_loader: DataLoader,
    device: str,
    epoch: int = 0,
):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for (levels,) in test_loader:
            levels.to(device)
            q_z_given_x, p_x_given_z = model.forward(levels)
            loss = model.elbo_loss_function(levels, q_z_given_x, p_x_given_z)
            running_loss += loss.item()

    print(f"Epoch {epoch}. Loss in test: {running_loss / len(test_loader)}")
    return running_loss / len(test_loader)


def run(
        max_epochs: int = 5000,
        batch_size: int = 32,    # original 64
        lr: int =  1e-3,         # 1e-3 worked in general better than 1e-4
        save_every: int = 50,
        overfit: bool = False,
):
        #timestamp = str(time()).replace(".", "")
        #comment = f"{timestamp}_mario_vae"

        comment = f"{lr}_{batch_size}_mario_vae"
        version = "v_Student-t"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Currently used device: {device}")

        # Loading data
        training_tensors, test_tensors = load_data()

        # Creating datasets
        dataset = TensorDataset(training_tensors)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        test_dataset = TensorDataset(test_tensors)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # Loading model and optimizer
        vae = VAEMario()
        #print("Model:")
        #print(vae)

        optimizer = optim.Adam(vae.parameters(), lr=lr)

        # Training and testing
        print(f"Training experiment {comment}")
        best_loss = np.Inf
        n_without_improvement = 0
        for epoch in range(max_epochs):
            print(f"Epoch {epoch + 1} of {max_epochs}.")
            _ = fit(vae, optimizer, data_loader, device)
            test_loss = test(vae, test_loader, device, epoch)

            if test_loss < best_loss:
                best_loss = test_loss
                n_without_improvement = 0

                # Saving the best model so far
                torch.save(vae.state_dict(), f"./trained_vae/{comment}_final_{version}.pt")
            else:
                if not overfit:
                    n_without_improvement += 1

            if save_every is not None and (epoch + 1) % save_every == 0:
                # Saving the model
                print(f"Saving the model at checkpoint {epoch}.")
                torch.save(vae.state_dict(), f"./trained_vae/{comment}_epoch_{epoch}_{version}.pt")

            # Early stopping:
            if n_without_improvement == 150:
                print("Stopping early")
                break


if __name__ == "__main__":
    run()
