'''
Training GAN

Structure based on:
González-Duque, M., 2023. Minimal implementation of a Variational Autoencoder on Super Mario Bros (0.1).
Available from: https://github.com/miguelgondu/minimal_VAE_on_Mario [Accessed 04 April 2023]
'''

from gan import GANMario, load_data
import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader



def gan_fit(
        model: GANMario,
        generator_optimizer: Optimizer,
        discriminator_optimizer: Optimizer,
        data_loader: DataLoader,
        device: str,
        num_discriminator_updates: int = 5,
):
    model.train()
    running_loss = 0.0
    dis_loss_ls = []

    for levels in data_loader:
        levels = levels[0].to(device)
        batch_size = levels.size(0)

        for _ in range(num_discriminator_updates):
            # Training the Discriminator
            real_samples = levels

            # # For BCE
            # fake_samples = model.generator(model.sample_noise(batch_size))  # Generate fake samples
            #
            # #real_samples = levels.view(batch_size, -1)
            # #_, fake_samples = model(levels)
            #
            # discriminator_optimizer.zero_grad()
            # #dis_loss = model.dis_losss(real_samples, fake_samples)
            # #dis_loss.backward()
            # model.dis_loss(real_samples, fake_samples).backward()

            # For Wasserstein
            fake_samples = model.generator(model.sample_noise(batch_size)).detach()

            discriminator_optimizer.zero_grad()
            dis_loss = model.dis_loss(real_samples, fake_samples)
            dis_loss.backward()

            dis_loss_ls.append(dis_loss.item())

            # Apply gradient clipping
            for param in model.discriminator.parameters():
                param.grad.data.clamp_(-0.01, 0.01)

            discriminator_optimizer.step()


        # Training the Generator
        generator_optimizer.zero_grad()
        fake_samples = model.generator(model.sample_noise(batch_size))  # Generate fake samples
        generator_loss = model.gen_loss(fake_samples)
        generator_loss.backward()
        generator_optimizer.step()

        running_loss += generator_loss.item()

        #print(np.mean(dis_loss_ls))
    return running_loss / len(data_loader)

def gan_run(
        max_epochs: int = 1000,
        batch_size: int = 64,
        lr_dis: int = 2e-3,
        lr_gen: int = 1e-4,
        save_every: int = 20,
):
        comment = f"{lr_gen}_{max_epochs}_mariogan"
        version = "vb"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Loading data
        training_tensors, test_tensors = load_data()

        # Creating datasets
        dataset = TensorDataset(training_tensors)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        test_dataset = TensorDataset(test_tensors)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # Loading the model and optimizer
        gan = GANMario()

        #print("Model:")
        #print(gan)

        generator_optimizer = optim.Adam(gan.generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
        discriminator_optimizer = optim.Adam(gan.discriminator.parameters(), lr=lr_dis, betas=(0.5, 0.999))

        # Training and testing
        print(f"Training experiment {comment}")
        for epoch in range(max_epochs):
            print(f"Epoch {epoch + 1} of {max_epochs}.")

            gen_loss_per_epoch = gan_fit(gan, generator_optimizer, discriminator_optimizer, data_loader, device)

            print(gen_loss_per_epoch)

            if save_every is not None and epoch % save_every == 0 and epoch != 0:
                # Saving the model
                print(f"Saving the model at checkpoint {epoch}.")
                torch.save(gan.state_dict(), f"./trained_gan/gan{comment}_epoch_{epoch}_{version}.pt")

                print(f"Saving the generator at checkpoint {epoch}.")
                torch.save(gan.generator.state_dict(), f"./trained_gan/generator/generator{comment}_epoch_{epoch}_{version}.pt")
            # ...

        # Always save final model
        print("Saving the final model.")
        torch.save(gan.state_dict(), f"./trained_gan/gan{comment}_final_{version}.pt")

        print("Saving generator model.")
        torch.save(gan.generator.state_dict(), f"./trained_gan/generator/generator{comment}_{version}.pt")

if __name__ == "__main__":
    gan_run()

