'''
Training DCGAN

Structure based on:
González-Duque, M., 2023. Minimal implementation of a Variational Autoencoder on Super Mario Bros (0.1).
Available from: https://github.com/miguelgondu/minimal_VAE_on_Mario [Accessed 04 April 2023]

DCGAN architecture inspired by:
Volz, V., Schrum, J., Liu, J., Lucas, S. M., Smith, A., & Risi, S.,  2018.
Evolving mario levels in the latent space of a deep convolutional generative adversarial network. In GECCO.
Aguirre, ed. Proceedings of the genetic and evolutionary computation conference, July 15 - 19, 2018, Kyoto Japan.
New York US: Association for Computing Machinery, pp. 221-228.
'''
import pandas as pd

from dcgan import DCGANMario, load_data
import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader

def gan_fit(
        model: DCGANMario,
        generator_optimizer: Optimizer,
        discriminator_optimizer: Optimizer,
        data_loader: DataLoader,
        device: str,
        num_discriminator_updates: int = 5,
):
    model.train()
    running_loss = 0.0
    dis_loss_ls = []
    gen_loss_ls = []
    for levels in data_loader:
        levels = levels[0].to(device)
        batch_size = levels.size(0)

        for _ in range(num_discriminator_updates):
            # Training Discriminator
            real_samples = levels
            fake_samples = model.generator(model.sample_noise(batch_size))

            discriminator_optimizer.zero_grad()
            dis_loss = model.dis_loss(real_samples, fake_samples)
            #print('dis loss')
            #print(dis_loss)
            dis_loss_ls.append(dis_loss)
            dis_loss.backward()
            discriminator_optimizer.step()

        # Training Generator
        generator_optimizer.zero_grad()
        fake_samples = model.generator(model.sample_noise(batch_size))
        generator_loss = model.gen_loss(fake_samples)
        #print('gen loss')
        #print(generator_loss)
        gen_loss_ls.append(generator_loss)
        generator_loss.backward()
        generator_optimizer.step()

        running_loss += generator_loss.item()
        #print('running loss')
        #print(running_loss)

        data = []
        for i, (gen_loss, disc_loss) in enumerate(zip(gen_loss_ls, dis_loss_ls)):
            data.append({'Iteration': i + 1, 'Generator Loss': gen_loss, 'Discriminator Loss': disc_loss})

        df = pd.DataFrame(data)

    return running_loss / len(data_loader)

def dcgan_run(
        max_epochs: int = 2500,
        batch_size: int = 64,
        lr_dis: int = 2e-4,
        lr_gen: int = 1e-4,
        save_every: int = 50,
):
    # Defining the name of the experiment
    custom = 'custom_name'
    comment = f'{lr_gen}_{max_epochs}_{custom}_mario_dcgan'
    version = '_v'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the data.
    training_tensors, test_tensors = load_data()

    # Creating datasets.
    dataset = TensorDataset(training_tensors)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_tensors)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Loading the model and optimizer
    print("Model:")
    dcgan = DCGANMario()
    print(dcgan)

    generator_optimizer = optim.Adam(dcgan.generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(dcgan.discriminator.parameters(), lr=lr_dis, betas=(0.5, 0.999))

    # Training and testing.
    print(f"Training experiment {comment}")
    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1} of {max_epochs}.")
        _ = gan_fit(dcgan, generator_optimizer, discriminator_optimizer,
                    data_loader, device, num_discriminator_updates=5)

        if save_every is not None and epoch % save_every == 0 and epoch != 0:
            # Saving the model
            print(f"Saving the model at checkpoint {epoch}.")
            torch.save(dcgan.state_dict(), f"./trained_dcgan/{comment}_epoch_{epoch}{version}.pt")

            print(f"Saving the generator at checkpoint {epoch}.")
            torch.save(dcgan.generator.state_dict(),
                       f"./trained_dcgan/generator/generator_{comment}_epoch_{epoch}{version}.pt")

    # Always save the final model
    print("Saving the final model.")
    torch.save(dcgan.state_dict(), f"./trained_dcgan/gan{comment}_final{version}.pt")

    print("Saving generator model.")
    torch.save(dcgan.generator.state_dict(), f"./trained_dcgan/generator/generator_{comment}_final_{version}.pt")

if __name__ == "__main__":
    dcgan_run()
