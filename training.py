import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os


def get_optimizers(generator, discriminator, lr=0.0002):
    """Ініціалізуємо оптимізатори для генератора і дискримінатора."""
    optim_g = optim.Adam(generator.parameters(), lr=lr)
    optim_d = optim.Adam(discriminator.parameters(), lr=lr)
    return optim_g, optim_d


def train_gan(generator, discriminator, dataloader, num_epochs=50, device="cuda"):
    """
    Функція для навчання GAN. Навчає дискримінатор і генератор, 
    збирає втрати для подальшого аналізу.
    """
    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCELoss()
    optim_g, optim_d = get_optimizers(generator, discriminator)

    losses = {"D": [], "G": []}

    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # ---- Навчання дискримінатора ----
            discriminator.zero_grad()
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            outputs_real = discriminator(real_images)
            loss_real = criterion(outputs_real, real_labels)

            z = torch.randn(batch_size, 100).to(device)
            fake_images = generator(z)
            outputs_fake = discriminator(fake_images.detach())
            loss_fake = criterion(outputs_fake, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            optim_d.step()

            # ---- Навчання генератора ----
            generator.zero_grad()
            z = torch.randn(batch_size, 100).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images)
            loss_g = criterion(outputs, real_labels)
            loss_g.backward()
            optim_g.step()

        # Логування втрат
        losses["D"].append(loss_d.item())
        losses["G"].append(loss_g.item())

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}"
        )

    return losses


def visualize_results(generator, device, num_images=16, output_dir="output"):
    """
    Генерує і зберігає зображення для оцінки прогресу навчання генератора.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generator.eval()
    z = torch.randn(num_images, 100).to(device)
    fake_images = generator(z).cpu().detach().numpy()
    fake_images = (fake_images + 1) / 2  # Перетворення до [0, 1]

    plt.figure(figsize=(8, 8))
    for i in range(num_images):
        plt.subplot(4, 4, i + 1)
        plt.imshow(fake_images[i].squeeze(), cmap="gray")
        plt.axis("off")
    plt.savefig(os.path.join(output_dir, "generated_images.png"))
    plt.close()
    generator.train()


def save_loss_plots(losses, output_dir="output"):
    """
    Зберігає графіки втрат генератора і дискримінатора в PNG-файл.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 5))
    plt.plot(losses["D"], label="Discriminator Loss")
    plt.plot(losses["G"], label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GAN Training Losses")
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()
