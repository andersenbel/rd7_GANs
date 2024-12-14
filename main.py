import torch
from data_loader import load_data
from generator import Generator
from discriminator import Discriminator
from training import train_gan, save_loss_plots, visualize_results


def main():
    # Параметри тренування
    batch_size = 128
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Завантаження даних
    dataloader = load_data(batch_size)

    # Ініціалізація моделей
    generator = Generator()
    discriminator = Discriminator()

    # Навчання GAN
    losses = train_gan(generator, discriminator,
                       dataloader, num_epochs, device)

    # Збереження моделі генератора
    torch.save(generator.state_dict(), "output/generator.pth")
    print("Модель генератора збережено у 'output/generator.pth'")

    # Збереження графіків втрат
    save_loss_plots(losses, output_dir="output")
    print("Графіки втрат збережено у папці 'output'.")


if __name__ == "__main__":
    main()
