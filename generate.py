import torch
import matplotlib.pyplot as plt
from generator import Generator


def generate_images(model_path="output/generator.pth", output_path="output/generated_images_test.png", num_images=16):
    # Завантажуємо модель генератора
    generator = Generator()
    generator.load_state_dict(torch.load(model_path))
    generator.eval()

    # Генеруємо випадковий шум
    z = torch.randn(num_images, 100)
    with torch.no_grad():
        fake_images = generator(z).cpu().detach().numpy()

    # Перетворення зображень у діапазон [0, 1]
    fake_images = (fake_images + 1) / 2

    # Візуалізуємо згенеровані зображення
    plt.figure(figsize=(8, 8))
    for i in range(num_images):
        plt.subplot(4, 4, i + 1)
        plt.imshow(fake_images[i].squeeze(), cmap="gray")
        plt.axis("off")
    plt.savefig(output_path)
    print(f"Згенеровані зображення збережено у {output_path}")


if __name__ == "__main__":
    generate_images()
