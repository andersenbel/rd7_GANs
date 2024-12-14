import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Нормалізуємо до [-1, 1]
    ])

    dataset = datasets.MNIST(root="./data", train=True,
                             download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
