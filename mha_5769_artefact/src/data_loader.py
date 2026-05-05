import torch
import torchvision
import torchvision.transforms as transforms
import os
from .config import DATA_DIR, BATCH_SIZE

def get_data_loaders(batch_size=BATCH_SIZE, data_dir=DATA_DIR):
    """
    Downloads and prepares MNIST data loaders.
    To make it compatible with our 3-channel 32x32 CNN model, we resize it
    and convert the 1-channel grayscale to 3-channels.
    """
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3), # Convert 1-channel to 3-channels
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # No augmentation for testing
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    print(f"Downloading/loading MNIST data to {data_dir}...")
    trainset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform_train)
    
    num_workers = 0 if os.name == 'nt' else 2

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    return trainloader, testloader, classes
