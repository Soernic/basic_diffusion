import torch
from torchvision import datasets, transforms


def MNIST():

    # Define a transform to normalize the data (convert to tensor and normalize)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Download and load the MNIST dataset
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    return trainloader, testloader


if __name__ == '__main__':
    trainloader, testloader = MNIST()
