from torchvision import datasets
import numpy as np


# return protocol:
# return trainx, trainy in numpy arrays or ndarrays with no preprocessing

# prepare preprocessed mnist dataset
def mnist():
    # download mnist dataset
    trainset = datasets.MNIST('data/cache', train=True, download=True)
    trainx = trainset.data.numpy().astype(np.uint8)
    trainy = trainset.targets.numpy().astype(np.uint8)

    return trainx, trainy


# prepare preprocessed cifar10 dataset
def cifar10():
    # download cifar10 dataset
    trainset = datasets.CIFAR10('data/cache', train=True, download=True)

    trainx = trainset.data.astype(np.uint8)
    # trainspose the data to fit the shape of pytorch
    trainx = np.transpose(trainx, (0, 3, 1, 2))

    trainy = np.array(trainset.targets).astype(np.uint8)

    return trainx, trainy
