from torch.utils.data import TensorDataset, Dataset
import numpy
import torch
import re  # ujson
import torch.nn.functional as F


def mnist(data):
    mean, std = 0.1307, 0.3081
    x = numpy.array(data['x'])/255.0
    # normalize x
    x = (x - mean) / std
    # convert mnist to tensor
    x = torch.tensor(x).type(torch.FloatTensor)

    y = numpy.array(data['y'])
    y = torch.tensor(y).type(torch.LongTensor)
    return TensorDataset(x, y)


def cifar10(data):
    # mean = numpy.array([0.4914, 0.4822, 0.4465])
    # std = numpy.array([0.2023, 0.1994, 0.2010])
    x = numpy.array(data['x'])/255.0

    # transpose x to fit pytorch format
    # x = x.transpose((0, 3, 1, 2))
    x = torch.tensor(x).type(torch.FloatTensor)

    y = numpy.array(data['y'])
    y = torch.tensor(y).type(torch.LongTensor)
    return TensorDataset(x, y)


class LocalDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = transform(x) if transform else x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y
