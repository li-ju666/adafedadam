import torch
from torch.utils.data import DataLoader
import importlib
from collections import OrderedDict
from utils.weights_utils import norm
from copy import deepcopy


class Client():
    def __init__(self, idx, setup, dataset, batch_size, device):
        # build the model
        get_model = importlib.import_module(f"model.{setup}").get_model
        self.model, self.optimizer = get_model(device)

        # split the dataset
        num_train_samples = int(len(dataset) * 0.8)
        num_test_samples = len(dataset) - num_train_samples
        trainset, testset = torch.utils.data.random_split(
            dataset, [num_train_samples, num_test_samples])

        self.traindl = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
        self.testdl = DataLoader(
            testset, batch_size=batch_size, shuffle=False, pin_memory=True)

        self.device = device
        self.size = len(trainset)
        self.idx = idx


    def train(self, epochs):
        self.model.train()

        for _ in range(epochs):
            for _, (x, y) in enumerate(self.traindl):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.model.loss(output, y)
                loss.backward()
                self.optimizer.step()

    def eval(self):
        self.model.eval()
        correct = 0
        total = 0
        accum_error = 0
        with torch.no_grad():
            for x, y in self.testdl:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                _, predicted = output.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                accum_error += self.model.loss(output, y).item()* y.size(0)

        return correct / total, accum_error / total, self.size

    def get_grad(self):
        self.model.train()

        # prepare a large batch of data (512)
        xs, ys = [], []
        sizes = 0
        for _, (x, y) in enumerate(self.traindl):
            xs.append(x)
            ys.append(y)
            sizes += y.shape[0]
            if sizes > min(self.size, 512):
                break
        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)
        xs, ys = xs.to(self.device), ys.to(self.device)

        self.model.zero_grad()
        output = self.model(xs)

        loss = self.model.loss(output, ys)
        loss.backward()  # calculate gradient

        gradients = OrderedDict()
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                gradients[k] = v.grad

        return loss.item(), gradients

    def get_weights(self):
        return deepcopy(self.model.state_dict())

    def set_weights(self, gweights):
        self.model.load_state_dict(gweights)
