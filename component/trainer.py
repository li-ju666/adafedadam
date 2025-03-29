from torch.optim import SGD, Adam
from component.optim import AdaAdam
from utils.weights_utils import weights_substraction
from collections import OrderedDict
from utils.weights_utils import norm, norm_l2
from math import log, isnan
import numpy
import importlib
import torch
import random
from math import ceil


class Trainer():
    def __init__(self, setup):
        self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'
        # build the model
        get_model = importlib.import_module(f"model.{setup}").get_model
        self.model, _ = get_model(self.device)
        self.optimizer = None

    def train(self, clients, participation, epochs):
        # select clients
        selected_clients = random.sample(clients, ceil(
            participation * len(clients)))

        # make local updates
        gweight = self.model.state_dict()

        local_updates = {}
        for client in selected_clients:
            local_return= self._train_client(client, gweight, epochs)
            local_updates[client.idx] = local_return # (update, weight, sup_info)

        # get the pseudo gradient
        pseudo_grad = self.get_pseudo_grad(local_updates)

        # update the global model
        self.update_gmodel(pseudo_grad)


    def eval(self, clients):
        # return avg_acc + all accs
        gweight = self.model.state_dict()

        accs = []
        errors = []
        weights = []

        for client in clients:
            client.set_weights(gweight)
            acc, error, weight = client.eval()
            accs.append(acc)
            errors.append(error)
            weights.append(weight)

        accs = numpy.array(accs)
        errors = numpy.array(errors)
        weights = numpy.array(weights)

        # compute the weighted average
        weights = weights / sum(weights)
        avg_acc = numpy.dot(weights, accs)
        avg_error = numpy.dot(weights, errors)
        # compute the std
        std_acc = accs.std()
        std_error = errors.std()

        return {
            'avg_acc': avg_acc,
            'avg_error': avg_error,
            'std_acc': std_acc,
            'std_error': std_error
        }

    def _train_client(self, client, gweight, epochs):
        client.set_weights(gweight)
        client.train(epochs)
        avg_weight = client.size
        local_update = weights_substraction(
            gweight, client.get_weights())
        return local_update, avg_weight, None

    @staticmethod
    def get_pseudo_grad(local_updates):
        total_weight = sum(
            [weight for _, (_, weight, _) in local_updates.items()])

        pseudo_grad = OrderedDict()
        for _, (update, weight, _) in local_updates.items():
            for key in update:
                if key not in pseudo_grad:
                    pseudo_grad[key] = weight/total_weight * update[key]
                else:
                    pseudo_grad[key] += weight/total_weight * update[key]

        return pseudo_grad

    def update_gmodel(self, pseudo_grad):
        self.optimizer.zero_grad()
        for k, w in self.model.named_parameters():
            if w.requires_grad:
                w.grad = pseudo_grad[k]
        self.optimizer.step()


class FedAvg(Trainer):
    def __init__(self, setup):
        super().__init__(setup)
        self.optimizer = SGD(
            lr=0.1, params=self.model.parameters())


class FedAdam(Trainer):
    def __init__(self, setup):
        super().__init__(setup)
        self.optimizer = Adam(
            lr=1e-3, params=self.model.parameters())


class AdaFedAdam(Trainer):
    def __init__(self, setup):
        super().__init__(setup)
        self.optimizer = AdaAdam(
            lr=1e-3, params=self.model.parameters())

    def train(self, clients, participation, epochs):
        # select clients
        selected_clients = random.sample(clients, ceil(
            participation * len(clients)))

        # make local updates
        gweight = self.model.state_dict()

        local_updates = {}
        for client in selected_clients:
            local_return= self._train_client(client, gweight, epochs)
            local_updates[client.idx] = local_return # (update, weight, certainty)

        # aggregate the certainity
        certainty = 0
        total_weight = 0
        for _, (_, weight, cert) in local_updates.items():
            total_weight += weight
            certainty += cert * weight
        certainty /= total_weight

        # get the pseudo gradient
        pseudo_grad = self.get_pseudo_grad(local_updates)

        # update the global model
        self.update_gmodel(pseudo_grad, certainty)

    def _train_client(self, client, gweight, epochs):
        client.set_weights(gweight)

        current_loss, grad = client.get_grad()
        client.train(epochs)

        update = weights_substraction(gweight, client.get_weights())

        # normalize the update
        norm_factor = norm(update) / norm(grad)
        for k in update:
            update[k] = update[k]/norm_factor

        # compute the certainty
        local_lr = client.optimizer.param_groups[0]['lr']
        certainty = log(norm_factor/local_lr) + 1

        aggregate_weight = client.size * current_loss

        return update, aggregate_weight, certainty

    def update_gmodel(self, pseudo_grad, certainty):
        self.optimizer.zero_grad()

        for k, w in self.model.named_parameters():
            if w.requires_grad:
                w.grad = pseudo_grad[k]
        # print(f"Certainty: {certainty}")
        self.optimizer.set_confidence(certainty)
        self.optimizer.step()

