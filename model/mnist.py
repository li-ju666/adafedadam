import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.classifier(self.encoder(x))

    def loss(self, pred, target):
        return F.cross_entropy(pred, target)


def get_model(device):
    model = MLP().to(device)
    optimizer = optim.SGD(
        lr=0.01, params=model.parameters(),
        momentum=0.9, weight_decay=5e-4)
    return model, optimizer
