# MIA Flower Base - Apprentissage fédéré simple + point d'entrée pour attaques/défenses

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# === Modèle de base (MLP simple pour MNIST) ===
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# === Chargement et partition des données ===
def load_data():
    transform = transforms.ToTensor()
    trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    testset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    return trainset, testset

def partition_data(trainset, cid, num_clients):
    total = len(trainset)
    split = total // num_clients
    start = cid * split
    end = (cid + 1) * split
    return Subset(trainset, list(range(start, end)))

# === Client Flower personnalisé ===
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_data, test_data):
        self.cid = cid
        self.model = model
        self.train_data = train_data
        self.test_data = test_data

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, val in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(val)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loader = DataLoader(self.train_data, batch_size=32, shuffle=True)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.model.train()
        for _ in range(1):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(self.model(x), y)
                loss.backward()
                optimizer.step()
        return self.get_parameters({}), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loader = DataLoader(self.test_data, batch_size=32)
        correct, total = 0, 0
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                pred = self.model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        accuracy = correct / total
        return float(1.0 - accuracy), len(self.test_data), {"accuracy": accuracy}

# === Fonction client_fn standard pour simulation ===
def client_fn(cid):
    cid = int(cid)
    trainset, testset = load_data()
    part = partition_data(trainset, cid, num_clients=5)
    return FlowerClient(cid, MLP(), part, testset)

# === Lancement simulation avec Flower ===
def main():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=5,
        min_available_clients=5
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )

if __name__ == "__main__":
    main()
