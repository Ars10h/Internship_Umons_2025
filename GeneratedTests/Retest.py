# federated_base.py

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from flwr.common import parameters_to_ndarrays


# === Configuration ===
NUM_CLIENTS = 5
ROUNDS = 20
BATCH_SIZE = 32
LR = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Modèle simple mais performant ===
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# === Données ===
def load_data():
    transform = transforms.ToTensor()
    trainset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    testset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    return trainset, testset

def partition_data(trainset, cid, num_clients):
    split_size = len(trainset) // num_clients
    start = cid * split_size
    end = start + split_size
    return Subset(trainset, list(range(start, end)))

# === Client FL personnalisé ===
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_data, test_data):
        self.cid = cid
        self.model = model.to(DEVICE)
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
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=LR)
        loader = DataLoader(self.train_data, batch_size=BATCH_SIZE, shuffle=True)
        for epoch in range(1):  # Une seule époque pour simuler
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = F.cross_entropy(self.model(x), y)
                loss.backward()
                optimizer.step()
        return self.get_parameters({}), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=BATCH_SIZE)
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = self.model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        accuracy = correct / total
        return float(1.0 - accuracy), total, {"accuracy": accuracy}

# === client_fn obligatoire ===
def client_fn(cid: str):
    cid_int = int(cid)
    trainset, testset = load_data()
    train_partition = partition_data(trainset, cid_int, NUM_CLIENTS)
    model = MLP()
    return FederatedClient(cid_int, model, train_partition, testset).to_client()

# === Lancement simulation ===
def main():
    global_model = MLP().to(DEVICE)
    weights_list = []

    # Stratégie modifiée pour sauvegarder les poids à chaque round
    class TrackWeights(fl.server.strategy.FedAvg):
        def aggregate_fit(self, rnd, results, failures):
            aggregated_params, _ = super().aggregate_fit(rnd, results, failures)
            if aggregated_params is not None:
                weights_list.append(aggregated_params)
            return aggregated_params, {}

    strategy = TrackWeights(
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    # Simulation FL
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )

    # Recharger les poids du dernier modèle global
    last_weights = weights_list[-1]
    weights = parameters_to_ndarrays(last_weights)
    state_dict = {k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), weights)}
    global_model.load_state_dict(state_dict)

    # MIA : comparer perte membre vs. non-membre
    print("\n[MIA] Attaque par inférence d’appartenance...")
    trainset, testset = load_data()
    member_data = partition_data(trainset, cid=0, num_clients=NUM_CLIENTS)
    nonmember_data = Subset(testset, list(range(min(len(member_data), len(testset)))))


    loss_fn = nn.CrossEntropyLoss(reduction="none")
    global_model.eval()

    def compute_losses(data):
        loader = DataLoader(data, batch_size=len(data))
        x, y = next(iter(loader))
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            preds = global_model(x)
            losses = loss_fn(preds, y).cpu().numpy()
        return losses

    member_losses = compute_losses(member_data)
    nonmember_losses = compute_losses(nonmember_data)

    scores = -np.concatenate([member_losses, nonmember_losses])
    labels = np.concatenate([np.ones_like(member_losses), np.zeros_like(nonmember_losses)])

    fpr, tpr, _ = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)

    print(f"[MIA] AUC: {auc_score:.4f}")
    plt.plot(fpr, tpr, label=f"MIA ROC (AUC={auc_score:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC - Membership Inference Attack")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
