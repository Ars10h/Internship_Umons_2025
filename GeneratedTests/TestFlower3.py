# MIA Flower Base - Apprentissage fédéré + attaques/défenses intégrées

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# === Paramètres globaux ===
USE_DP = False
USE_CONFREG = False
USE_SECUREAGG = False
EPSILON = 0.1
CONF_WEIGHT = 0.5
SECURE_NOISE = 0.05
USE_ENTROPY_SCORE = False
USE_CONFIDENCE_SCORE = False

# === Modèle amélioré (ajout dropout pour régularisation) ===
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
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

# === Client Flower personnalisé avec défenses intégrables ===
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_data, test_data):
        self.cid = cid
        self.model = model
        self.train_data = train_data
        self.test_data = test_data

    def get_parameters(self, config):
        params = [val.cpu().numpy() for val in self.model.state_dict().values()]
        if USE_SECUREAGG:
            params = [p + np.random.normal(0, SECURE_NOISE, size=p.shape) for p in params]
        return params

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
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                if USE_CONFREG:
                    conf = torch.softmax(output, dim=1).max(dim=1)[0].mean()
                    loss += CONF_WEIGHT * conf
                loss.backward()
                if USE_DP:
                    for p in self.model.parameters():
                        p.grad += torch.randn_like(p.grad) * EPSILON
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

# === MIA avancée basée sur plusieurs scores ===
def mia_attack(model, member_data, nonmember_data):
    model.eval()
    with torch.no_grad():
        x1, y1 = next(iter(DataLoader(member_data, batch_size=len(member_data))))
        x2, y2 = next(iter(DataLoader(nonmember_data, batch_size=len(nonmember_data))))
        logits1 = model(x1)
        logits2 = model(x2)

        # Cross-entropy loss
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        l1 = loss_fn(logits1, y1).numpy()
        l2 = loss_fn(logits2, y2).numpy()

        # Confidence (max softmax)
        conf1 = torch.softmax(logits1, dim=1).max(dim=1)[0].numpy()
        conf2 = torch.softmax(logits2, dim=1).max(dim=1)[0].numpy()

        # Entropy
        def entropy(p):
            return -np.sum(p * np.log(p + 1e-10), axis=1)
        ent1 = entropy(torch.softmax(logits1, dim=1).numpy())
        ent2 = entropy(torch.softmax(logits2, dim=1).numpy())

        labels = np.concatenate([np.ones_like(l1), np.zeros_like(l2)])

        # Combine scores
        scores = -l1.tolist() + conf1.tolist() + (-ent1).tolist() + \
                 -l2.tolist() + conf2.tolist() + (-ent2).tolist()
        scores = np.array(scores)

        fpr, tpr, _ = roc_curve(np.concatenate([np.ones_like(l1), np.zeros_like(l2)]), scores)
        auc_score = auc(fpr, tpr)

        print(f"AUC (MIA): {auc_score:.4f}")
        plt.plot(fpr, tpr, label=f"MIA ROC (AUC={auc_score:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC - Membership Inference Attack")
        plt.legend()
        plt.grid(True)
        plt.show()

# === Fonction client_fn ===
def client_fn(cid):
    cid = int(cid)
    trainset, testset = load_data()
    part = partition_data(trainset, cid, num_clients=5)
    return FlowerClient(cid, MLP(), part, testset).to_client()

# === Simulation FL + MIA test ===
def main():
    global_weights = []  # stockage manuel des poids

    class SaveWeightsStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(self, rnd, results, failures):
            aggregated_params, _ = super().aggregate_fit(rnd, results, failures)
            if aggregated_params is not None:
                global_weights.append(aggregated_params)
            return aggregated_params, {}

    strategy = SaveWeightsStrategy(
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

    # Récupération du modèle global après simulation
    global_model = MLP()
    weights = fl.common.parameters_to_ndarrays(global_weights[-1])
    state_dict = {k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), weights)}
    global_model.load_state_dict(state_dict)

    # Évaluation MIA avec vrais membres vs. non-membres
    trainset, testset = load_data()
    member_data = partition_data(trainset, cid=0, num_clients=5)  # utilisé pendant le FL
    nonmember_data = Subset(testset, list(range(64)))             # jamais vu
    mia_attack(global_model, member_data, nonmember_data)

if __name__ == "__main__":
    main()
