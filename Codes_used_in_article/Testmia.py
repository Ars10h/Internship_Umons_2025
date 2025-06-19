import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from flwr.common import parameters_to_ndarrays
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# === Configuration ===
NUM_CLIENTS = 5
ROUNDS = 20
BATCH_SIZE = 64
LR = 0.01
EPOCHS = 5  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CNN CIFAR-10 ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# === Data CIFAR-10 ===
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    return trainset, testset

def partition_data(trainset, cid, num_clients):
    split_size = len(trainset) // num_clients
    start = cid * split_size
    end = start + split_size
    return Subset(trainset, list(range(start, end)))

# === Client ===
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
        for _ in range(EPOCHS):  # Plus d'epochs
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
        return float(1 - accuracy), total, {"accuracy": accuracy}

# === client_fn ===
def client_fn(cid: str):
    cid_int = int(cid)
    trainset, testset = load_data()
    train_partition = partition_data(trainset, cid_int, NUM_CLIENTS)
    model = SimpleCNN()
    return FederatedClient(cid_int, model, train_partition, testset).to_client()

# === Simulation ===
def main():
    global_model = SimpleCNN().to(DEVICE)
    weights_list = []

    class SaveWeights(fl.server.strategy.FedAvg):
        def aggregate_fit(self, rnd, results, failures):
            aggregated, _ = super().aggregate_fit(rnd, results, failures)
            if aggregated:
                weights_list.append(aggregated)
            return aggregated, {}

    strategy = SaveWeights(
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )

    final_params = parameters_to_ndarrays(weights_list[-1])
    state_dict = {k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), final_params)}
    global_model.load_state_dict(state_dict)

    trainset, testset = load_data()
    member_data = partition_data(trainset, cid=0, num_clients=NUM_CLIENTS)
    nonmember_data = Subset(testset, list(range(len(member_data))))

    print("\n[MIA-Supervised] Attaque supervisée avancée...")

    def get_features(data):
        loader = DataLoader(data, batch_size=64)
        features = []
        labels = []
        global_model.eval()
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = global_model(x)
                probs = torch.softmax(logits, dim=1)
                max_conf = torch.max(probs, dim=1).values
                loss = loss_fn(logits, y)
                combined = torch.stack([loss.cpu(), max_conf.cpu()], dim=1)
                features.extend(combined.numpy())
                labels.extend(y.cpu().numpy())
        return np.array(features)

    member_features = get_features(member_data)
    nonmember_features = get_features(nonmember_data)

    X = np.concatenate([member_features, nonmember_features])
    y = np.concatenate([np.ones(len(member_features)), np.zeros(len(nonmember_features))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    attack_model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
    attack_model.fit(X_train, y_train)

    y_scores = attack_model.predict_proba(X_test)[:, 1]
    y_pred = attack_model.predict(X_test)
    auc_mia = roc_auc_score(y_test, y_scores)

    print(f"[MIA-Supervised] AUC de l’attaque : {auc_mia:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Non-membre", "Membre"]))

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.plot(fpr, tpr, label=f"MIA ROC (AUC={auc_mia:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC - Membership Inference Attack (Supervised Advanced)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
