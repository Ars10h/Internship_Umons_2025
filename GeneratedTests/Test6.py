# Projet : Apprentissage Fédéré + Attaque MIA + Défenses Multiples + Visualisation

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# === Paramètres Généraux ===
NUM_CLIENTS = 5
EPOCHS = 1
BATCH_SIZE = 64
EPSILONS = [0.0, 0.1, 0.3, 0.5]  # Niveaux de bruit
DEFENSES = ["dp", "clip", "confreg", "secureagg", "dp+confreg"]
CLIP_NORM = 1.0
CONFIDENCE_REG_WEIGHT = 0.5

def seed_all(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_all()

# === Modèle de Base ===
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x)

# === Données MNIST ===
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

def split_data(dataset, num_clients):
    indices = np.random.permutation(len(dataset))
    split = np.array_split(indices, num_clients)
    return [Subset(dataset, s) for s in split]

clients = split_data(train_data, NUM_CLIENTS)
test_loader = DataLoader(test_data, batch_size=128)

# === Entraînement Local avec options de défense ===
def train(model, loader, defense, epsilon):
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        for x, y in loader:
            opt.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)

            if defense in ["confreg", "dp+confreg"]:
                softmax_preds = torch.softmax(preds, dim=1)
                confidence = torch.max(softmax_preds, dim=1)[0]
                reg_term = CONFIDENCE_REG_WEIGHT * confidence.mean()
                loss += reg_term

            loss.backward()

            if defense == "clip":
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            if defense in ["dp", "dp+confreg"]:
                for param in model.parameters():
                    param.grad += torch.randn_like(param.grad) * epsilon

            opt.step()
    return model.state_dict()

# === Agrégation Moyenne ou Secure ===
def aggregate(models, defense):
    if defense == "secureagg":
        noise_scale = 0.1
        noisy_models = []
        for m in models:
            noisy_m = {k: v + torch.randn_like(v) * noise_scale for k, v in m.items()}
            noisy_models.append(noisy_m)
        models = noisy_models

    avg = {}
    for k in models[0]:
        avg[k] = sum(m[k] for m in models) / len(models)
    return avg

# === Évaluation ===
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# === MIA Basée sur la Perte ===
def mia(model, member_x, member_y, nonmember_x, nonmember_y):
    crit = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        loss_m = crit(model(member_x), member_y)
        loss_nm = crit(model(nonmember_x), nonmember_y)
    threshold = (loss_m.mean() + loss_nm.mean()) / 2
    tpr = (loss_m < threshold).float().mean().item()
    fpr = (loss_nm < threshold).float().mean().item()
    return tpr, fpr

# === Simulation FL ===
def federated_learning(defense, epsilon):
    global_model = MLP()
    updates = []
    for client_data in clients:
        model = MLP()
        model.load_state_dict(global_model.state_dict())
        loader = DataLoader(client_data, batch_size=BATCH_SIZE, shuffle=True)
        weights = train(model, loader, defense=defense, epsilon=epsilon)
        updates.append(weights)
    global_model.load_state_dict(aggregate(updates, defense))
    return global_model

# === Expérimentations croisées ===
results = []
member_batch = next(iter(DataLoader(clients[0], batch_size=64)))
nonmember_batch = next(iter(test_loader))

for defense in DEFENSES:
    for eps in EPSILONS:
        model = federated_learning(defense=defense, epsilon=eps)
        acc = evaluate(model, test_loader)
        tpr, fpr = mia(model, *member_batch, *nonmember_batch)
        results.append({"defense": defense, "epsilon": eps, "accuracy": acc, "tpr": tpr, "fpr": fpr})

# === Graphique (version lisible) ===
metrics = ["accuracy", "tpr", "fpr"]
colors = cm.get_cmap('tab10')
plt.figure(figsize=(12, 8))
for i, metric in enumerate(metrics):
    plt.subplot(3, 1, i+1)
    for j, defense in enumerate(DEFENSES):
        y = [r[metric] for r in results if r["defense"] == defense]
        x = EPSILONS
        plt.plot(x, y, marker='o', label=defense, color=colors(j))
    plt.title(metric.upper())
    plt.ylabel("Valeur")
    if i == 2:
        plt.xlabel("Epsilon (bruit)")
    plt.grid(True)
    if i == 0:
        plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.2), loc='upper center')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle("Comparaison des défenses (par métrique)", fontsize=14)
plt.show()

# === Suggestion automatique de combinaison optimale (post-analyse) ===
best_combo = max(results, key=lambda x: x['accuracy'] - (x['tpr'] - x['fpr']))
print("\n>>> Suggestion de combinaison efficace :")
print("Défense:", best_combo['defense'], "| Epsilon:", best_combo['epsilon'])
