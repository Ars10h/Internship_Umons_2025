# Projet : Apprentissage Fédéré + MIA + Défenses Optimisées + ROC/AUC

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import roc_curve, auc

# === Paramètres Généraux ===
NUM_CLIENTS = 5
EPOCHS = 1
BATCH_SIZE = 64
EPSILONS = [0.0, 0.1, 0.3, 0.5]
# Raffinement : comparer uniquement les meilleures variantes
DEFENSES = ["confreg", "dp+confreg", "dp+confreg+secureagg"]
CLIP_NORM = 1.0
CONFIDENCE_REG_WEIGHT = 0.5
SECURE_NOISE = 0.1

# === Préparation ===
def seed_all(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_all()

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

transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

def split_data(dataset, num_clients):
    indices = np.random.permutation(len(dataset))
    split = np.array_split(indices, num_clients)
    return [Subset(dataset, s) for s in split]

clients = split_data(train_data, NUM_CLIENTS)
test_loader = DataLoader(test_data, batch_size=128)

# === Entraînement local ===
def train(model, loader, defense, epsilon):
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        for x, y in loader:
            opt.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            if "confreg" in defense:
                conf = torch.softmax(preds, dim=1).max(dim=1)[0]
                loss += CONFIDENCE_REG_WEIGHT * conf.mean()
            loss.backward()
            if "dp" in defense:
                for p in model.parameters():
                    p.grad += torch.randn_like(p.grad) * epsilon
            opt.step()
    return model.state_dict()

# === Agrégation ===
def aggregate(models, defense):
    if "secureagg" in defense:
        models = [
            {k: v + torch.randn_like(v) * SECURE_NOISE for k, v in m.items()}
            for m in models
        ]
    avg = {k: sum(m[k] for m in models) / len(models) for k in models[0]}
    return avg

# === Évaluation et attaque ===
def evaluate(model, loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def mia(model, member_x, member_y, nonmember_x, nonmember_y):
    crit = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        loss_m = crit(model(member_x), member_y)
        loss_nm = crit(model(nonmember_x), nonmember_y)
    threshold = (loss_m.mean() + loss_nm.mean()) / 2
    tpr = (loss_m < threshold).float().mean().item()
    fpr = (loss_nm < threshold).float().mean().item()
    return tpr, fpr

# === Entraînement fédéré raffiné ===
def federated_learning(defense, epsilon):
    global_model = MLP()
    updates = []
    for client_data in clients:
        model = MLP()
        model.load_state_dict(global_model.state_dict())
        loader = DataLoader(client_data, batch_size=BATCH_SIZE, shuffle=True)
        updated = train(model, loader, defense=defense, epsilon=epsilon)
        updates.append(updated)
    global_model.load_state_dict(aggregate(updates, defense))
    return global_model

results = []
member_batch = next(iter(DataLoader(clients[0], batch_size=64)))
nonmember_batch = next(iter(test_loader))

for defense in DEFENSES:
    for eps in EPSILONS:
        model = federated_learning(defense, eps)
        acc = evaluate(model, test_loader)
        tpr, fpr = mia(model, *member_batch, *nonmember_batch)
        results.append({"defense": defense, "epsilon": eps, "accuracy": acc, "tpr": tpr, "fpr": fpr, "model": model})

# === Affichage résultats ===
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
plt.suptitle("Défenses optimisées contre MIA", fontsize=14)
plt.show()

# === Courbes ROC/AUC ===
def compute_roc_auc(model, member_x, member_y, nonmember_x, nonmember_y):
    crit = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        member_loss = crit(model(member_x), member_y).cpu().numpy()
        nonmember_loss = crit(model(nonmember_x), nonmember_y).cpu().numpy()
    scores = np.concatenate([-member_loss, -nonmember_loss])
    labels = np.concatenate([np.ones_like(member_loss), np.zeros_like(nonmember_loss)])
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr, tpr, auc(fpr, tpr)

plt.figure(figsize=(8, 6))
for j, defense in enumerate(DEFENSES):
    model = next(r['model'] for r in results if r['defense'] == defense and r['epsilon'] == 0.3)
    fpr, tpr, roc_auc = compute_roc_auc(model, *member_batch, *nonmember_batch)
    plt.plot(fpr, tpr, label=f"{defense} (AUC={roc_auc:.2f})", color=colors(j))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("Taux de Faux Positifs")
plt.ylabel("Taux de Vrais Positifs")
plt.title("Courbe ROC - MIA (ε = 0.3)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Suggestion automatique finale ===
best_combo = max(results, key=lambda x: x['accuracy'] - (x['tpr'] - x['fpr']))
print("\n>>> Suggestion optimale :")
print("Défense:", best_combo['defense'], "| Epsilon:", best_combo['epsilon'])
