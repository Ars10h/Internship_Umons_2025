# Installation nécessaire : pip install torch torchvision

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# === 1. Modèle simple ===
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Dropout(0.4),  # Régularisation simple
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x)

# === 2. Données MNIST ===
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# Diviser pour 2 clients
def split_data(dataset, num_clients=2):
    indices = np.random.permutation(len(dataset))
    return [Subset(dataset, indices[i::num_clients]) for i in range(num_clients)]

clients = split_data(train_data, 2)
test_loader = DataLoader(test_data, batch_size=128)

# === 3. Entraînement local ===
def train(model, loader, epochs=1):
    model.train()
    optim_ = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            optim_.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optim_.step()
    return model.state_dict()

# === 4. Agrégation moyenne ===
def aggregate(models):
    avg_state = {}
    for k in models[0]:
        avg_state[k] = sum(m[k] for m in models) / len(models)
    return avg_state

# === 5. Évaluation ===
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# === 6. Membership Inference Attack ===
def mia(model, member_x, member_y, nonmember_x, nonmember_y):
    criterion = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        loss_mem = criterion(model(member_x), member_y)
        loss_non = criterion(model(nonmember_x), nonmember_y)
    threshold = torch.quantile(loss_mem, 0.9)
    tpr = (loss_mem < threshold).float().mean().item()
    fpr = (loss_non < threshold).float().mean().item()
    return tpr, fpr

# === 7. Simuler FL + MIA ===
global_model = MLP()
client_models = []

for client_data in clients:
    model = MLP()
    model.load_state_dict(global_model.state_dict())
    loader = DataLoader(client_data, batch_size=64, shuffle=True)
    new_weights = train(model, loader, epochs=1)
    client_models.append(new_weights)

# Agréger
global_model.load_state_dict(aggregate(client_models))

# Évaluer
acc = evaluate(global_model, test_loader)
print("Accuracy globale:", acc)

# Données pour attaque
member_batch = next(iter(DataLoader(clients[0], batch_size=64)))
nonmember_batch = next(iter(test_loader))

# MIA : True Positive Rate et False Positive Rate
tpr, fpr = mia(global_model, *member_batch, *nonmember_batch)
print("TPR (MIA):", tpr)
print("FPR (MIA):", fpr)
