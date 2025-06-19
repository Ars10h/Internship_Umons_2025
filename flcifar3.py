import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import flwr as fl
from flwr.common import Context

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Loss function (used in global evaluation)
loss_fn = nn.CrossEntropyLoss()

# Define transform for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Create DataLoader for testing
testloader = DataLoader(testset, batch_size=1000, shuffle=False)

# CIFAR-10 class names
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']

# Simple CNN for CIFAR-10
class CIFARNet(nn.Module):
    def __init__(self):
        super(CIFARNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Partition data among clients
NUM_CLIENTS = 5
client_indices = {i: [] for i in range(NUM_CLIENTS)}
for idx, (_, label) in enumerate(trainset):
    client_indices[label % NUM_CLIENTS].append(idx)

client_datasets = [Subset(trainset, client_indices[i]) for i in range(NUM_CLIENTS)]
for i in range(NUM_CLIENTS):
    print(f"Client {i} has {len(client_indices[i])} samples")

# Client class
def get_model_parameters(model):
    return [param.cpu().detach().numpy() for param in model.parameters()]

def set_model_params(model, params):
    for param, new_val in zip(model.parameters(), params):
        param.data = torch.tensor(new_val, device=device)

class CIFARClient(fl.client.NumPyClient):
    def __init__(self, cid, train_data, test_data=None):
        self.cid = cid
        self.trainloader = DataLoader(train_data, batch_size=16, shuffle=True)
        self.testloader = DataLoader(test_data, batch_size=1000, shuffle=False) if test_data else None
        self.model = CIFARNet().to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.loss_fn = nn.CrossEntropyLoss()

    def get_parameters(self, config=None):
        return get_model_parameters(self.model)

    def fit(self, parameters, config=None):
        set_model_params(self.model, parameters)
        self.model.train()
        for images, labels in self.trainloader:
            images, labels = images.to(device), labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
        return get_model_parameters(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config=None):
        set_model_params(self.model, parameters)
        self.model.eval()
        if not self.testloader:
            return 0.0, 0, {"accuracy": 0.0}
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss += self.loss_fn(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return loss / len(self.testloader), total, {"accuracy": correct / total}

# Global evaluation
global_model = CIFARNet().to(device)
global_testloader = DataLoader(testset, batch_size=1000, shuffle=False)
acc_history = []

def evaluate_global(server_round, parameters, config):
    set_model_params(global_model, parameters)
    global_model.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in global_testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = global_model(images)
            loss += loss_fn(outputs, labels).item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    acc_history.append(accuracy)
    print(f"Server-side evaluation - Round accuracy: {accuracy*100:.2f}%")
    return loss / len(global_testloader), {"accuracy": accuracy}

def client_fn(context: Context):
    cid = int(context.node_config["partition-id"])
    return CIFARClient(cid, train_data=client_datasets[cid], test_data=testset).to_client()

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.0,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_fn=evaluate_global
)

NUM_ROUNDS = 5
history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources={"num_cpus": 2}
)

# Plot accuracy over rounds
rounds = range(1, len(acc_history) + 1)
plt.figure()
plt.plot(rounds, [a * 100 for a in acc_history], marker='o')
plt.title("Federated Learning on CIFAR-10: Accuracy vs Rounds")
plt.xlabel("Round")
plt.ylabel("Test Accuracy (%)")
plt.grid(True)
plt.show()

# Confusion matrix for final model
y_true_fed, y_pred_fed = [], []
global_model.eval()
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = global_model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_pred_fed.extend(preds)
        y_true_fed.extend(labels.cpu().numpy())

ConfusionMatrixDisplay.from_predictions(
    y_true_fed, y_pred_fed, cmap="Blues", display_labels=cifar10_labels
)
plt.title("Final Federated Model Confusion Matrix")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"Final Federated Model Test Accuracy: {acc_history[-1]*100:.2f}%")
