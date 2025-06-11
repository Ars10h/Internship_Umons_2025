
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import roc_auc_score
import random

# ----------- CONFIG -----------
NUM_CLIENTS = 10
ROUNDS = 20
EPOCHS = 3
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ----------- MODEL -----------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
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

# ----------- DATASET -----------
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

def partition_dataset(dataset, num_clients=NUM_CLIENTS, classes_per_client=2):
    class_indices = [[] for _ in range(10)]
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    client_indices = [[] for _ in range(num_clients)]
    for client_id in range(num_clients):
        chosen_classes = np.random.choice(range(10), classes_per_client, replace=False)
        for cls in chosen_classes:
            selected = np.random.choice(class_indices[cls], len(class_indices[cls]) // num_clients, replace=False)
            client_indices[client_id].extend(selected)
            class_indices[cls] = list(set(class_indices[cls]) - set(selected))
    return client_indices

client_data_indices = partition_dataset(train_set)

# ----------- FEDERATED TRAINING -----------
def train_local(model, dataloader, epochs=EPOCHS, lr=0.01):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def aggregate_models(global_model, client_models):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([client_models[i][key].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

global_model = SimpleCNN().to(DEVICE)
for r in range(ROUNDS):
    client_models = []
    for cid in range(NUM_CLIENTS):
        local_model = SimpleCNN().to(DEVICE)
        local_model.load_state_dict(global_model.state_dict())
        loader = DataLoader(Subset(train_set, client_data_indices[cid]), batch_size=BATCH_SIZE, shuffle=True)
        updated = train_local(local_model, loader)
        client_models.append(updated)
    global_model = aggregate_models(global_model, client_models)
    print(f"Completed round {r + 1}")

# ----------- MIA ATTACK -----------
def mia_attack(model, member_loader, non_member_loader):
    model.eval()
    member_confidences = []
    non_member_confidences = []
    with torch.no_grad():
        for x, y in member_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            probs = F.softmax(out, dim=1)
            conf = probs[range(len(y)), y]
            member_confidences.extend(conf.cpu().numpy())

        for x, y in non_member_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            probs = F.softmax(out, dim=1)
            conf = probs[range(len(y)), y]
            non_member_confidences.extend(conf.cpu().numpy())

    labels = [1] * len(member_confidences) + [0] * len(non_member_confidences)
    scores = member_confidences + non_member_confidences
    auc = roc_auc_score(labels, scores)
    return auc

member_loader = DataLoader(Subset(train_set, client_data_indices[0]), batch_size=BATCH_SIZE, shuffle=False)
non_member_indices = list(set(range(len(test_set))) - set(client_data_indices[0]))
non_member_loader = DataLoader(Subset(test_set, non_member_indices[:len(client_data_indices[0])]), batch_size=BATCH_SIZE, shuffle=False)

auc_score = mia_attack(global_model, member_loader, non_member_loader)
print(f"MIA AUC Score: {auc_score:.4f}")
