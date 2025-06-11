import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset
import flwr as fl
from flwr.common import Context

# Device configuration: use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define transform: convert images to tensor and normalize pixel values
transform = transforms.Compose([
 transforms.ToTensor(), # convert image to PyTorch tensor
 transforms.Normalize((0.1307,), (0.3081,)) # normalize with MNIST mean and std
 ])

# Download and load the MNIST training and test sets
trainset = datasets.MNIST(root="./data", train=True, download=True,transform=transform)
testset = datasets.MNIST(root="./data", train=False, download=True,transform=transform)

# Create DataLoader for batching
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=1000, shuffle=False)

print(f"Loaded MNIST dataset with {len(trainset)} training examples and {len(testset)} test examples.")

# Define a simple CNN model for MNIST
class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) # 1 input channel (grayscale), 16 filters
      self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 32filters
      self.pool = nn.MaxPool2d(2, 2) # 2x2 max pooling
      self.fc1 = nn.Linear(32 * 7 * 7, 128) # 32 feature maps * 7x7 after pooling
      self.fc2 = nn.Linear(128, 10) # 10 output classes (digits 0-9)

    def forward(self, x):
     # Two convolutional layers with ReLU and pooling
     x = self.pool(F.relu(self.conv1(x)))
     x = self.pool(F.relu(self.conv2(x)))
     x = x.view(-1, 32 * 7 * 7) # flatten
     x = F.relu(self.fc1(x))
     x = self.fc2(x)
     return x
    
# Initialize the network and move to device
model = Net().to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop for a certain number of epochs
epochs = 5
for epoch in range(1, epochs+1):
    model.train() # set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad() # reset gradients
        outputs = model(images) # forward pass
        loss = loss_fn(outputs, labels) # compute loss
        loss.backward() # backpropagate
        optimizer.step() # update weights
        running_loss += loss.item()
        # compute training accuracy for this batch (optional)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = running_loss / len(trainloader)
    accuracy = correct / total
    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Training Accuracy = {accuracy*100:.2f}%")

model.eval() # set model to evaluation mode
y_true = []
y_pred = []
with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().detach().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.detach().numpy())

# Calculate accuracy
test_accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy (Centralized model): {test_accuracy*100:.2f}%")

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", display_labels=list(range(10)))
plt.show()

# Number of clients in our federated simulation
NUM_CLIENTS = 5

# Create a dictionary to hold indices for each client
clients_indices = {i: [] for i in range(NUM_CLIENTS)}
for idx, (_, label) in enumerate(trainset):
    # Assign index to a client based on the label
    if label in [0, 1]:
        clients_indices[0].append(idx)
    elif label in [2, 3]:
        clients_indices[1].append(idx)
    elif label in [4, 5]:
        clients_indices[2].append(idx)
    elif label in [6, 7]:
        clients_indices[3].append(idx)
    else:
        clients_indices[4].append(idx)
        
# Use the indices to create subset datasets for each client
client_datasets = []
for i in range(NUM_CLIENTS):
        client_datasets.append(Subset(trainset, clients_indices[i]))
        print(f"Client {i} has {len(clients_indices[i])} samples, labels: {set(trainset.targets[clients_indices[i]].detach().numpy())}")
        
class MNISTClient(fl.client.NumPyClient):
    def __init__(self, cid, train_data, test_data=None):
        self.cid = cid # client ID (string in Flower)
        self.trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
        self.testloader = DataLoader(test_data, batch_size=1000, shuffle=False) if test_data else None
        # Each client has its own model and optimizer
        self.model = Net().to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.02)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def get_parameters(self, config=None):
        """Return current model parameters as a list of numpy arrays."""
        self.model.eval()
        params = [param.cpu().detach().numpy() for param in self.model.parameters()]
        return params

    def fit(self, parameters, config=None):
        """Receive global model parameters, train on local data, return updated parameters."""
        # Load global weights into the local model
        for param, new_val in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_val, device=device)
            # Train for 1 epoch on local dataset
            self.model.train()
        for images, labels in self.trainloader:
            images, labels = images.to(device), labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
        # After training, get updated weights
        updated_params = [param.cpu().detach().numpy() for param in self.model.parameters()]
        # Optionally, we can return the number of examples used (for weighting) and any metrics
        num_samples = len(self.trainloader.dataset)
        return updated_params, num_samples, {}

    def evaluate(self, parameters, config=None):
        """Evaluate the current model on local test data (if provided)."""
        # Load global weights into model
        for param, new_val in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_val, device=device)
        self.model.eval()
        if not self.testloader:
            return 0.0, 0, {"accuracy": 0.0}
        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss += self.loss_fn(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        loss = loss / len(self.testloader)
        return loss, total, {"accuracy": accuracy}

# Create a global model and test DataLoader for server-side evaluation
global_model = Net().to(device)
global_testloader = DataLoader(testset, batch_size=1000, shuffle=False)

# Define a server-side evaluation function
acc_history = []  # to record accuracy each round

def evaluate_global(server_rounf,parameters, config):
    # Load parameters into the global model
    param_tensors = [torch.tensor(np.array(p), device=device) for p in parameters]
    for param, new_val in zip(global_model.parameters(), param_tensors):
        param.data = new_val

    # Compute test accuracy and loss on the global test set
    global_model.eval()
    total, correct = 0, 0
    loss = 0.0
    with torch.no_grad():
        for images, labels in global_testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = global_model(images)
            loss += loss_fn(outputs, labels).item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    loss = loss / len(global_testloader)
    acc_history.append(accuracy)
    print(f"Server-side evaluation - Round accuracy: {accuracy*100:.2f}%")

    # Return (loss, metrics) as required by Flower
    return loss, {"accuracy": accuracy}

# Define client creation function
# def client_fn(cid: str):
    # cid = int(cid)
    # return MNISTClient(cid, train_data=client_datasets[cid], test_data=testset)


def client_fn(context: Context):
    cid = int(context.node_config["partition-id"])  # ou "partition-id" selon la version
    return MNISTClient(cid, train_data=client_datasets[cid], test_data=testset).to_client()

# Configure the federated learning strategy (FedAvg) and run simulation
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,          # use all clients each round
    fraction_evaluate=0.0,     # centralized evaluation only
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_fn=evaluate_global  # use our server-side evaluation function
)

# Run the simulation for a few rounds
NUM_ROUNDS = 5
history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy
)

# After training, plot accuracy vs. rounds
rounds = range(1, len(acc_history) + 1)
plt.figure()
plt.plot(rounds, [a * 100 for a in acc_history], marker='o')
plt.title("Federated Learning: Accuracy vs Rounds")
plt.xlabel("Round")
plt.ylabel("Test Accuracy (%)")
plt.grid(True)
plt.show()

# Compute final confusion matrix for federated model
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
    y_true_fed, y_pred_fed, cmap="Blues", display_labels=list(range(10))
)
plt.title("Federated Model Confusion Matrix")
plt.show()


print(f"Final Federated Model Test Accuracy: {acc_history[-1]*100:.2f}%")
