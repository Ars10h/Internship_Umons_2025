import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

# Calculate accuracy
test_accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy (Centralized model): {test_accuracy*100:.2f}%")

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", display_labels=list(range(10)))
plt.show()