import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# =========================
# Hyperparameters
# =========================
BATCH_SIZE, LR, EPOCHS = 256, 0.5, 10
NUM_INPUTS, NUM_HIDDENS1, NUM_HIDDENS2, NUM_OUTPUTS = 784, 256, 256, 10
DROPOUT1, DROPOUT2 = 0.2, 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# Dataset & DataLoader
# =========================
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root="../data", train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root="../data", train=False, transform=transform, download=True)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# Model definition
# =========================
class MLPDropout(nn.Module):
    """A simple MLP with Dropout regularization for FashionMNIST."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(NUM_INPUTS, NUM_HIDDENS1),
            nn.ReLU(),
            nn.Dropout(DROPOUT1),
            nn.Linear(NUM_HIDDENS1, NUM_HIDDENS2),
            nn.ReLU(),
            nn.Dropout(DROPOUT2),
            nn.Linear(NUM_HIDDENS2, NUM_OUTPUTS)
        )
        self.initialize_weights()

    def forward(self, X):
        return self.net(X)

    def initialize_weights(self):
        """Initialize weights with a small Gaussian distribution."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)


# =========================
# Training & Evaluation
# =========================
def evaluate_accuracy(model, dataloader):
    """Evaluate model accuracy on a given dataset."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X)
            correct += (preds.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total


def train(model, train_loader, test_loader, loss_fn, optimizer, epochs):
    """Train the model and evaluate after each epoch."""
    train_losses, test_accs = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X)
            loss = loss_fn(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        acc = evaluate_accuracy(model, test_loader)
        train_losses.append(avg_loss)
        test_accs.append(acc)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Acc: {acc:.4f}")

    return train_losses, test_accs


def plot_metrics(train_losses, test_accs):
    """Plot training loss and test accuracy curves."""
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    ax1.plot(range(1, len(train_losses)+1), train_losses, 'b-', label="Train Loss")
    ax2.plot(range(1, len(test_accs)+1), test_accs, 'r-', label="Test Accuracy")

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color="b")
    ax2.set_ylabel("Accuracy", color="r")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("Training Loss and Test Accuracy")
    plt.show()


# =========================
# Main execution
# =========================
if __name__ == "__main__":
    model = MLPDropout().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, test_accs = train(model, train_dataloader, test_dataloader, loss_fn, optimizer, EPOCHS)
    plot_metrics(train_losses, test_accs)
