import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ---------------- Improved LeNet Model ----------------
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(
            # Convolutional layer 1
            nn.Conv2d(1, 32, kernel_size=5, padding=2),  # Input channels=1, output channels=32
            nn.BatchNorm2d(32),  # <-- Optimization: Add Batch Normalization
            nn.ReLU(),            # Activation
            nn.MaxPool2d(2, 2),   # <-- Optimization: Max pooling with kernel_size=2, stride=2

            # Convolutional layer 2
            nn.Conv2d(32, 64, kernel_size=5),  # Input channels=32, output channels=64
            nn.BatchNorm2d(64),  # <-- Optimization: Add Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # <-- Optimization: Max pooling

            # Fully connected layers
            nn.Flatten(),          # Flatten feature maps to a vector
            nn.Linear(64 * 5 * 5, 120),
            nn.BatchNorm1d(120),  # <-- Optimization: BatchNorm for fully connected layer
            nn.ReLU(),
            nn.Dropout(0.3),       # <-- Optimization: Dropout with 0.3 probability

            nn.Linear(120, 84),
            nn.BatchNorm1d(84),   # <-- Optimization: BatchNorm
            nn.ReLU(),
            nn.Dropout(0.3),       # <-- Optimization: Dropout

            nn.Linear(84, 10)     # Output layer (10 classes)
        )

    def forward(self, x):
        return self.net(x)


# ---------------- Accuracy Evaluation Function ----------------
def evaluate_accuracy(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    correct, total = 0, 0
    with torch.no_grad():  # No gradient calculation for evaluation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = outputs.argmax(dim=1)  # Predicted class
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total  # Return accuracy


# ---------------- Training Function (with LR Scheduler) ----------------
def train(model, train_loader, test_loader, num_epochs, lr, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    weight_decay_rate = 0.0001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay_rate)
    
    # Optimization: Cosine Annealing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses, train_accs, test_accs = [], [], []

    print("Training started...")
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss, correct, total = 0.0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss and accuracy
            running_loss += loss.item() * X.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        
        # Update learning rate
        scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total
        test_acc = evaluate_accuracy(model, test_loader, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

    print("Training finished!")
    return train_losses, train_accs, test_accs


# ---------------- Main Script (with Data Augmentation) ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optimization: Data augmentation for training set
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),   # Random horizontal flip
        transforms.RandomRotation(10),       # Random rotation ±10 degrees
        transforms.ToTensor(),               # Convert to tensor
        transforms.Normalize((0.5,), (0.5,)) # Normalize to mean=0.5, std=0.5
    ])
    
    # Test set: no augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load datasets
    train_dataset = datasets.FashionMNIST(root="../data", train=True,
                                          transform=train_transform, download=True)
    test_dataset = datasets.FashionMNIST(root="../data", train=False,
                                         transform=test_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Initialize model
    model = LeNet()

    # Train
    train_losses, train_accs, test_accs = train(
        model, train_loader, test_loader,
        num_epochs=20,  # <-- Optimization: Increase number of epochs
        lr=0.001,
        device=device
    )

    # Plot training results
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label="Train Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-o', label="Train Acc")
    plt.plot(epochs, test_accs, 'r-o', label="Test Acc")
    plt.title("Training & Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
