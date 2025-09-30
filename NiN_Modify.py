import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# ------------------ Define NiN Block (No changes here) ------------------
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    """
    Network in Network block: Conv -> ReLU -> 1x1 Conv -> ReLU -> 1x1 Conv -> ReLU
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(inplace=True)
    )

# ------------------ Define MODIFIED NiN Model ------------------
# This architecture is simplified and adapted for 28x28 images
class NiN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Input: (1, 28, 28)
            nin_block(1, 96, kernel_size=5, stride=1, padding=2), # Output: (96, 28, 28)
            nn.MaxPool2d(3, stride=2),                             # Output: (96, 13, 13)
            
            nin_block(96, 256, kernel_size=5, stride=1, padding=2), # Output: (256, 13, 13)
            nn.MaxPool2d(3, stride=2),                              # Output: (256, 6, 6)
            
            nin_block(256, 384, kernel_size=3, stride=1, padding=1), # Output: (384, 6, 6)
            nn.Dropout(0.5),
            
            # ---------------------- [ CORE MODIFICATION ] ----------------------
            # The final block maps features to the number of classes.
            # We use a standard Conv2d layer here instead of a nin_block to AVOID
            # applying a final ReLU activation to the output logits.
            # CrossEntropyLoss expects raw logits, not values processed by ReLU.
            nn.Conv2d(384, num_classes, kernel_size=3, stride=1, padding=1),#nn.Conv2d(384, num_classes, kernel_size=3, stride=1, padding=1), # Output: (10, 6, 6)
            # -------------------------------------------------------------------
            
            # Global Average Pooling replaces the final fully connected layers
            nn.AdaptiveAvgPool2d((1, 1))                           # Output: (10, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return x.flatten(1)  # Flatten to (batch_size, num_classes)

# ------------------ Instantiate Model ------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
net = NiN(num_classes=10).to(device)

# ------------------ Test Forward Pass (Optional, good for debugging) ------------------
# Input size is now 1x28x28
X = torch.rand(1, 1, 28, 28).to(device)
with torch.no_grad():
    y = net(X)
    print("Test forward pass output shape:", y.shape)


# ------------------ Prepare Data ------------------
batch_size = 128
transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(root='../data', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='../data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ------------------ Training Loop (No changes here) ------------------
def train_model(model, train_loader, test_loader, num_epochs, lr, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_hat = model(X_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            total_correct += (y_hat.argmax(dim=1) == y_batch).sum().item()
            total_samples += X_batch.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # Evaluate on test set
        model.eval()
        test_correct, test_samples = 0, 0
        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                y_pred = model(X_test)
                test_correct += (y_pred.argmax(dim=1) == y_test).sum().item()
                test_samples += X_test.size(0)

        test_acc = test_correct / test_samples
        scheduler.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# ------------------ Train the Model ------------------
train_model(net, train_loader, test_loader, num_epochs=20, lr=0.001, device=device)