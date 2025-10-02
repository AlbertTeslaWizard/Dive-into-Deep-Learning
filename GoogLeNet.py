import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# ---------------- Inception Block ----------------
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # Path 1: 1x1 Conv
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)

        # Path 2: 1x1 Conv -> 3x3 Conv
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        # Path 3: 1x1 Conv -> 5x5 Conv
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        # Path 4: 3x3 MaxPooling -> 1x1 Conv
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # Concatenate along the channel dimension
        return torch.cat((p1, p2, p3, p4), dim=1)


# ---------------- GoogLeNet ----------------
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten()
        )

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return self.fc(x)


# ---------------- Data Loading ----------------
batch_size = 128
transform = transforms.Compose([
    transforms.Resize(96),  # Resize images to 96x96
    transforms.ToTensor()
])

train_dataset = datasets.FashionMNIST(root="../data", train=True,
                                      transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root="../data", train=False,
                                     transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# ---------------- Evaluation Function ----------------
def evaluate(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


# ---------------- Training Function (with AdamW + CosineAnnealingLR) ----------------
def train(model, train_loader, test_loader, device, epochs=10, lr=0.001):
    """
    Train the model using AdamW optimizer with CosineAnnealingLR scheduler.
    - AdamW: More robust weight decay compared to Adam
    - CosineAnnealingLR: Cosine learning rate decay, helps convergence
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # AdamW optimizer (better regularization than Adam)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # CosineAnnealingLR scheduler:
    # T_max = number of iterations before LR restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("Start Training...")
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss, correct, total = 0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        # Update scheduler per epoch
        scheduler.step()

        # Compute metrics
        train_loss = total_loss / total
        train_acc = correct / total
        test_acc = evaluate(model, test_loader, device)
        end_time = time.time()

        print(f"Epoch [{epoch + 1}/{epochs}]: "
              f"Loss {train_loss:.4f}, "
              f"Train Acc {train_acc:.4f}, Test Acc {test_acc:.4f}, "
              f"LR {scheduler.get_last_lr()[0]:.6f}, "
              f"Time: {end_time - start_time:.1f}s")
    print("Training Finished.")


# ---------------- Main ----------------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    net = GoogLeNet(num_classes=10)
    # Train with AdamW + CosineAnnealingLR
    train(net, train_loader, test_loader, device, epochs=20, lr=0.001)
