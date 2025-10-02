import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# ------------------ 1. Residual Block Definition ------------------
class Residual(nn.Module):
    """Core of ResNet - Residual Block"""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            # 1x1 convolution to match the channel number and dimension of the shortcut
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        # Main path
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        # Shortcut path
        if self.conv3:
            X = self.conv3(X)
        # Merge main path and shortcut
        Y += X
        return F.relu(Y)

# ------------------ 2. ResNet Model Definition ------------------
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Initial convolution layer
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 4 residual stages
        self.b2 = self._make_layer(64, 64, 2, first_block=True)
        self.b3 = self._make_layer(64, 128, 2)
        self.b4 = self._make_layer(128, 256, 2)
        self.b5 = self._make_layer(256, 512, 2)
        
        # Global average pooling + classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    
    def _make_layer(self, input_channels, num_channels, num_residuals, first_block=False):
        """Build one residual stage (e.g. b2, b3)"""
        blk = []
        for i in range(num_residuals):
            # First residual block in each stage handles dimension change
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return nn.Sequential(*blk)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.classifier(x)
        return x

# ------------------ 3. Dataset Preparation ------------------
# Hyperparameters
lr, num_epochs, batch_size = 0.05, 10, 256
resize_dim = 96 # Resize FashionMNIST images to 96x96

# Data transform
transform = transforms.Compose([
    transforms.Resize(resize_dim),
    transforms.ToTensor(),
])

# Load FashionMNIST dataset
train_dataset = datasets.FashionMNIST(root="../data", train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root="../data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ------------------ 4. Training and Evaluation Logic ------------------
def evaluate_accuracy(model, data_loader, device):
    """Evaluate model accuracy on given dataset"""
    model.eval() # Switch to evaluation mode
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

def train_model(model, train_loader, test_loader, device, epochs, lr):
    """Full training loop"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # For ResNet, SGD with momentum is a classic and effective choice
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    print("Start Training...")
    for epoch in range(epochs):
        start_time = time.time()
        model.train() # Switch to training mode
        
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
            
        train_loss = total_loss / total
        train_acc = correct / total
        test_acc = evaluate_accuracy(model, test_loader, device)
        end_time = time.time()
        
        print(f"Epoch [{epoch + 1}/{epochs}]: "
              f"Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Test Acc {test_acc:.4f}, "
              f"Time: {end_time - start_time:.1f}s")
        
    print("Training Finished.")

# ------------------ 5. Main ------------------
if __name__ == '__main__':
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Instantiate model
    net = ResNet(num_classes=10)
    
    # (Optional) Check output shapes with input size 96x96
    X_check = torch.rand(size=(1, 1, resize_dim, resize_dim))
    for name, layer in net.named_children():
        X_check = layer(X_check)
        print(f"{name} output shape: {X_check.shape}")
        
    # Start training
    train_model(net, train_loader, test_loader, device, num_epochs, lr)
