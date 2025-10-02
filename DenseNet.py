import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

# ---------------- 1. DenseNet components ----------------
def conv_block(input_channels, num_channels):
    """BN -> ReLU -> Conv"""
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
    """
    A dense block consisting of multiple conv_blocks.
    The output of each conv_block is concatenated with its input along the channel dimension.
    """
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            # Calculate the input channels for each conv_block
            in_channels = input_channels + i * num_channels
            layer.append(conv_block(in_channels, num_channels))
        # Use ModuleList instead of Sequential so that PyTorch tracks each submodule properly
        self.net = nn.ModuleList(layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate input and output along the channel dimension
            X = torch.cat((X, Y), dim=1)
        return X

def transition_block(input_channels, num_channels):
    """
    Transition layer between dense blocks.
    Reduces the number of channels using a 1x1 convolution,
    and halves the feature map resolution using average pooling.
    """
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))

# ---------------- 2. DenseNet model ----------------
class DenseNet(nn.Module):
    def __init__(self, num_convs_in_dense_blocks, growth_rate, num_classes=10):
        super().__init__()
        # Initial convolution layer
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dynamically create DenseBlocks and TransitionBlocks
        num_channels = 64
        features = []
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            features.append(DenseBlock(num_convs, num_channels, growth_rate))
            # Update channel count
            num_channels += num_convs * growth_rate
            # Add TransitionBlock after each DenseBlock except the last
            if i != len(num_convs_in_dense_blocks) - 1:
                features.append(transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
        
        self.features = nn.Sequential(*features)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(num_channels), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_channels, num_classes)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------- 3. Data loading ----------------
batch_size = 256
transform = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor()
])
train_dataset = datasets.FashionMNIST(root="../data", train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root="../data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ---------------- 4. Training and evaluation ----------------
def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

def train_model(model, train_loader, test_loader, device, epochs, lr):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # --- Core Change 1 ---
    # Use AdamW optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # --- Core Change 2 ---
    # Cosine Annealing learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("Start Training...")
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * y.size(0)
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)
            
        # --- Core Change 3 ---
        # Update learning rate after each epoch
        scheduler.step()
        
        train_loss = total_loss / total
        train_acc = correct / total
        test_acc = evaluate_accuracy(model, test_loader, device)
        end_time = time.time()
        
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}, "
              f"Time: {end_time - start_time:.1f}s")
    print("Training Finished.")

# ---------------- 5. Run training ----------------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model hyperparameters
    growth_rate = 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]  # Similar to DenseNet-121
    
    # Instantiate the model
    net = DenseNet(num_convs_in_dense_blocks, growth_rate, num_classes=10)
    
    # Training hyperparameters (better suited for AdamW + Cosine Annealing)
    lr, num_epochs = 0.001, 10
    
    # Train the model
    train_model(net, train_loader, test_loader, device, num_epochs, lr)
