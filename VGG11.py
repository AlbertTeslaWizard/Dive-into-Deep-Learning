import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import List, Tuple

# Set device for training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Model Definition
# --------------------------------------------------------------------------------------

def vgg_block(num_convs: int, in_channels: int, out_channels: int) -> nn.Sequential:
    """Constructs a VGG block consisting of convolutional layers and a max pooling layer."""
    layers = []
    for _ in range(num_convs):
        # Convolutional layer (kernel=3, padding=1 maintains spatial size)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels  # Update input channels for the next layer
    
    # Max pooling layer (reduces spatial size by half)
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGG(nn.Module):
    """VGG network architecture implementation."""
    def __init__(self, conv_arch: List[Tuple[int, int]]):
        super().__init__()
        
        # 1. Build the convolutional blocks
        conv_blks = []
        in_channels = 1  # FashionMNIST is a single-channel grayscale image
        for num_convs, out_channels in conv_arch:
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        
        # 2. Define the overall network structure (Conv + Fully Connected)
        self.features = nn.Sequential(*conv_blks)
        
        # Calculate the size of the feature map after convolutional layers
        # Input image is 224x224. VGG uses 5 max-pooling layers, each halving the size:
        # 224 / (2^5) = 224 / 32 = 7. 
        # The number of channels is the last `out_channels` from the loop (which is `in_channels`)
        final_feature_size = in_channels * 7 * 7
        
        # 3. Define the fully connected layers (Classifier)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_feature_size, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 10) # 10 classes for FashionMNIST
        )
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        X = self.features(X)
        X = self.classifier(X)
        return X

## Data Preparation
# --------------------------------------------------------------------------------------

def load_data_fashion_mnist(batch_size: int, img_size: int) -> Tuple[DataLoader, DataLoader]:
    """Load and preprocess the FashionMNIST dataset."""
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), # Resize for VGG standard input
        transforms.ToTensor()
    ])

    # Load datasets
    train_dataset = datasets.FashionMNIST(root="../data", train=True, 
                                          transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root="../data", train=False, 
                                         transform=transform, download=True)

    # Create DataLoaders
    # num_workers=0 is safer for Windows environments
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    
    return train_loader, test_loader

## Training and Evaluation Functions
# --------------------------------------------------------------------------------------

def evaluate_accuracy(net: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """Computes the accuracy of the model on the provided data loader."""
    net.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            # Find the predicted class index (argmax) and compare with true labels
            correct += (net(X).argmax(dim=1) == y).sum().item()
            n += y.shape[0]
    return correct / n


def train(net: nn.Module, train_loader: DataLoader, test_loader: DataLoader, 
          optimizer: torch.optim.Optimizer, loss_fn: nn.Module, num_epochs: int, 
          device: torch.device):
    """Main training loop for the VGG network."""
    for epoch in range(num_epochs):
        net.train()
        train_loss, train_acc, n = 0.0, 0.0, 0
        
        # Training phase
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            y_hat = net(X)
            loss = loss_fn(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y.shape[0]
            train_acc += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

        # Evaluation phase
        test_acc = evaluate_accuracy(net, test_loader, device)
        print(f"epoch {epoch+1}, loss {train_loss/n:.4f}, "
              f"train acc {train_acc/n:.3f}, test acc {test_acc:.3f}")

## Execution
# --------------------------------------------------------------------------------------

if __name__ == '__main__':
    # Hyperparameters
    BATCH_SIZE = 128
    IMAGE_SIZE = 224
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.05
    
    # VGG-11 architecture (A configuration): (num_convs, out_channels)
    FULL_CONV_ARCH = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
    
    # Scale down channels for faster training/less memory usage (as in original code)
    RATIO = 4
    SMALL_CONV_ARCH = [(num_convs, out_channels // RATIO) for (num_convs, out_channels) in FULL_CONV_ARCH]

    # 1. Prepare Data
    train_loader, test_loader = load_data_fashion_mnist(BATCH_SIZE, IMAGE_SIZE)

    # 2. Define Network
    net = VGG(SMALL_CONV_ARCH).to(DEVICE)
    
    # 3. Setup Training Components
    LOSS_FN = nn.CrossEntropyLoss()
    OPTIMIZER = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)

    # 4. Start Training
    print(f"Starting training on ...")
    train(net, train_loader, test_loader, OPTIMIZER, LOSS_FN, NUM_EPOCHS, DEVICE)