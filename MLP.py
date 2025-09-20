import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hyperparameters
EPOCHS = 10
LEARNING_RATE = 0.1
BATCH_SIZE = 256
DATA_ROOT = '../data'


def initialize_weights(m):
    """
    Initialize weights for linear layers using normal distribution.
    
    Args:
        m (nn.Module): Neural network module
    """
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)


def create_model():
    """
    Create a simple feedforward neural network.
    
    Returns:
        nn.Sequential: Neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    return model


def load_data():
    """
    Load and preprocess Fashion-MNIST dataset.
    
    Returns:
        tuple: (train_loader, test_loader) data loaders
    """
    # Define data transformation
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Load datasets
    train_dataset = datasets.FashionMNIST(
        root=DATA_ROOT, 
        train=True, 
        transform=transform, 
        download=True
    )
    
    test_dataset = datasets.FashionMNIST(
        root=DATA_ROOT, 
        train=False, 
        transform=transform, 
        download=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    
    return train_loader, test_loader


def train_epoch(net, train_loader, loss_fn, optimizer):
    """
    Train model for one epoch.
    
    Args:
        net (nn.Module): Neural network model
        train_loader (DataLoader): Training data loader
        loss_fn (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
    
    Returns:
        tuple: (average_loss, accuracy) training metrics
    """
    net.train()
    total_examples, total_acc, total_loss = 0, 0, 0
    
    for X, y_true in train_loader:
        y_pred = net(X)
        loss = loss_fn(y_pred, y_true)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_examples += X.shape[0]
        total_acc += (y_pred.argmax(dim=1) == y_true).sum().item()
        total_loss += loss.item() * X.shape[0]
    
    avg_loss = total_loss / total_examples
    avg_acc = total_acc / total_examples
    
    return avg_loss, avg_acc


def evaluate_model(net, test_loader):
    """
    Evaluate model on test dataset.
    
    Args:
        net (nn.Module): Neural network model
        test_loader (DataLoader): Test data loader
    
    Returns:
        float: Accuracy on test dataset
    """
    net.eval()
    total_examples, total_acc = 0, 0
    
    with torch.no_grad():  # Disable gradient computation for evaluation
        for X, y_true in test_loader:
            y_pred = net(X)
            total_examples += X.shape[0]
            total_acc += (y_pred.argmax(dim=1) == y_true).sum().item()
    
    return total_acc / total_examples


def main():
    """
    Main training loop.
    """
    # Create model, loss function and optimizer
    net = create_model()
    net.apply(initialize_weights)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)
    
    # Load data
    train_loader, test_loader = load_data()
    
    # Training loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(net, train_loader, loss_fn, optimizer)
        test_acc = evaluate_model(net, test_loader)
        print(f'Epoch {epoch + 1:2d}/{EPOCHS}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, '
              f'Test Acc: {test_acc:.4f}')


if __name__ == '__main__':
    main()