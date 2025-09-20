"""
Softmax Regression Implementation for FashionMNIST Classification
===============================================================

This script implements a Softmax regression model using PyTorch to classify
FashionMNIST images. The model uses a simple linear layer with CrossEntropyLoss
for multi-class classification.

Author: [Albert_Tesla]
Date: [2025/9/20]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 0.1
EPOCHS = 10

# Data paths
DATA_ROOT = "../data"


def load_data():
    """
    Load and preprocess FashionMNIST dataset.
    
    Returns:
        tuple: (train_loader, test_loader) Data loaders for training and testing
    """
    # Define data transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Load training and testing datasets
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


def initialize_weights(module):
    """
    Initialize weights for linear layers using normal distribution.
    
    Args:
        module (nn.Module): Neural network module to initialize
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.01)


def create_model():
    """
    Create Softmax regression model.
    
    Returns:
        nn.Sequential: Softmax regression model
    """
    model = nn.Sequential(
        nn.Flatten(),                    # Flatten 28x28 images to 784-dimensional vectors
        nn.Linear(28 * 28, 10)          # Linear transformation to 10 classes
    )
    
    # Apply weight initialization
    model.apply(initialize_weights)
    
    return model


def train_epoch(model, train_loader, criterion, optimizer):
    """
    Train model for one epoch.
    
    Args:
        model (nn.Module): Neural network model
        train_loader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimization algorithm
        
    Returns:
        tuple: (average_loss, accuracy) Training metrics
    """
    model.train()
    total_examples, total_correct, total_loss = 0, 0, 0
    
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        total_examples += inputs.shape[0]
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_loss += loss.item() * inputs.shape[0]
    
    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    
    return average_loss, accuracy


def evaluate_model(model, test_loader):
    """
    Evaluate model performance on test dataset.
    
    Args:
        model (nn.Module): Neural network model
        test_loader (DataLoader): Test data loader
        
    Returns:
        float: Test accuracy
    """
    model.eval()
    total_examples, total_correct = 0, 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            total_examples += inputs.shape[0]
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
    
    accuracy = total_correct / total_examples
    return accuracy


def main():
    """
    Main training loop for Softmax regression model.
    """
    # Load data
    train_loader, test_loader = load_data()
    
    # Create model, loss function and optimizer
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        
        # Evaluate on test set
        test_acc = evaluate_model(model, test_loader)
        
        # Print progress
        print(f'Epoch [{epoch + 1}/{EPOCHS}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, '
              f'Test Acc: {test_acc:.4f}')


if __name__ == '__main__':
    main()