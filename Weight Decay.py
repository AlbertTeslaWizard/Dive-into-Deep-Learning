import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# =======================
# Hyperparameters
# =======================
epochs = 100
lr = 0.01
batch_size = 5
num_inputs = 200   # Input dimension
num_train = 20     # Training set size
num_test = 100     # Test set size
true_w = torch.ones(num_inputs, 1) * 0.01
true_b = 0.05
weight_decay = 1   # L2 penalty coefficient

# Set random seed for reproducibility
torch.manual_seed(42)


# =======================
# Synthetic dataset
# =======================
def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, w.shape[0]))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y


def load_data(w, b, num_examples, is_train=True):
    """Return a data iterator (DataLoader)."""
    X, y = synthetic_data(w, b, num_examples)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


# =======================
# Model, loss and optimizer
# =======================
net = nn.Linear(num_inputs, 1)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)

train_loader = load_data(true_w, true_b, num_train)
test_loader = load_data(true_w, true_b, num_test, is_train=False)


# =======================
# Training and evaluation
# =======================
def train_epoch(net, data_loader, loss_fn, optimizer):
    """Train for one epoch."""
    net.train()
    for X, y in data_loader:
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def evaluate_loss(net, data_loader, loss_fn):
    """Evaluate average loss on a dataset."""
    net.eval()
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_loader:
            loss = loss_fn(net(X), y)
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
    return loss_sum / n


# =======================
# Main training loop
# =======================
def main():
    history_train, history_test = [], []

    for epoch in range(epochs):
        train_loss = train_epoch(net, train_loader, loss_fn, optimizer)
        test_loss = evaluate_loss(net, test_loader, loss_fn)

        # Record loss every 10 epochs (and at epoch 0)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            history_train.append((epoch, train_loss))
            history_test.append((epoch, test_loss))
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")

    # =======================
    # Plot results
    # =======================
    epochs_train, losses_train = zip(*history_train)
    epochs_test, losses_test = zip(*history_test)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs_train, losses_train, label="Train Loss")
    plt.plot(epochs_test, losses_test, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Weight Decay = {weight_decay}")
    plt.legend()
    plt.show()

    # Print L2 norm of learned weights
    print("L2 norm of learned weights:", net.weight.norm().item())


if __name__ == "__main__":
    main()
