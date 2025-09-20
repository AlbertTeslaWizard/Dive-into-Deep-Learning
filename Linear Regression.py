import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from d2l import torch as d2l


# -------------------------
# 1. Set true parameters and generate synthetic data
# -------------------------
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

batch_size = 10
lr = 0.03
num_epochs = 10


# -------------------------
# 2. Build dataset and data loader
# -------------------------
train_dataset = TensorDataset(features, labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# -------------------------
# 3. Define the model
# -------------------------
net = nn.Sequential(
    nn.Linear(2, 1)
)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)


# -------------------------
# 4. Define loss function and optimizer
# -------------------------
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr)


# -------------------------
# 5. Training loop
# -------------------------
for epoch in range(num_epochs):
    for X, y_true in train_loader:
        y_pred = net(X)
        loss = criterion(y_pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate on the full dataset
    train_loss = criterion(net(features), labels)
    print(f"epoch {epoch+1}, loss {train_loss:.6f}")


# -------------------------
# 6. Print the results
# -------------------------
print(f"\ntrue_w: {true_w}, learned_w: {net[0].weight.data.reshape(-1)}")
print(f"true_b: {true_b}, learned_b: {net[0].bias.data.item():.4f}")
