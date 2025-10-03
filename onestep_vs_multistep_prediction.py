import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# --- 1. Generate Time Series Data ---
# Define the total number of time steps
T = 1000
# Create a time vector from 1 to T
time = torch.arange(1, T + 1, dtype=torch.float32)
# Generate synthetic time series data: a sine wave plus some Gaussian noise
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))


# --- 2. Create Feature-Label Pairs using a Sliding Window ---
# `tau` is the embedding dimension: we use `tau` past values to predict the next value.
tau = 4
# `features` will store the input sequences. We stack shifted versions of the time series.
# Example: features[0] will be [x[0], x[1], x[2], x[3]]
#          features[1] will be [x[1], x[2], x[3], x[4]]
features = torch.stack([x[i : T - tau + i] for i in range(tau)], dim=1)
# `labels` are the target values we want to predict.
# The label for features[0] (i.e., x[0:4]) is x[4].
labels = x[tau:].reshape(-1, 1)


# --- 3. Set up the DataLoader for Training ---
# Define batch size and the number of samples for training
batch_size, n_train = 16, 600
# Create a dataset from the first `n_train` feature-label pairs
train_dataset = TensorDataset(features[:n_train], labels[:n_train])
# Create a DataLoader to handle batching and shuffling of the training data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# --- 4. Define the Network Model ---
# A simple Multi-Layer Perceptron (MLP)
net = nn.Sequential(
    nn.Linear(tau, 10),  # Input layer: `tau` features, 10 hidden units
    nn.ReLU(),
    nn.Linear(10, 1)     # Output layer: 10 hidden units, 1 output value
)


# --- 5. Define Loss Function and Optimizer ---
# Use Mean Squared Error (MSE) loss, suitable for regression tasks
criterion = nn.MSELoss()
# Use the Adam optimizer to update network weights
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)


# --- 6. Training Loop ---
epochs = 5
print("--- Starting Training ---")
# Set the model to training mode
net.train()
for epoch in range(epochs):
    total_loss = 0
    for X, y in train_loader:
        optimizer.zero_grad()    # Clear previous gradients
        y_hat = net(X)           # Forward pass: get predictions
        loss = criterion(y_hat, y) # Calculate loss
        loss.backward()          # Backward pass: compute gradients
        optimizer.step()         # Update model parameters
        # Accumulate the loss for this epoch
        total_loss += loss.item() * X.shape[0]
    
    # Calculate and print the average loss for the epoch
    avg_loss = total_loss / n_train
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
print("--- Training Complete ---")


# --- Switch to Evaluation Mode ---
# This is important for layers like Dropout or BatchNorm, though not strictly necessary for this simple MLP.
net.eval()


# --- 7. One-Step-Ahead Predictions ---
# These predictions are made using the ground truth data as input for each step.
# This shows how well the model has learned the underlying pattern.
onestep_preds = net(features)


# --- 8. Multi-Step-Ahead Predictions (Iterative Forecasting) ---
# Here, the model uses its own previous predictions to forecast future values.
# This simulates a real-world scenario where future ground truth is unavailable.
multistep_preds = torch.zeros(T)
# Initialize with the ground truth data up to the end of the training period
multistep_preds[:n_train + tau] = x[:n_train + tau]

# Iteratively generate predictions for the rest of the time series
for i in range(n_train + tau, T):
    # The input sequence is the last `tau` values from our `multistep_preds` array
    input_seq = multistep_preds[i - tau:i].reshape(1, -1)
    # Use the model to predict the next value and store it
    multistep_preds[i] = net(input_seq)


# --- 9. k-Step-Ahead Predictions Visualization ---
# This section demonstrates how prediction error accumulates as we predict further into the future (increasing k).
max_steps = 64
# `k_step_features` will store sequences and their multi-step predictions
k_step_features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))

# Fill the first `tau` columns with the ground truth observations
for i in range(tau):
    k_step_features[:, i] = x[i: i + T - tau - max_steps + 1]

# Iteratively generate predictions for k = 1 to max_steps
for i in range(tau, tau + max_steps):
    # Use the window of `tau` previous values (which could be observations or predictions)
    # to make the next prediction.
    preds = net(k_step_features[:, i - tau:i])
    # Store the predictions in the next column
    k_step_features[:, i] = preds.reshape(-1)

# Define which prediction steps (k) we want to visualize
steps = (1, 4, 16, 64)


# --- 10. Plotting the k-Step-Ahead Predictions ---
plt.figure(figsize=(10, 6))
# Plot the original data for reference
plt.plot(time.numpy(), x.numpy(), label='Original Data', color='gray', alpha=0.7)

# Plot the predictions for each specified step k
for i in steps:
    # Determine the correct time axis for the k-step predictions
    time_steps = time[tau + i - 1 : T - max_steps + i]
    # Get the corresponding predictions from our matrix. Column (tau + i - 1) holds the i-step predictions.
    predictions = k_step_features[:, (tau + i - 1)].detach().numpy()
    
    plt.plot(time_steps, predictions, label=f'{i}-step preds')

# Formatting the plot
plt.xlabel("Time")
plt.ylabel("Value")
plt.xlim(5, 1000)
plt.legend()
plt.title("k-Step-Ahead Predictions")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()