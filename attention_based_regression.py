import torch
from torch import nn
import matplotlib.pyplot as plt

# --- 1. Data Generation ---

# Define the number of training samples.
n_train = 50
# Generate training data points (x_train) and sort them for easier visualization.
x_train, _ = torch.sort(torch.rand(n_train) * 5)

def f(x):
    """The true underlying function we want to learn."""
    return 2 * torch.sin(x) + x**0.8

# Generate training labels (y_train) by applying the true function
# and adding Gaussian noise to simulate real-world data.
y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))

# Generate test data points (x_test) for evaluating the model.
x_test = torch.arange(0, 5, 0.1)
# Generate the ground truth labels (y_truth) for the test data without noise.
y_truth = f(x_test)
n_test = len(x_test)


# --- 2. Model Definition ---

class NWKernelRegression(nn.Module):
    """
    Nadaraya-Watson Kernel Regression with a learnable bandwidth parameter.

    This model predicts the output for a query by computing a weighted average
    of the values in a dataset. The weights (attention) are determined by the
    similarity between the query and the keys, modeled by a Gaussian kernel.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize a learnable parameter 'w'. This parameter controls the
        # width (or bandwidth) of the Gaussian kernel. A larger 'w' leads to
        # more focused attention on closer keys.
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        """
        The forward pass computes the weighted average of 'values'.
        'queries', 'keys', and 'values' are expected to be 2D tensors
        where the first dimension is the batch size (number of queries).
        """
        # Expand 'queries' to match the shape of 'keys' for element-wise operations.
        # Original queries shape: (batch_size,)
        # Reshaped queries: (batch_size, num_keys) where each row repeats a query.
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))

        # Calculate attention scores using a Gaussian (RBF) kernel. The distance
        # between queries and keys is scaled by the learnable parameter 'w'.
        # The negative squared distance forms the basis of the kernel.
        # Shape: (batch_size, num_keys)
        unnormalized_weights = -((queries - keys) * self.w)**2 / 2

        # Apply softmax to the scores along the 'keys' dimension (dim=1)
        # to get normalized attention weights that sum to 1 for each query.
        # These weights represent how much attention to pay to each key-value pair.
        # Shape: (batch_size, num_keys)
        self.attention_weights = nn.functional.softmax(unnormalized_weights, dim=1)

        # Compute the final prediction by performing a batch matrix multiplication
        # between the attention weights and the values.
        # attention_weights shape: (batch_size, 1, num_keys)
        # values shape: (batch_size, num_keys, 1)
        # bmm result shape: (batch_size, 1, 1) --> reshaped to (batch_size,)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)


# --- 3. Training Preparation (Leave-One-Out Approach) ---

# To train the model, we use a leave-one-out approach. For each training
# point (x_train[i], y_train[i]), we treat x_train[i] as a query and all
# other points (x_train[j], y_train[j]) where j!=i as the key-value pairs.

# Create a tile of x_train to facilitate creating the keys.
# X_tile shape: (n_train, n_train), each row is a copy of x_train.
X_tile = x_train.repeat((n_train, 1))
# Create a tile of y_train to facilitate creating the values.
# Y_tile shape: (n_train, n_train), each row is a copy of y_train.
Y_tile = y_train.repeat((n_train, 1))

# Create a boolean mask to exclude the i-th element for the i-th row.
# `torch.eye(n_train)` is the identity matrix. `1 - eye` creates a matrix
# with zeros on the diagonal and ones elsewhere.
mask = (1 - torch.eye(n_train)).type(torch.bool)

# Select the keys using the mask. For each row `i`, this selects all
# x_train values except for x_train[i].
# Shape: (n_train, n_train - 1)
keys_train = X_tile[mask].reshape((n_train, -1))

# Select the values using the mask. For each row `i`, this selects all
# y_train values except for y_train[i].
# Shape: (n_train, n_train - 1)
values_train = Y_tile[mask].reshape((n_train, -1))


# --- 4. Training Loop ---

# Instantiate the model, loss function, and optimizer.
net = NWKernelRegression()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.5)
epochs = 10

print("Starting training...")
for epoch in range(epochs):
    optimizer.zero_grad()

    # The queries for training are the original training inputs.
    # The keys and values are the leave-one-out sets prepared earlier.
    # The model predicts y_train[i] using all other training data.
    y_hat_train = net(x_train, keys_train, values_train)

    # Calculate the Mean Squared Error loss between predictions and actuals.
    l = loss_fn(y_hat_train, y_train)
    
    # Backpropagate the loss and update the model's weight 'w'.
    l.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss {float(l):.6f}')


# --- 5. Prediction (on Test Set) ---

# After training, use the model's learned parameter 'w' to make predictions
# on the test set. Here, for each test query, the *entire* training set is used
# as the key-value pairs.

# Prepare the keys for the test set. Each test query will be compared against
# all training keys.
# keys_test shape: (n_test, n_train), each row is a copy of x_train.
keys_test = x_train.repeat((n_test, 1))

# Prepare the values for the test set.
# values_test shape: (n_test, n_train), each row is a copy of y_train.
values_test = y_train.repeat((n_test, 1))

# Make predictions on the test set. Use `torch.no_grad()` for inference.
with torch.no_grad():
    y_hat_test = net(x_test, keys_test, values_test)

# The attention weights for the test predictions are stored in the model.
# Detach from the computation graph for visualization.
attention_weights_test = net.attention_weights.detach()


# --- 6. Visualization ---

def plot_predictions(x_train, y_train, x_test, y_truth, y_hat):
    """Plots the training data, true function, and model predictions."""
    plt.figure(figsize=(10, 6))
    plt.plot(x_test, y_truth, label='True Function', linewidth=2)
    plt.plot(x_test, y_hat, label='Model Prediction', linewidth=2, linestyle='--')
    plt.plot(x_train, y_train, 'o', alpha=0.5, label='Training Data')
    plt.title('Nadaraya-Watson Kernel Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.xlim([0, 5])
    plt.ylim([-1, 5])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_heatmap(weights, xlabel, ylabel, xticks, yticks):
    """Plots a heatmap of the attention weights."""
    plt.figure(figsize=(8, 8))
    im = plt.imshow(weights, cmap='Reds', interpolation='none', aspect='auto',
                    origin='lower', extent=[xticks[0], xticks[-1], yticks[0], yticks[-1]])
    plt.title('Attention Heatmap')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(im, label='Attention Weight')
    plt.show()

# Plot the regression results.
plot_predictions(x_train, y_train, x_test, y_truth, y_hat_test)

# Plot the attention heatmap. The heatmap shows for each test input (y-axis),
# which training inputs (x-axis) received the most attention.
plot_heatmap(attention_weights_test,
             xlabel='Sorted training inputs (Keys)',
             ylabel='Sorted testing inputs (Queries)',
             xticks=x_train, yticks=x_test)