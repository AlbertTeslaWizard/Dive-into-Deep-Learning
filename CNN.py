import torch
from torch import nn

# -----------------------------
# Helper function: 2D cross-correlation
# -----------------------------
def corr2d(X, K):
    """
    Compute 2D cross-correlation between input X and kernel K.
    
    Args:
        X (Tensor): Input tensor of shape (H, W)
        K (Tensor): Kernel tensor of shape (h, w)
    
    Returns:
        Tensor: Output tensor of shape (H-h+1, W-w+1)
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

# -----------------------------
# Create input data
# -----------------------------
# Input matrix with a vertical stripe of zeros
X = torch.ones((6, 8))
X[:, 2:6] = 0

# Define a simple kernel
K = torch.tensor([[1.0, -1.0]])

# Compute target output using manual 2D correlation
Y = corr2d(X, K)

# -----------------------------
# Define Conv2D layer
# -----------------------------
conv2d = nn.Conv2d(
    in_channels=1,       # single input channel
    out_channels=1,      # single output channel
    kernel_size=(1, 2),  # same size as manual kernel
    bias=False           # no bias term
)

# Reshape input and target to 4D tensors: [batch, channels, height, width]
X = X.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 6, 8]
Y = Y.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 6, 7]

# -----------------------------
# Training settings
# -----------------------------
lr = 3e-2
num_epochs = 10

# Optimizer
optimizer = torch.optim.SGD(conv2d.parameters(), lr=lr)

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(1, num_epochs + 1):
    # Forward pass
    Y_hat = conv2d(X)
    
    # Compute MSE loss
    loss = (Y_hat - Y) ** 2
    
    # Backward pass and optimization
    optimizer.zero_grad()      # clear previous gradients
    loss.sum().backward()      # compute gradients
    optimizer.step()           # update parameters
    
    # Print loss every epoch
    print(f'Epoch {epoch}, loss {loss.sum():.5f}')

# -----------------------------
# Print learned kernel
# -----------------------------
print("Learned kernel:\n", conv2d.weight.data.reshape((1, 2)))
